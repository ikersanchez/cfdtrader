"""Auditoría de los costes declarados del ``SPX500:CFD`` (tarea #8).

Este módulo **mide y verifica**; **no** implementa el modelo de coste del motor de
*backtest* — eso es la tarea **#11** (``src/cfdtrader/backtest/costs.py``). Esa
frontera existe para que no haya dos definiciones divergentes de «coste»: aquí se
reproduce la tabla **declarada** de ``plan.md`` §3.3 y se consolida lo que un humano
anota a mano en ``config/cost_observations.yaml``.

**Qué se mide.** La tabla declarada del documento del bróker se reproduce con
aritmética exacta (``decimal.Decimal``): diferencial 0,42 $ / 0,0042 %, tenencia
-0,18 $ / -0,0018 % por noche en corto, +1,82 $ / +0,0182 % por noche en largo,
cambio de divisa 0,00 %, y los totales de ida y vuelta **0,24 $** (corto) y
**2,24 $** (largo) sobre 10.000 $ de nocional. Se consolida además el **spread
cotizado** por tramo de sesión y por tamaño a partir de observaciones anotadas a
mano, y el *tracking difference* cuando hay pares CFD/índice en el mismo instante.

**Lo que no se puede medir, y el informe declara en vez de rellenar:**

- ***Slippage* de ejecución**: a 2026-09-18 **no existe ninguna ejecución real**, así
  que sale ``state: "unmeasured"``, ``value: null``, con motivo y con el
  procedimiento de ``plan.md`` §8.5 para rellenarlo. Prohibido cualquier valor de
  relleno: ni constante provisional, ni un ``0`` con motivo, ni un rango inventado.
- **Corte de financiación**: **sigue sin verificar**. Vale ``null`` con
  ``state: "unverified"`` y con la pregunta literal al bróker. Asumir una hora de
  corte fija está prohibido: si el corte cae antes del cierre, el intradía puro
  pagaría tenencia igualmente.
- **Tracking difference**: solo con pares CFD/índice del mismo instante (tolerancia
  declarada ≤ 1 s); **nunca** se aproxima con una media diaria.

**Las tres medidas nunca se fusionan.** ``spread_cotizado``, ``tracking_difference``
y ``slippage_ejecucion`` son tres bloques de primer nivel, cada uno con su ``state``,
su ``value``, su ``reason`` y su ``source``, y **no existe ningún campo que las
sume** en un «coste total» único: sumarlas sería inventar una magnitud que nadie ha
medido.

Fuente de verdad funcional: ``plan.md`` §3.3 (tabla declarada) y §8.5 (mediciones
previas obligatorias de la Fase 0). La **ventana de cotización** (sesión regular del
S&P 500, derivada en ``America/New_York``) y la **divisa de liquidación** (USD)
están confirmadas por el usuario el 2026-09-18 y se registran con su procedencia; el
**corte de financiación no**, y no se inventa.

**Regla que gobierna todo el artefacto: ``null`` es «no medido» y nunca se sustituye
por ``0``.** Un cero solo se admite con ``state: "measured"``, su fuente y su motivo.
Los importes se serializan como cadenas decimales exactas, no como ``float``: la
comparación de la tabla declarada es exacta, no aproximada.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from cfdtrader.data.calendar import MarketCalendar, SessionInfo, load_calendar
from cfdtrader.data.settings import ConfigurationError, load_settings

__all__ = [
    "ANNUALISATION_TOLERANCE",
    "BROKER_QUESTIONS",
    "CARRY_LONG_PCT_PER_NIGHT",
    "CARRY_LONG_USD_PER_NIGHT",
    "CARRY_SHORT_PCT_PER_NIGHT",
    "CARRY_SHORT_USD_PER_NIGHT",
    "DEFAULT_TEMPLATE_PATH",
    "HOLDING_NIGHTS",
    "LIMITATIONS",
    "PAIR_TOLERANCE_SECONDS",
    "REFERENCE_NOTIONAL_USD",
    "SESSION_TRANCHES",
    "SIZE_LADDER_USD",
    "SPREAD_PCT",
    "SPREAD_USD",
    "CostAudit",
    "CostAuditError",
    "CostObservations",
    "CostTemplateError",
    "Execution",
    "FxCost",
    "MeasureState",
    "SessionPoint",
    "SessionTranche",
    "Side",
    "SpreadObservation",
    "TrackingPair",
    "TrancheSpec",
    "analyse",
    "consolidate",
    "load_template",
    "main",
    "render_markdown",
    "report_payload",
    "session_tranches",
    "tranche_of",
]

# ─────────────────────────────────────────────────────────────────────────────
# La tabla declarada (`plan.md` §3.3), como constantes con nombre y unidad
# ─────────────────────────────────────────────────────────────────────────────
#: Nocional de referencia de toda la tabla declarada, en dólares.
REFERENCE_NOTIONAL_USD: Final[Decimal] = Decimal("10000")

#: Nociones de la consolidación por tamaño (A12). Incluye el de referencia.
SIZE_LADDER_USD: Final[tuple[Decimal, ...]] = (
    Decimal("1000"),
    Decimal("5000"),
    Decimal("10000"),
)

#: Noches de la tabla declarada: tenencia de un día (intradía puro con una noche).
HOLDING_NIGHTS: Final[int] = 1

#: Diferencial declarado: 0,42 $ y 0,0042 % sobre 10.000 $ de nocional.
SPREAD_USD: Final[Decimal] = Decimal("0.42")
SPREAD_PCT: Final[Decimal] = Decimal("0.0042")

#: Tenencia declarada en CORTO: carry positivo, cobra 0,18 $ por noche.
CARRY_SHORT_USD_PER_NIGHT: Final[Decimal] = Decimal("-0.18")
CARRY_SHORT_PCT_PER_NIGHT: Final[Decimal] = Decimal("-0.0018")

#: Tenencia declarada en LARGO: paga 1,82 $ por noche (5× el diferencial).
CARRY_LONG_USD_PER_NIGHT: Final[Decimal] = Decimal("1.82")
CARRY_LONG_PCT_PER_NIGHT: Final[Decimal] = Decimal("0.0182")

#: Cambio de divisa declarado. Es 0 **con motivo** (liquidación en USD), no un cero mudo.
FX_COST_PCT: Final[Decimal] = Decimal("0.00")

#: Anualizado declarado (procedencia: documento del bróker + declaración del usuario).
#: **No se deriva** de las cifras diarias: se registra como declarado y se publica la
#: ratio implicada y si las dos ratios son coherentes entre sí.
ANNUALISED_SHORT_PCT: Final[Decimal] = Decimal("-0.6528")
ANNUALISED_LONG_PCT: Final[Decimal] = Decimal("6.6647")

#: Si las dos ratios implicadas difieren en más de esto, la anualización es incoherente.
ANNUALISATION_TOLERANCE: Final[Decimal] = Decimal("0.01")

#: Unidades que acompañan a todo importe (A10): sin unidad no hay cifra interpretable.
UNIT_USD: Final[str] = "$ sobre el nocional de referencia"
UNIT_PCT: Final[str] = "% del nocional"
UNIT_PCT_PER_NIGHT: Final[str] = "% del nocional por noche"
UNIT_PCT_ANNUAL: Final[str] = "% anual"
UNIT_POINTS: Final[str] = "puntos de índice"

#: Procedencias declaradas, para que ninguna cifra quede sin origen.
SOURCE_DECLARED_TABLE: Final[str] = "plan.md §3.3 (documento del bróker recogido en el plan)"
SOURCE_USER_DECLARATION: Final[str] = "declaración del usuario"
SOURCE_ANNUALISED: Final[str] = "plan.md §3.3 y declaración del usuario (2026-09-18)"

#: La ventana y la divisa están **confirmadas por el usuario**, no por el bróker.
WINDOW_CONFIRMED_ON: Final[date] = date(2026, 9, 18)
DECLARED_SETTLEMENT_CURRENCY: Final[str] = "USD"

#: El corte de financiación sigue sin verificar: **no hay valor por defecto** y asumir
#: una hora de corte fija está prohibido. La constante existe para poder afirmarlo en
#: un test.
DECLARED_FINANCING_CUT: Final[None] = None

#: Nota del *tracking difference*: no se mide sobre un nocional (A10).
TRACKING_NOTIONAL_NOTE: Final[str] = (
    "el tracking difference no se mide sobre un nocional: se mide sobre el precio del índice "
    "(puntos y % del índice), así que no tiene nocional propio"
)

#: Tolerancia de emparejamiento CFD/índice del *tracking difference* (A16).
PAIR_TOLERANCE_SECONDS: Final[int] = 1

#: Decimales con los que se publican las medias y las ratios (nada de ruido de float).
_MEAN_QUANTUM: Final[Decimal] = Decimal("0.00000001")
_RATIO_QUANTUM: Final[Decimal] = Decimal("0.0001")
_RELATIVE_QUANTUM: Final[Decimal] = Decimal("0.000001")

DEFAULT_TEMPLATE_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "config" / "cost_observations.yaml"
)

#: Anclajes de los tramos, **relativos a la sesión**: nunca una hora fija (A11).
TrancheAnchor = Literal["open", "close"]


@dataclass(frozen=True, slots=True)
class SessionPoint:
    """Un instante relativo a la sesión: ``-10`` minutos desde la apertura, etc."""

    anchor: TrancheAnchor
    offset_minutes: int


@dataclass(frozen=True, slots=True)
class TrancheSpec:
    """Definición de un tramo: instante de referencia y ventana, ambos relativos."""

    name: str
    label: str
    reference: SessionPoint
    window_start: SessionPoint
    window_end: SessionPoint


#: Los cinco tramos de ``plan.md`` §8.5, en constantes relativas a la sesión:
#: pre-subasta, recién abierto, mediodía, pre-cierre y cierre. En una sesión completa
#: sus referencias caen en la apertura menos 10 minutos, la apertura más 5, la apertura
#: más 90, el cierre menos 15 y el cierre; en una media sesión se desplazan con el
#: cierre, que es justo lo que se quiere.
SESSION_TRANCHES: Final[tuple[TrancheSpec, ...]] = (
    TrancheSpec(
        name="pre_subasta",
        label="pre-subasta (10 min antes de la apertura)",
        reference=SessionPoint("open", -10),
        window_start=SessionPoint("open", -10),
        window_end=SessionPoint("open", 0),
    ),
    TrancheSpec(
        name="recien_abierto",
        label="recién abierto (primeros 30 min)",
        reference=SessionPoint("open", +5),
        window_start=SessionPoint("open", 0),
        window_end=SessionPoint("open", 30),
    ),
    TrancheSpec(
        name="mediodia",
        label="mediodía (90 min tras la apertura)",
        reference=SessionPoint("open", 90),
        window_start=SessionPoint("open", 30),
        window_end=SessionPoint("close", -30),
    ),
    TrancheSpec(
        name="pre_cierre",
        label="pre-cierre (15 min antes del cierre)",
        reference=SessionPoint("close", -15),
        window_start=SessionPoint("close", -30),
        window_end=SessionPoint("close", 0),
    ),
    TrancheSpec(
        name="cierre",
        label="cierre (subasta de cierre)",
        reference=SessionPoint("close", 0),
        window_start=SessionPoint("close", 0),
        window_end=SessionPoint("close", 10),
    ),
)


class MeasureState(StrEnum):
    """Estado de una medida. Un desconocido nunca vale ``0``: vale ``null``."""

    MEASURED = "measured"
    """Hay dato, con su fuente y su número de observaciones."""

    UNMEASURED = "unmeasured"
    """No hay dato. El valor es ``null`` y el motivo es obligatorio."""


class Side(StrEnum):
    """Lado de la operación: importa porque el *carry* es asimétrico."""

    LONG = "long"
    SHORT = "short"


class CostAuditError(Exception):
    """No se puede consolidar el informe."""


class CostTemplateError(CostAuditError):
    """La plantilla de captura falta, no es YAML o no pasa la validación."""


# ─────────────────────────────────────────────────────────────────────────────
# Plantilla de captura: validación con Pydantic
# ─────────────────────────────────────────────────────────────────────────────
def _to_utc(value: datetime, *, field_name: str) -> datetime:
    """Instante UTC. Un *timestamp* sin zona horaria **no** vale (A17, A19)."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{field_name} debe llevar zona horaria explícita (se guarda en UTC, "
            "ISO-8601 con +00:00 o Z)"
        )
    return value.astimezone(UTC)


def _positive(value: Decimal, *, field_name: str) -> Decimal:
    if value <= 0:
        raise ValueError(f"{field_name} debe ser > 0: un valor no positivo no mide nada")
    return value


class SpreadObservation(BaseModel):
    """Un ``ask - bid`` anotado a mano en un instante concreto."""

    model_config = ConfigDict(extra="forbid")

    timestamp_utc: datetime
    notional_usd: Decimal
    bid: Decimal
    ask: Decimal

    @field_validator("timestamp_utc")
    @classmethod
    def _check_timestamp(cls, value: datetime) -> datetime:
        return _to_utc(value, field_name="timestamp_utc")

    @field_validator("notional_usd")
    @classmethod
    def _check_notional(cls, value: Decimal) -> Decimal:
        return _positive(value, field_name="notional_usd")

    @model_validator(mode="after")
    def _check_order(self) -> SpreadObservation:
        if self.ask < self.bid:
            raise ValueError(
                f"ask ({self.ask}) es menor que bid ({self.bid}): el spread no puede ser negativo"
            )
        return self


class TrackingPair(BaseModel):
    """CFD e índice leídos en el mismo instante (o casi)."""

    model_config = ConfigDict(extra="forbid")

    series_id: str
    cfd_timestamp_utc: datetime
    cfd_price: Decimal
    index_timestamp_utc: datetime
    index_price: Decimal

    @field_validator("cfd_timestamp_utc", "index_timestamp_utc")
    @classmethod
    def _check_timestamp(cls, value: datetime, info: ValidationInfo) -> datetime:
        return _to_utc(value, field_name=info.field_name or "timestamp")

    @field_validator("cfd_price", "index_price")
    @classmethod
    def _check_price(cls, value: Decimal, info: ValidationInfo) -> Decimal:
        return _positive(value, field_name=info.field_name or "price")


class Execution(BaseModel):
    """Una ejecución real: precio obtenido frente al precio de referencia."""

    model_config = ConfigDict(extra="forbid")

    timestamp_utc: datetime
    side: Side
    notional_usd: Decimal
    reference_price: Decimal
    filled_price: Decimal

    @field_validator("timestamp_utc")
    @classmethod
    def _check_timestamp(cls, value: datetime) -> datetime:
        return _to_utc(value, field_name="timestamp_utc")

    @field_validator("notional_usd", "reference_price", "filled_price")
    @classmethod
    def _check_positive(cls, value: Decimal, info: ValidationInfo) -> Decimal:
        return _positive(value, field_name=info.field_name or "importe")


class CostObservations(BaseModel):
    """Contenido de ``config/cost_observations.yaml``.

    Los siete campos son **obligatorios**: una plantilla incompleta no es una
    plantilla válida (A22: campo obligatorio ausente ⇒ no se escribe informe). Un
    campo desconocido tampoco se ignora en silencio (``extra="forbid"``).
    """

    model_config = ConfigDict(extra="forbid")

    trading_window: None
    """Placeholder documentado: la ventana está confirmada y las fronteras las
    deriva el calendario, así que aquí solo puede haber ``null``."""

    settlement_currency: str | None
    """``null`` = la declaración del usuario (USD). Otra divisa se rechaza: su coste
    de conversión no está medido y no se emite un cero en su lugar."""

    financing_cut: datetime | None
    """``null`` = **sin verificar**. Es el estado real a 2026-09-18."""

    minimum_commission_usd: Decimal | None
    """Mínimo absoluto de comisión o de spread en puntos. ``null`` = no declarado
    (que no es lo mismo que ``0``: un mínimo de 0 es «sin mínimo» y se rechaza)."""

    spread_observations: tuple[SpreadObservation, ...]
    tracking_pairs: tuple[TrackingPair, ...]
    executions: tuple[Execution, ...]

    @field_validator("settlement_currency")
    @classmethod
    def _check_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if normalized != DECLARED_SETTLEMENT_CURRENCY:
            raise ValueError(
                f"la divisa de liquidación está declarada como "
                f"{DECLARED_SETTLEMENT_CURRENCY} ({SOURCE_USER_DECLARATION}, "
                f"{WINDOW_CONFIRMED_ON.isoformat()}): con {normalized} el coste de "
                "conversión no está medido y este informe no puede emitir un 0 en su lugar"
            )
        return normalized

    @field_validator("financing_cut")
    @classmethod
    def _check_cut(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _to_utc(value, field_name="financing_cut")

    @field_validator("minimum_commission_usd")
    @classmethod
    def _check_minimum(cls, value: Decimal | None) -> Decimal | None:
        if value is None:
            return None
        if value == 0:
            raise ValueError(
                "minimum_commission_usd = 0 es «sin mínimo», no «no declarado»: déjalo "
                "a null para no confundir un cero con un desconocido"
            )
        return _positive(value, field_name="minimum_commission_usd")


def load_template(path: Path | str) -> CostObservations:
    """Carga y valida la plantilla de captura.

    Raises
    ------
    CostTemplateError
        Si el fichero no existe, no es YAML, no es un *mapping* o no valida contra el
        esquema (errores tipados de Pydantic, nunca un ``assert``).
    """
    target = Path(path)
    if not target.is_file():
        raise CostTemplateError(f"no existe la plantilla de captura: {target}")
    try:
        loaded: object = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise CostTemplateError(f"YAML inválido en {target}: {error}") from error
    if not isinstance(loaded, dict):
        raise CostTemplateError(f"{target} debe contener un mapping en la raíz")
    try:
        return CostObservations.model_validate(loaded)
    except ValidationError as error:
        raise CostTemplateError(f"plantilla inválida en {target}: {error}") from error


# ─────────────────────────────────────────────────────────────────────────────
# Tramos de sesión (relativos a la apertura y al cierre, nunca horas fijas)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SessionTranche:
    """Un tramo ya materializado sobre una sesión concreta."""

    name: str
    label: str
    reference_utc: datetime
    window_start_utc: datetime
    window_end_utc: datetime
    reference_et: str
    label_et: str
    window_label_et: str


def _et_offset_text(instant: datetime) -> str:
    """Offset vigente en ET: ``-04:00`` en horario de verano, ``-05:00`` si no."""
    offset = MarketCalendar.to_et(instant).utcoffset() or timedelta()
    total_minutes = int(offset.total_seconds()) // 60
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def _et_label(instant: datetime) -> str:
    """Etiqueta en ``America/New_York`` con su offset vigente (A11, A17)."""
    return f"{MarketCalendar.to_et(instant):%H:%M} ET {_et_offset_text(instant)}"


def session_tranches(calendar: MarketCalendar, day: date) -> tuple[SessionTranche, ...]:
    """Materializa los cinco tramos sobre una sesión concreta.

    Las fronteras salen de ``MarketCalendar`` (apertura y cierre reales del día, con
    media sesión incluida) más los desplazamientos declarados en
    :data:`SESSION_TRANCHES`: **no hay ninguna hora literal** en el cálculo.
    """
    info = calendar.session(day)
    open_utc = info.open_utc
    close_utc = info.close_utc
    if not info.is_session or open_utc is None or close_utc is None:
        raise CostAuditError(f"{day.isoformat()} no es sesión: {info.reason}")

    def resolve(point: SessionPoint) -> datetime:
        base = open_utc if point.anchor == "open" else close_utc
        return base + timedelta(minutes=point.offset_minutes)

    tranches: list[SessionTranche] = []
    for spec in SESSION_TRANCHES:
        start = resolve(spec.window_start)
        end = resolve(spec.window_end)
        reference = resolve(spec.reference)
        tranches.append(
            SessionTranche(
                name=spec.name,
                label=spec.label,
                reference_utc=reference,
                window_start_utc=start,
                window_end_utc=end,
                reference_et=MarketCalendar.to_et(reference).isoformat(),
                label_et=_et_label(reference),
                window_label_et=f"{_et_label(start)} → {_et_label(end)}",
            )
        )
    return tuple(tranches)


def tranche_of(instant_utc: datetime, tranches: Sequence[SessionTranche]) -> str | None:
    """Nombre del tramo al que cae el instante, o ``None`` si cae fuera de la sesión."""
    for tranche in tranches:
        if tranche.window_start_utc <= instant_utc < tranche.window_end_utc:
            return tranche.name
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Serialización exacta de importes
# ─────────────────────────────────────────────────────────────────────────────
def _num(value: Decimal) -> str:
    """Decimal → cadena exacta, sin notación científica y sin ceros de relleno."""
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _mean(values: Sequence[Decimal]) -> Decimal:
    """Media exacta, cuantizada para que el JSON no arrastre 28 dígitos."""
    return (sum(values, Decimal(0)) / Decimal(len(values))).quantize(
        _MEAN_QUANTUM, rounding=ROUND_HALF_UP
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entradas confirmadas y campos sin verificar
# ─────────────────────────────────────────────────────────────────────────────
def _confirmed_inputs(info: SessionInfo) -> dict[str, Any]:
    """La ventana y la divisa, con su procedencia. No son mediciones (A1, A3)."""
    open_utc = info.open_utc
    close_utc = info.close_utc
    return {
        "trading_window": {
            "state": "confirmed",
            "value": "sesión regular del S&P 500",
            "source": SOURCE_USER_DECLARATION,
            "confirmed_on": WINDOW_CONFIRMED_ON.isoformat(),
            "reference_timezone": "America/New_York",
            "open_utc": None if open_utc is None else open_utc.isoformat(),
            "close_utc": None if close_utc is None else close_utc.isoformat(),
            "open_et": None if open_utc is None else MarketCalendar.to_et(open_utc).isoformat(),
            "close_et": None if close_utc is None else MarketCalendar.to_et(close_utc).isoformat(),
            "half_day": info.is_half_day,
            "borders_derived_by": "cfdtrader.data.calendar.MarketCalendar",
            "note": (
                "no se vuelve a marcar como pendiente de verificación: las fronteras se "
                "derivan del calendario en America/New_York, con el DST incluido"
            ),
        },
        "settlement_currency": {
            "state": "confirmed",
            "value": DECLARED_SETTLEMENT_CURRENCY,
            "source": SOURCE_USER_DECLARATION,
            "confirmed_on": WINDOW_CONFIRMED_ON.isoformat(),
            "note": (
                "Esto es el coste de conversión (0 % con motivo); la exposición de divisa "
                "de una cuenta en EUR con nocional en USD es otra cosa → #27"
            ),
        },
    }


def _financing_cut(value: datetime | None) -> dict[str, Any]:
    """El corte de financiación: sin verificar mientras no haya respuesta del bróker."""
    if value is None:
        return {
            "state": "unverified",
            "value_utc": None,
            "value_et": None,
            "reason": (
                "no verificado a 2026-09-18: el documento del bróker dice «por cada noche» "
                "pero no fija el instante, y asumir una hora de corte fija está prohibido "
                "(si el corte cae antes del cierre, el intradía puro pagaría tenencia)"
            ),
            "broker_question": FINANCING_CUT_QUESTION,
            "how_to_fill": (
                "escribir en config/cost_observations.yaml el instante ISO-8601 con zona "
                "horaria que devuelva el bróker; el informe lo convertirá a UTC y a "
                "America/New_York"
            ),
            "impact": (
                "sin este dato no se puede afirmar que la tenencia intradía sea 0: el "
                "escenario de una noche mal cerrada cuesta 0,0224 % en largo (5× el diferencial)"
            ),
        }
    return {
        "state": "verified",
        "value_utc": value.astimezone(UTC).isoformat(),
        "value_et": MarketCalendar.to_et(value).isoformat(),
        "reason": f"contestado por el bróker y anotado a mano ({SOURCE_USER_DECLARATION})",
        "broker_question": FINANCING_CUT_QUESTION,
        "how_to_fill": None,
        "impact": (
            "el intradía puro solo queda libre de tenencia si el corte es posterior al cierre"
        ),
    }


class FxCost(BaseModel):
    """El coste de conversión: un cero **con motivo**, nunca un cero silencioso (A3).

    El modelo existe precisamente para eso: sin ``reason`` (o con la cadena vacía) no
    valida, así que no hay manera de emitir un 0 sin explicar de dónde sale.
    """

    model_config = ConfigDict(extra="forbid")

    value_usd: Decimal
    value_pct: Decimal
    source: str = Field(min_length=1)
    reason: str = Field(min_length=1)


def _fx_cost() -> dict[str, Any]:
    """Coste de conversión 0 % con motivo: el nocional se liquida en USD."""
    cost = FxCost(
        value_usd=Decimal("0"),
        value_pct=FX_COST_PCT,
        source=f"{SOURCE_USER_DECLARATION} ({WINDOW_CONFIRMED_ON.isoformat()})",
        reason="nocional liquidado en USD: no hay conversión de divisa que costear",
    )
    return {
        "state": MeasureState.MEASURED.value,
        "value_usd": _num(cost.value_usd),
        "value_usd_unit": UNIT_USD,
        "value_pct": _num(cost.value_pct),
        "value_pct_unit": UNIT_PCT,
        "notional_usd": _num(REFERENCE_NOTIONAL_USD),
        "source": cost.source,
        "reason": cost.reason,
        "note": (
            "es un cero con procedencia, no un sustituto de un valor desconocido; "
            "la exposición de divisa es otra cosa → #27"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# La tabla declarada, por tamaño y la anualización
# ─────────────────────────────────────────────────────────────────────────────
def _amount(
    usd: Decimal | None,
    pct: Decimal | None,
    *,
    unit_usd: str = UNIT_USD,
    unit_pct: str = UNIT_PCT,
    notional_usd: Decimal = REFERENCE_NOTIONAL_USD,
) -> dict[str, Any]:
    """Un importe **con su unidad y su nocional**: sin eso no hay cifra interpretable."""
    return {
        "usd": None if usd is None else _num(usd),
        "usd_unit": unit_usd,
        "pct": None if pct is None else _num(pct),
        "pct_unit": unit_pct,
        "notional_usd": _num(notional_usd),
    }


def _declared_rows() -> list[dict[str, Any]]:
    """La tabla de ``plan.md`` §3.3, fila a fila, con nombre y unidad declarados."""
    return [
        {
            "concept": "diferencial (spread) declarado",
            "direction": None,
            "per": "ida y vuelta (media al entrar + media al salir)",
            "source": SOURCE_DECLARED_TABLE,
            "amount": _amount(SPREAD_USD, SPREAD_PCT),
        },
        {
            "concept": "tenencia declarada en CORTO",
            "direction": Side.SHORT.value,
            "per": "por noche",
            "source": SOURCE_DECLARED_TABLE,
            "amount": _amount(
                CARRY_SHORT_USD_PER_NIGHT, CARRY_SHORT_PCT_PER_NIGHT, unit_pct=UNIT_PCT_PER_NIGHT
            ),
        },
        {
            "concept": "tenencia declarada en LARGO",
            "direction": Side.LONG.value,
            "per": "por noche",
            "source": SOURCE_DECLARED_TABLE,
            "amount": _amount(
                CARRY_LONG_USD_PER_NIGHT, CARRY_LONG_PCT_PER_NIGHT, unit_pct=UNIT_PCT_PER_NIGHT
            ),
        },
        {
            "concept": "cambio de divisa declarado",
            "direction": None,
            "per": "por operación",
            "source": f"{SOURCE_DECLARED_TABLE} + {SOURCE_USER_DECLARATION}",
            "reason": "nocional liquidado en USD (0 % con motivo, no un cero mudo)",
            "amount": _amount(Decimal("0"), FX_COST_PCT),
        },
    ]


def _round_trip() -> dict[str, Any]:
    """Totales de ida y vuelta, exactos: CORTO 0,24 $ y LARGO 2,24 $ (A8)."""
    short_usd = SPREAD_USD + CARRY_SHORT_USD_PER_NIGHT * HOLDING_NIGHTS
    short_pct = SPREAD_PCT + CARRY_SHORT_PCT_PER_NIGHT * HOLDING_NIGHTS
    long_usd = SPREAD_USD + CARRY_LONG_USD_PER_NIGHT * HOLDING_NIGHTS
    long_pct = SPREAD_PCT + CARRY_LONG_PCT_PER_NIGHT * HOLDING_NIGHTS
    return {
        "holding_nights": HOLDING_NIGHTS,
        "short": {
            "direction": Side.SHORT.value,
            "formula": (
                f"{_num(SPREAD_USD)} $ + ({_num(CARRY_SHORT_USD_PER_NIGHT)} $/noche × "
                f"{HOLDING_NIGHTS} noche)"
            ),
            "amount": _amount(short_usd, short_pct),
        },
        "long": {
            "direction": Side.LONG.value,
            "formula": (
                f"{_num(SPREAD_USD)} $ + ({_num(CARRY_LONG_USD_PER_NIGHT)} $/noche × "
                f"{HOLDING_NIGHTS} noche)"
            ),
            "amount": _amount(long_usd, long_pct),
        },
    }


def _annualisation() -> dict[str, Any]:
    """Cifras anualizadas declaradas y coherencia de la anualización (A9).

    Las cifras anualizadas **se registran como declaradas**, no se derivan de las
    diarias. Para no elegir una convención en silencio, se publica la ratio implicada
    de cada lado (``anualizado / por noche``) y se marca ``annualisation_consistent``
    si las dos ratios no se separan más del ``ANNUALISATION_TOLERANCE`` declarado.
    """
    ratio_short = (ANNUALISED_SHORT_PCT / CARRY_SHORT_PCT_PER_NIGHT).quantize(
        _RATIO_QUANTUM, rounding=ROUND_HALF_UP
    )
    ratio_long = (ANNUALISED_LONG_PCT / CARRY_LONG_PCT_PER_NIGHT).quantize(
        _RATIO_QUANTUM, rounding=ROUND_HALF_UP
    )
    relative = (abs(ratio_short - ratio_long) / max(abs(ratio_short), abs(ratio_long))).quantize(
        _RELATIVE_QUANTUM, rounding=ROUND_HALF_UP
    )
    return {
        "short": {
            "pct": _num(ANNUALISED_SHORT_PCT),
            "pct_unit": UNIT_PCT_ANNUAL,
            "notional_usd": _num(REFERENCE_NOTIONAL_USD),
            "source": SOURCE_ANNUALISED,
            "ratio_annualised_over_per_night": _num(ratio_short),
        },
        "long": {
            "pct": _num(ANNUALISED_LONG_PCT),
            "pct_unit": UNIT_PCT_ANNUAL,
            "notional_usd": _num(REFERENCE_NOTIONAL_USD),
            "source": SOURCE_ANNUALISED,
            "ratio_annualised_over_per_night": _num(ratio_long),
        },
        "annualisation_consistent": relative <= ANNUALISATION_TOLERANCE,
        "relative_difference": _num(relative),
        "relative_difference_unit": "% (fracción) entre las dos ratios implicadas",
        "tolerance": _num(ANNUALISATION_TOLERANCE),
        "derived_from_daily": False,
        "note": (
            "las cifras anualizadas son declaradas: no se derivan de las diarias. Las dos "
            "ratios implicadas no coinciden exactamente (~362,7 en corto frente a ~366,2 en "
            "largo), y con la tolerancia declarada del 1 % el documento es coherente consigo "
            "mismo; la base exacta (365 días, noches de calendario o sesiones) hay que "
            "preguntarla al bróker"
        ),
    }


def _declared_table() -> dict[str, Any]:
    """Bloque declarado completo: filas, totales, anualizado y su procedencia."""
    annualisation = _annualisation()
    return {
        "source": SOURCE_DECLARED_TABLE,
        "reference_notional_usd": _num(REFERENCE_NOTIONAL_USD),
        "holding_nights": HOLDING_NIGHTS,
        "units": {
            "usd": UNIT_USD,
            "pct": UNIT_PCT,
            "pct_per_night": UNIT_PCT_PER_NIGHT,
            "pct_annual": UNIT_PCT_ANNUAL,
        },
        "rows": _declared_rows(),
        "round_trip": _round_trip(),
        "annualised": {
            "short": annualisation["short"],
            "long": annualisation["long"],
        },
        "annualisation": annualisation,
        "note": (
            "la tabla es el documento del bróker, no una medición; los totales de ida y "
            "vuelta son los mismos que #11 debe reproducir dentro del motor de backtest"
        ),
    }


def _by_size(minimum_commission_usd: Decimal | None) -> dict[str, Any]:
    """Consolidación por tamaño, con detección de anomalía de escala (A12)."""
    computed: list[tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]] = []
    for notional in SIZE_LADDER_USD:
        proportional = SPREAD_PCT / Decimal(100) * notional
        spread_usd = (
            proportional
            if minimum_commission_usd is None
            else max(proportional, minimum_commission_usd)
        )
        carry_short = CARRY_SHORT_PCT_PER_NIGHT / Decimal(100) * notional * HOLDING_NIGHTS
        carry_long = CARRY_LONG_PCT_PER_NIGHT / Decimal(100) * notional * HOLDING_NIGHTS
        total_short = spread_usd + carry_short
        total_long = spread_usd + carry_long
        computed.append(
            (
                notional,
                spread_usd,
                carry_short,
                carry_long,
                total_short / notional * Decimal(100),
                total_long / notional * Decimal(100),
                total_long,
            )
        )

    short_pcts = {row[4] for row in computed}
    long_pcts = {row[5] for row in computed}
    constant = len(short_pcts) == 1 and len(long_pcts) == 1
    reference_short, reference_long = computed[-1][4], computed[-1][5]

    rows: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    for notional, spread_usd, carry_short, carry_long, short_pct, long_pct, total_long in computed:
        total_short = spread_usd + carry_short
        rows.append(
            {
                "notional_usd": _num(notional),
                "spread_usd": _num(spread_usd),
                "spread_pct": _num(spread_usd / notional * Decimal(100)),
                "carry_short_usd": _num(carry_short),
                "carry_long_usd": _num(carry_long),
                "round_trip_short_usd": _num(total_short),
                "round_trip_short_pct": _num(short_pct),
                "round_trip_long_usd": _num(total_long),
                "round_trip_long_pct": _num(long_pct),
                "minimum_applied_usd": (
                    None if minimum_commission_usd is None else _num(minimum_commission_usd)
                ),
            }
        )
        detail.append(
            {
                "notional_usd": _num(notional),
                "round_trip_short_pct": _num(short_pct),
                "round_trip_long_pct": _num(long_pct),
                "vs_reference_short_pct": _num(short_pct - reference_short),
                "vs_reference_long_pct": _num(long_pct - reference_long),
            }
        )
    return {
        "notionals_usd": [_num(notional) for notional in SIZE_LADDER_USD],
        "reference_notional_usd": _num(REFERENCE_NOTIONAL_USD),
        "units": {"usd": "$ (nocional de la fila)", "pct": UNIT_PCT},
        "minimum_commission_usd": (
            None if minimum_commission_usd is None else _num(minimum_commission_usd)
        ),
        "rows": rows,
        "percentage_is_constant": constant,
        "size_scaling_anomaly": not constant,
        "detail": detail,
        "reason": (
            "el modelo declarado es proporcional: el % no cambia con el tamaño (es "
            "constancia del modelo declarado, no una medición; `plan.md` §21 pregunta 5 "
            "sigue sin verificar)"
            if constant
            else (
                "con el mínimo absoluto declarado el % deja de ser constante entre tamaños: "
                "en los nocionales pequeños pesa más. Se publica el detalle en vez de promediar"
            )
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Las tres medidas, cada una con su estado. Nunca se fusionan (A13)
# ─────────────────────────────────────────────────────────────────────────────
def _unmeasured(
    *, reason: str, how_to_fill: str | None = None, observations: int = 0
) -> dict[str, Any]:
    """Bloque de una medida no medida. ``null`` explícito, jamás un ``0``."""
    return {
        "state": MeasureState.UNMEASURED.value,
        "value_usd": None,
        "value_usd_unit": UNIT_USD,
        "value_pct": None,
        "value_pct_unit": UNIT_PCT,
        "value_points": None,
        "value_points_unit": UNIT_POINTS,
        "notional_usd": _num(REFERENCE_NOTIONAL_USD),
        "source": None,
        "observations": observations,
        "reason": reason,
        "how_to_fill": how_to_fill,
    }


def _spread_block(
    observations: Sequence[SpreadObservation], tranches: Sequence[SessionTranche], *, template: str
) -> dict[str, Any]:
    """Consolidación del spread cotizado por tramo, comparada con el declarado (A15)."""
    in_session: dict[str, list[SpreadObservation]] = {tranche.name: [] for tranche in tranches}
    out_of_session = 0
    for observation in observations:
        name = tranche_of(observation.timestamp_utc, tranches)
        if name is None:
            out_of_session += 1
        else:
            in_session[name].append(observation)

    rows: list[dict[str, Any]] = []
    for tranche in tranches:
        mine = in_session[tranche.name]
        base = {
            "tranche": tranche.name,
            "label": tranche.label,
            "label_et": tranche.label_et,
            "reference_et": tranche.reference_et,
            "window_utc": [
                tranche.window_start_utc.isoformat(),
                tranche.window_end_utc.isoformat(),
            ],
            "window_et": tranche.window_label_et,
            "declared_usd": _num(SPREAD_USD),
            "declared_pct": _num(SPREAD_PCT),
        }
        if not mine:
            rows.append(
                {
                    **base,
                    "state": MeasureState.UNMEASURED.value,
                    "observations": 0,
                    "timestamps_utc": [],
                    "spread_usd": None,
                    "spread_pct": None,
                    "observed_minus_declared_usd": None,
                    "observed_minus_declared_pct": None,
                    "wider_than_declared": None,
                    "source": None,
                    "reason": "sin observaciones de bid/ask en este tramo",
                }
            )
            continue

        spreads_usd = [observation.ask - observation.bid for observation in mine]
        spreads_pct = [
            (observation.ask - observation.bid) / observation.notional_usd * Decimal(100)
            for observation in mine
        ]
        mean_usd = _mean(spreads_usd)
        mean_pct = _mean(spreads_pct)
        at_reference = all(
            observation.notional_usd == REFERENCE_NOTIONAL_USD for observation in mine
        )
        difference_usd = (mean_usd - SPREAD_USD) if at_reference else None
        difference_pct = mean_pct - SPREAD_PCT
        rows.append(
            {
                **base,
                "state": MeasureState.MEASURED.value,
                "observations": len(mine),
                "timestamps_utc": sorted(
                    observation.timestamp_utc.isoformat() for observation in mine
                ),
                "notionals_usd": sorted({_num(o.notional_usd) for o in mine}),
                "spread_usd": _num(mean_usd),
                "spread_pct": _num(mean_pct),
                "observed_minus_declared_usd": (
                    None if difference_usd is None else _num(difference_usd)
                ),
                "observed_minus_declared_pct": _num(difference_pct),
                "wider_than_declared": mean_pct > SPREAD_PCT,
                "source": f"{len(mine)} observaciones anotadas a mano en {template}",
                "reason": (
                    "medido con las observaciones de bid/ask anotadas a mano; el $ solo se "
                    "compara con el declarado cuando el nocional es el de referencia"
                    if at_reference
                    else "medido, pero con un nocional distinto del de referencia: la "
                    "comparación en $ no es comparable (la de % sí)"
                ),
            }
        )

    measured = [row for row in rows if row["state"] == MeasureState.MEASURED.value]
    base_block = _unmeasured(
        reason=(
            "no hay observaciones de bid/ask en la plantilla: el spread cotizado no se ha "
            "medido (el declarado de plan.md §3.3 es una cifra del documento del bróker)"
            if not observations
            else (
                "ninguna observación cae dentro de los tramos de la sesión auditada: no hay "
                "nada que consolidar"
            )
        ),
        how_to_fill=(
            "anotar `ask - bid` en el SPX500:CFD antes de la subasta, recién abierto, al "
            "mediodía, antes del cierre y en el cierre (plan.md §8.5 y §21 pregunta 2), con "
            "su `timestamp_utc` y el nocional; la apertura suele ser bastante más ancha"
        ),
        observations=len(observations),
    )
    if not measured:
        return {**base_block, "out_of_session": out_of_session, "by_tranche": rows}

    all_in_session = [row for name in in_session for row in in_session[name]]
    usd_values = [(observation.ask - observation.bid) for observation in all_in_session]
    pct_values = [
        (observation.ask - observation.bid) / observation.notional_usd * Decimal(100)
        for observation in all_in_session
    ]
    return {
        "state": MeasureState.MEASURED.value,
        "value_usd": _num(_mean(usd_values)),
        "value_usd_unit": UNIT_USD,
        "value_pct": _num(_mean(pct_values)),
        "value_pct_unit": UNIT_PCT,
        "value_points": None,
        "value_points_unit": UNIT_POINTS,
        "notional_usd": _num(REFERENCE_NOTIONAL_USD),
        "source": f"{len(all_in_session)} observaciones anotadas a mano en {template}",
        "observations": len(all_in_session),
        "reason": (
            "media simple de las observaciones dentro de la sesión; el detalle por tramo está "
            "en `by_tranche`, que es donde se ve qué tramo se ensancha (plan.md §3.3)"
        ),
        "how_to_fill": None,
        "out_of_session": out_of_session,
        "aggregation": "media simple de las observaciones dentro de la sesión",
        "by_tranche": rows,
    }


def _tracking_block(pairs: Sequence[TrackingPair], *, template: str) -> dict[str, Any]:
    """*Tracking difference* con pares del mismo instante: nunca una media diaria (A16)."""
    details: list[dict[str, Any]] = []
    accepted: list[tuple[Decimal, Decimal]] = []
    excluded = 0
    for index, pair in enumerate(pairs):
        lag = abs((pair.cfd_timestamp_utc - pair.index_timestamp_utc).total_seconds())
        difference_points = pair.cfd_price - pair.index_price
        difference_pct = difference_points / pair.index_price * Decimal(100)
        in_tolerance = lag <= PAIR_TOLERANCE_SECONDS
        if in_tolerance:
            accepted.append((difference_points, difference_pct))
        else:
            excluded += 1
        details.append(
            {
                "pair": index,
                "series_id": pair.series_id,
                "cfd_timestamp_utc": pair.cfd_timestamp_utc.isoformat(),
                "index_timestamp_utc": pair.index_timestamp_utc.isoformat(),
                "lag_seconds": round(lag, 3),
                "in_tolerance": in_tolerance,
                "difference_points": _num(difference_points),
                "difference_pct": _num(difference_pct),
            }
        )

    if not accepted:
        return {
            **_unmeasured(
                reason=(
                    "no hay pares CFD/índice en la plantilla: el tracking difference no se ha "
                    "medido. No se aproxima con una media diaria, porque comparar el CFD con "
                    "el índice en instantes distintos mide otra cosa"
                    if not pairs
                    else (
                        f"los {excluded} pares superan la tolerancia declarada de "
                        "emparejamiento, así que no hay ningún par comparable"
                    )
                ),
                how_to_fill=(
                    "anotar, en 5–10 sesiones, la cotización del CFD y la del índice en el "
                    "momento (dentro de la tolerancia declarada de "
                    f"{PAIR_TOLERANCE_SECONDS} s), con sus dos timestamps"
                ),
                observations=len(pairs),
            ),
            "notional_usd": None,
            "notional_note": TRACKING_NOTIONAL_NOTE,
            "tolerance_seconds": PAIR_TOLERANCE_SECONDS,
            "pairs_excluded_out_of_tolerance": excluded,
            "pairs": details,
        }

    return {
        "state": MeasureState.MEASURED.value,
        "value_usd": None,
        "value_usd_unit": UNIT_USD,
        "value_pct": _num(_mean([pct for _, pct in accepted])),
        "value_pct_unit": "% del índice",
        "value_points": _num(_mean([points for points, _ in accepted])),
        "value_points_unit": UNIT_POINTS,
        "notional_usd": None,
        "notional_note": TRACKING_NOTIONAL_NOTE,
        "source": f"{len(accepted)} pares pareados anotados a mano en {template}",
        "observations": len(accepted),
        "reason": (
            "solo pares con el CFD y el índice leídos en el mismo instante (tolerancia "
            f"{PAIR_TOLERANCE_SECONDS} s); ninguno se aproxima con una media diaria"
        ),
        "how_to_fill": None,
        "tolerance_seconds": PAIR_TOLERANCE_SECONDS,
        "pairs_excluded_out_of_tolerance": excluded,
        "pairs": details,
    }


def _slippage_block(executions: Sequence[Execution], *, template: str) -> dict[str, Any]:
    """*Slippage* de ejecución. Sin ejecución real **no se mide**: no se rellena (A5)."""
    if not executions:
        return {
            **_unmeasured(
                reason=(
                    "no existe ninguna ejecución real: a 2026-09-18 no se ha operado, así que "
                    "el slippage no se ha medido y no se emite ningún valor de relleno"
                ),
                how_to_fill=SLIPPAGE_HOW_TO_FILL,
                observations=0,
            ),
            "forbidden": (
                "prohibido cualquier valor de relleno: ni constante provisional, ni un 0 con "
                "motivo, ni un rango inventado"
            ),
        }

    details: list[dict[str, Any]] = []
    pct_values: list[Decimal] = []
    usd_values: list[Decimal] = []
    for execution in executions:
        raw = (execution.filled_price - execution.reference_price) / execution.reference_price
        signed = raw if execution.side is Side.LONG else -raw
        pct = signed * Decimal(100)
        usd = pct / Decimal(100) * execution.notional_usd
        pct_values.append(pct)
        usd_values.append(usd)
        details.append(
            {
                "timestamp_utc": execution.timestamp_utc.isoformat(),
                "side": execution.side.value,
                "notional_usd": _num(execution.notional_usd),
                "reference_price": _num(execution.reference_price),
                "filled_price": _num(execution.filled_price),
                "slippage_pct": _num(pct),
                "slippage_usd": _num(usd),
            }
        )

    return {
        "state": MeasureState.MEASURED.value,
        "value_usd": _num(_mean(usd_values)),
        "value_usd_unit": UNIT_USD,
        "value_pct": _num(_mean(pct_values)),
        "value_pct_unit": "% del nocional (positivo = coste)",
        "value_points": None,
        "value_points_unit": UNIT_POINTS,
        "notional_usd": _num(REFERENCE_NOTIONAL_USD),
        "source": f"{len(executions)} ejecuciones anotadas a mano en {template}",
        "observations": len(executions),
        "reason": (
            "medido con ejecuciones reales: precio obtenido frente al precio de referencia en "
            "el instante de la orden. Positivo = coste (comprar más caro o vender más barato)"
        ),
        "how_to_fill": None,
        "sign_convention": (
            "largo: (obtenido - referencia)/referencia; corto: (referencia - obtenido)/referencia"
        ),
        "executions": details,
    }


def _phase0_gate_b(slippage: dict[str, Any]) -> dict[str, Any]:
    """La puerta (b) de #9 no es evaluable hoy. **No** se emite veredicto (A6)."""
    evaluable = slippage["state"] == MeasureState.MEASURED.value
    return {
        "criterion": "slippage sistemático > ~20 % de R (tasks.md, tarea 9, puerta (b))",
        "threshold_pct_of_r": "20",
        "evaluable": evaluable,
        "reason": (
            "la condición se evalúa con el slippage medido; hoy hay ejecuciones anotadas"
            if evaluable
            else (
                "el slippage de ejecución está sin medir (no existe ninguna ejecución real), "
                "así que la condición (b) no es evaluable con este artefacto"
            )
        ),
        "verdict_owner": "#9",
        "note": (
            "aquí solo se deja dicho si la condición es evaluable; el veredicto de "
            "continuidad, reencuadre o parada es de #9 y no se emite en este informe"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Preguntas al bróker, limitaciones y notas
# ─────────────────────────────────────────────────────────────────────────────
FINANCING_CUT_QUESTION: Final[str] = (
    "¿en qué instante exacto se cobra el coste de tenencia (*swap*) de una posición en "
    "`SPX500:CFD`, en hora del bróker y su equivalente en `America/New_York`? ¿Se cobra por "
    "noche de calendario o por sesión?"
)

SLIPPAGE_HOW_TO_FILL: Final[str] = (
    "plan.md §8.5: anotar el precio que se obtiene frente al precio de referencia en el "
    "instante de la orden, repetido 10–15 veces, en la apertura (que es el momento más "
    "difícil), con el timestamp preciso de cada intento; en `config/cost_observations.yaml`, "
    "en `executions`"
)

#: Preguntas pendientes, con su origen. Se publican tal cual en el informe.
BROKER_QUESTIONS: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "financing_cut",
        "question": FINANCING_CUT_QUESTION,
        "why": (
            "es la verificación más crítica de la Fase 0: si el corte cae antes del cierre, el "
            "intradía puro paga tenencia igualmente (plan.md §8.5 y §21 pregunta 3)"
        ),
        "owner": "#9 (puerta de Fase 0) y #59 (bróker definitivo)",
    },
    {
        "id": "annualisation_basis",
        "question": (
            "¿sobre qué base anualiza el documento del bróker el coste de tenencia: 365 días "
            "naturales, noches de calendario o sesiones?"
        ),
        "why": (
            "las cifras anualizadas declaradas implican ~362,7 noches en corto y ~366,2 en "
            "largo, y las dos ratios no coinciden entre sí"
        ),
        "owner": "#8 (este informe) y #59",
    },
    {
        "id": "minimum_commission",
        "question": (
            "¿aplica el bróker una comisión mínima o un spread mínimo en puntos que haga que el "
            "coste en % sea mayor con nocionales pequeños?"
        ),
        "why": (
            "las cifras declaradas son sobre 10.000 $; en cuentas pequeñas el porcentaje real "
            "puede ser mucho mayor (plan.md §21 pregunta 5)"
        ),
        "owner": "#8 y #11 (modelo de coste del motor)",
    },
    {
        "id": "points_vs_notional",
        "question": (
            "¿cuál es el tamaño de contrato del `SPX500:CFD`, es decir, la equivalencia exacta "
            "entre puntos de índice y $ de nocional?"
        ),
        "why": (
            "el diferencial declarado (0,42 $ sobre 10.000 $ = 0,0042 %) hay que poder leerlo "
            "también en puntos, y sin el tamaño de contrato la conversión no está fijada"
        ),
        "owner": "#8 y #11",
    },
    {
        "id": "effective_entry_exit_hour",
        "question": (
            "¿a qué hora efectiva se puede ejecutar la entrada en la subasta de apertura y el "
            "cierre obligatorio en su plataforma?"
        ),
        "why": (
            "el horario real de ejecución mueve el slippage y es la única casilla abierta del "
            "calendario canónico (plan.md §4.1 y §21 pregunta 7)"
        ),
        "owner": "#9 y #10",
    },
)

#: Limitaciones que el informe declara en vez de esconder (A30).
LIMITATIONS: Final[tuple[str, ...]] = (
    "**No hay ejecución real ⇒ el *slippage* no está medido.** No existe ninguna operación "
    "ejecutada a 2026-09-18, así que `slippage_ejecucion` sale `unmeasured` con `value: null`. "
    "No se emite ninguna constante provisional, ningún 0 con motivo ni ningún rango.",
    "**El corte de financiación está sin verificar.** El campo vale `null` con "
    '`state: "unverified"` y con la pregunta literal al bróker. Asumir una hora de corte '
    "fija está prohibido: si el corte cae antes del cierre, el intradía puro pagaría "
    "tenencia.",
    "**La tabla declarada proviene del documento del bróker, no de una medición**, y el "
    "instrumento es el `SPX500:CFD`, no `^GSPC` ni `ES=F`: el CFD replica el índice con "
    "diferencial y financiación, así que la magnitud medida sobre el índice no es "
    "directamente la del CFD.",
    "**La puerta (b) de #9 no se puede evaluar con esto**: «slippage sistemático > ~20 % de "
    "R» necesita el slippage medido. Aquí solo se declara que no es evaluable; el veredicto "
    "de continuidad es de #9.",
    "**Los ceros que aparecen son ceros con motivo.** El 0 % de cambio de divisa sale de que "
    "el nocional se liquida en USD (declaración del usuario) y va con fuente y motivo; nunca "
    "es el sustituto de un desconocido.",
    "**El spread y el tracking difference solo están medidos si alguien los anota a mano**: "
    "no hay fuente pública gratuita del bid/ask del `SPX500:CFD` (#50), así que la plantilla "
    "vacía es el estado normal hasta que un humano la rellene.",
)

NOTES: Final[tuple[str, ...]] = (
    "Este módulo **mide y verifica**; el modelo de coste del motor de backtest es #11 "
    "(`src/cfdtrader/backtest/costs.py`). Los totales de ida y vuelta (0,24 $ corto / 2,24 $ "
    "largo) son los que #11 debe reproducir dentro del motor.",
    "Los importes se publican como cadenas decimales exactas (`Decimal`), no como `float`: "
    "la comparación con la tabla declarada es exacta, no aproximada.",
    "Las tres medidas son bloques separados y **no se suman**: sumarlas daría un «coste "
    "total» que nadie ha medido.",
    "Las fronteras de sesión se derivan de `America/New_York` con "
    "`cfdtrader.data.calendar.MarketCalendar`; los tramos son desplazamientos relativos a la "
    "apertura y al cierre, nunca horas fijas, para que el DST y las medias sesiones salgan bien.",
    "La exposición de divisa (cuenta en EUR, nocional en USD) es distinta del coste de "
    "conversión y queda fuera de alcance aquí → #27.",
)


# ─────────────────────────────────────────────────────────────────────────────
# Consolidación: función pura (entra el modelo validado, sale el informe)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CostAudit:
    """El informe consolidado. El JSON es el artefacto; el Markdown lo presenta."""

    as_of: datetime
    session_day: date
    template_path: str
    payload: dict[str, Any]

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``cost_audit_<AAAA-MM-DD>``, como los demás estudios."""
        return f"cost_audit_{self.session_day.isoformat()}"

    def json_text(self) -> str:
        """JSON determinista: mismas entradas ⇒ mismo texto byte a byte (A23)."""
        return json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n"

    def write(self, directory: Path) -> tuple[Path, Path]:
        """Escribe el informe en JSON y Markdown."""
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{self.report_stem}.json"
        markdown_path = directory / f"{self.report_stem}.md"
        json_path.write_text(self.json_text(), encoding="utf-8")
        markdown_path.write_text(render_markdown(self), encoding="utf-8")
        return json_path, markdown_path


def _audit_session(calendar: MarketCalendar, now: datetime) -> date:
    """Sesión que audita el informe: la del día de ``now`` en UTC, o la anterior."""
    day = now.date()
    if calendar.is_session(day):
        return day
    return calendar.previous_session(day)


def consolidate(
    observations: CostObservations,
    *,
    calendar: MarketCalendar,
    now: datetime,
    template_path: Path | str = DEFAULT_TEMPLATE_PATH,
) -> CostAudit:
    """Consolida el informe a partir del modelo **ya validado**. Función pura.

    No lee ni escribe disco: la plantilla entra validada y sale el informe, lo que
    permite probar la consolidación en memoria sin tocar el almacén.
    """
    day = _audit_session(calendar, now)
    info = calendar.session(day)
    tranches = session_tranches(calendar, day)
    template = str(template_path)

    spread = _spread_block(observations.spread_observations, tranches, template=template)
    tracking = _tracking_block(observations.tracking_pairs, template=template)
    slippage = _slippage_block(observations.executions, template=template)

    payload: dict[str, Any] = {
        "task": "#8",
        "artifact": "cost_audit",
        "title": "Auditoría de los costes declarados del SPX500:CFD",
        "as_of_utc": now.astimezone(UTC).isoformat(),
        "template_path": template,
        "session": {
            "day": day.isoformat(),
            "is_session": info.is_session,
            "is_half_day": info.is_half_day,
            "duration_hours": info.duration_hours,
            "open_utc": None if info.open_utc is None else info.open_utc.isoformat(),
            "close_utc": None if info.close_utc is None else info.close_utc.isoformat(),
            "open_et": (
                None if info.open_utc is None else MarketCalendar.to_et(info.open_utc).isoformat()
            ),
            "close_et": (
                None if info.close_utc is None else MarketCalendar.to_et(info.close_utc).isoformat()
            ),
            "reference_timezone": "America/New_York",
        },
        "confirmed_inputs": _confirmed_inputs(info),
        "fx_cost": _fx_cost(),
        "financing_cut": _financing_cut(observations.financing_cut),
        "declared_table": _declared_table(),
        "spread_cotizado": spread,
        "tracking_difference": tracking,
        "slippage_ejecucion": slippage,
        "by_size": _by_size(observations.minimum_commission_usd),
        "phase0_gate_b": _phase0_gate_b(slippage),
        "broker_questions": [dict(question) for question in BROKER_QUESTIONS],
        "limitations": list(LIMITATIONS),
        "notes": list(NOTES),
    }
    return CostAudit(
        as_of=now.astimezone(UTC), session_day=day, template_path=template, payload=payload
    )


def report_payload(audit: CostAudit) -> dict[str, Any]:
    """El informe como *mapping* JSON-serializable (el artefacto que se escribe)."""
    return audit.payload


# ─────────────────────────────────────────────────────────────────────────────
# Informe en prosa
# ─────────────────────────────────────────────────────────────────────────────
def _measure_row(name: str, block: dict[str, Any]) -> str:
    """Una fila de la tabla de las tres medidas."""
    if block["state"] == MeasureState.MEASURED.value:
        parts: list[str] = []
        if block.get("value_usd") is not None:
            parts.append(f"{block['value_usd']} $")
        if block.get("value_pct") is not None:
            parts.append(f"{block['value_pct']} %")
        if block.get("value_points") is not None:
            parts.append(f"{block['value_points']} puntos")
        value = " · ".join(parts) if parts else "—"
    else:
        value = "`null` (no medido)"
    source = block.get("source") or "—"
    return (
        f"| `{name}` | `{block['state']}` | {value} | {source} | "
        f"{block['observations']} | {block['reason']} |"
    )


def render_markdown(audit: CostAudit) -> str:
    """Informe legible: la tabla declarada, las tres medidas y lo que falta por saber."""
    payload = audit.payload
    session = payload["session"]
    confirmed = payload["confirmed_inputs"]
    declared = payload["declared_table"]
    lines: list[str] = [
        "# Auditoría de costes declarados del `SPX500:CFD` (tarea #8)",
        "",
        f"- **Sesión auditada:** {session['day']} "
        f"({session['open_et']} → {session['close_et']}, "
        f"media sesión: {'sí' if session['is_half_day'] else 'no'})",
        f"- **Calculado:** {payload['as_of_utc']} (UTC)",
        f"- **Plantilla de captura:** `{payload['template_path']}`",
        f"- **Nocional de referencia:** {declared['reference_notional_usd']} $ · "
        f"tenencia de {declared['holding_nights']} noche",
        "",
        "Este informe **mide y verifica**; el modelo de coste del motor de backtest es **#11**.",
        "",
        "## Entradas confirmadas (no son mediciones)",
        "",
        f"- **Ventana de cotización:** {confirmed['trading_window']['value']}, fronteras "
        f"derivadas por `{confirmed['trading_window']['borders_derived_by']}` en "
        f"`America/New_York` ({session['open_et']} → {session['close_et']}). "
        f"Procedencia: {confirmed['trading_window']['source']}, "
        f"{confirmed['trading_window']['confirmed_on']}. No se vuelve a marcar como pendiente.",
        f"- **Divisa de liquidación:** {confirmed['settlement_currency']['value']} "
        f"({confirmed['settlement_currency']['source']}, "
        f"{confirmed['settlement_currency']['confirmed_on']}). "
        f"{confirmed['settlement_currency']['note']}",
        "",
        "## Tabla declarada (`plan.md` §3.3)",
        "",
        "| concepto | $ | % | unidad | nocional | periodicidad | procedencia |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in declared["rows"]:
        amount = row["amount"]
        lines.append(
            f"| {row['concept']} | {amount['usd']} | {amount['pct']} | "
            f"{amount['usd_unit']} / {amount['pct_unit']} | {amount['notional_usd']} | "
            f"{row['per']} | {row['source']} |"
        )

    round_trip = declared["round_trip"]
    lines.extend(
        [
            "",
            "### Totales de ida y vuelta (comparación exacta, no «≈»)",
            "",
            "| lado | fórmula | $ | % | nocional |",
            "|---|---|---|---|---|",
        ]
    )
    for side in ("short", "long"):
        block = round_trip[side]
        amount = block["amount"]
        lines.append(
            f"| {side} | `{block['formula']}` | **{amount['usd']}** | **{amount['pct']}** | "
            f"{amount['notional_usd']} |"
        )

    annual = declared["annualisation"]
    short_ratio = annual["short"]["ratio_annualised_over_per_night"]
    long_ratio = annual["long"]["ratio_annualised_over_per_night"]
    lines.extend(
        [
            "",
            "### Anualizado declarado y coherencia de la anualización",
            "",
            "Las cifras anualizadas **se registran como declaradas**; no se derivan de las "
            "diarias. Se publica la ratio implicada `anualizado / por-noche` de cada lado en "
            "vez de elegir una convención en silencio.",
            "",
            "| lado | anualizado | ratio implicada | procedencia |",
            "|---|---|---|---|",
            f"| corto | {annual['short']['pct']} % | {short_ratio} | {annual['short']['source']} |",
            f"| largo | {annual['long']['pct']} % | {long_ratio} | {annual['long']['source']} |",
            "",
            f"- **`annualisation_consistent`: "
            f"`{str(annual['annualisation_consistent']).lower()}`** — diferencia relativa entre "
            f"las dos ratios: {annual['relative_difference']} (tolerancia declarada "
            f"{annual['tolerance']}).",
            f"- {annual['note']}",
            "",
            "## Coste por tramo de sesión",
            "",
            "Los cinco tramos de `plan.md` §8.5 son **desplazamientos relativos** a la apertura "
            "y al cierre de esta sesión, no horas fijas: así el DST y las medias sesiones salen "
            "bien. Cada etiqueta va en `America/New_York` con su offset vigente y cada "
            "observación lleva su instante UTC.",
            "",
            "| tramo | etiqueta ET | ventana UTC | estado | spread $ | spread % | declarado % | "
            "obs-decl $ | obs-decl % | ¿más ancho? | obs |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in payload["spread_cotizado"]["by_tranche"]:
        window = " → ".join(row["window_utc"])
        wider = "—" if row["wider_than_declared"] is None else str(row["wider_than_declared"])
        spread_usd = "`null`" if row["spread_usd"] is None else row["spread_usd"]
        spread_pct = "`null`" if row["spread_pct"] is None else row["spread_pct"]
        minus_usd = (
            "—"
            if row["observed_minus_declared_usd"] is None
            else (row["observed_minus_declared_usd"])
        )
        minus_pct = (
            "—"
            if row["observed_minus_declared_pct"] is None
            else (row["observed_minus_declared_pct"])
        )
        lines.append(
            f"| `{row['tranche']}` | {row['label_et']} | {window} | `{row['state']}` | "
            f"{spread_usd} | {spread_pct} | {row['declared_pct']} | {minus_usd} | "
            f"{minus_pct} | {wider} | {row['observations']} |"
        )

    by_size = payload["by_size"]
    lines.extend(
        [
            "",
            "## Coste por tamaño",
            "",
            "| nocional $ | spread $ | spread % | ida y vuelta corto $ | % | "
            "ida y vuelta largo $ | % |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in by_size["rows"]:
        lines.append(
            f"| {row['notional_usd']} | {row['spread_usd']} | {row['spread_pct']} | "
            f"{row['round_trip_short_usd']} | {row['round_trip_short_pct']} | "
            f"{row['round_trip_long_usd']} | {row['round_trip_long_pct']} |"
        )
    minimum = by_size["minimum_commission_usd"] or "`null` (no declarado)"
    lines.extend(
        [
            "",
            f"- **`size_scaling_anomaly`: "
            f"`{str(by_size['size_scaling_anomaly']).lower()}`** — {by_size['reason']}",
            f"- Mínimo absoluto declarado: {minimum}.",
            "",
            "## Las tres medidas (nunca se fusionan)",
            "",
            "| medida | estado | valor | fuente | observaciones | motivo |",
            "|---|---|---|---|---|---|",
        ]
    )
    for name in ("spread_cotizado", "tracking_difference", "slippage_ejecucion"):
        lines.append(_measure_row(name, payload[name]))
    cut = payload["financing_cut"]
    cut_instant = cut["value_utc"] or "`null` (desconocido)"
    lines.extend(
        [
            "",
            "No existe ningún campo que sume las tres: sumarlas daría un «coste total» que "
            "nadie ha medido.",
            "",
            "## Corte de financiación",
            "",
            f"- **`state`: `{cut['state']}`** — {cut['reason']}",
            f"- Instante: {cut_instant}.",
            f"- **Pregunta literal al bróker:** «{cut['broker_question']}»",
            "",
            "## Puerta (b) de la Fase 0 (#9)",
            "",
            f"- Criterio: {payload['phase0_gate_b']['criterion']}.",
            f"- **`evaluable`: `{str(payload['phase0_gate_b']['evaluable']).lower()}`** — "
            f"{payload['phase0_gate_b']['reason']}",
            f"- {payload['phase0_gate_b']['note']}",
            "",
            "## Preguntas pendientes al bróker",
            "",
        ]
    )
    for index, question in enumerate(payload["broker_questions"], start=1):
        lines.append(f"{index}. `{question['id']}` — {question['question']} ({question['why']})")
    lines.extend(["", "## Limitaciones (declaradas, no escondidas)", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.extend(["", "## Notas", ""])
    lines.extend(f"- {item}" for item in payload["notes"])
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución
# ─────────────────────────────────────────────────────────────────────────────
def analyse(
    *,
    template: Path | str = DEFAULT_TEMPLATE_PATH,
    calendar: MarketCalendar | None = None,
    now: datetime,
    reports_dir: Path | None = None,
) -> CostAudit:
    """Carga la plantilla, consolida el informe y, si se pide, lo escribe."""
    observations = load_template(template)
    audit = consolidate(
        observations,
        calendar=calendar or load_calendar(),
        now=now,
        template_path=template,
    )
    if reports_dir is not None:
        json_path, markdown_path = audit.write(reports_dir)
        logger.info("informe de costes: {} y {}", json_path, markdown_path)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del informe de costes.

    Códigos de salida: ``0`` = informe escrito (aunque haya medidas sin medir);
    ``2`` = no se puede consolidar ni siquiera lo declarado (plantilla ilegible o
    campo obligatorio ausente) ⇒ no se escribe informe y el motivo sale por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.analysis.cost_audit", description="Auditoría de costes declarados"
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raíz del almacén")
    parser.add_argument(
        "--template", type=Path, default=DEFAULT_TEMPLATE_PATH, help="plantilla de captura"
    )
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument("--now", default=None, help="instante de referencia ISO (tests)")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.settings)
    except ConfigurationError as error:
        print(f"no se puede leer la configuración: {error}", file=sys.stderr)
        return 2

    data_root = args.data_root if args.data_root is not None else settings.data.root
    now = _parse_now(args.now)
    try:
        audit = analyse(
            template=args.template,
            now=now,
            reports_dir=Path(data_root) / "derived" / "reports",
        )
    except CostAuditError as error:
        print(f"no se puede consolidar el informe de costes: {error}", file=sys.stderr)
        return 2

    logger.info(
        "costes: spread cotizado {}, tracking {}, slippage {}; puerta (b) evaluable: {}",
        audit.payload["spread_cotizado"]["state"],
        audit.payload["tracking_difference"]["state"],
        audit.payload["slippage_ejecucion"]["state"],
        audit.payload["phase0_gate_b"]["evaluable"],
    )
    return 0


def _parse_now(value: str | None) -> datetime:
    """Instante de referencia: el de verdad, o el que fije el test."""
    if value is None:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
