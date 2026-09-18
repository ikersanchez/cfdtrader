"""Motor de costes del *backtest* — tarea #11.

Modelo **puro y determinista** del coste de ida y vuelta de una operación sobre el
``SPX500:CFD`` (``plan.md`` §3.3). El motor **cobra**; medir y auditar lo declarado es
la tarea **#8** (``cfdtrader.analysis.cost_audit``), de donde este módulo **importa** la
tabla declarada: aquí no se vuelve a escribir ni una de sus cifras.

**Sin coste explícito no hay resultado.** ``cost_breakdown`` exige ``model`` y
``slippage`` como argumentos *keyword-only* **sin valor por defecto**: ningún coste tiene
un valor de relleno y el motor no inventa una comisión, un diferencial ni un *slippage*.
Un ``CostModel`` construido sin validar (``model_construct``) se rechaza con
``CostModelError``.

**El *slippage* es el término dominante y un parámetro obligatorio de primera clase.**
Tiene tres estados mutuamente distinguibles y **nunca fusionados** (``measured``,
``assumed``, ``unmeasured``); «no medido» vale ``null`` con motivo, jamás ``0``. El
supuesto pesimista del propietario (2026-09-18, #64) viaja como ``assumed`` y **no**
cierra el total: sin ``R`` decidido (decisión abierta 5 → **#60**) no hay un ``%`` del
nocional que cobrar. El equivalente ilustrativo en bp solo se publica si el llamante lo
pide **explícitamente**, etiquetado ``illustrative``/``decision: false``, y nunca
alimenta ``c_total``.

**Intradía puro por defecto.** ``nights = 0`` es el valor por defecto (``plan.md`` §12,
regla 6: sin *overnight*) y con ``0`` noches no se cobra tenencia. Con ``nights >= 1`` el
llamante **tiene** que declarar el motivo (regla 16: cierre manual obligatorio). El motor
**no** deduce las noches de ningún *timestamp*, del corte de financiación ni de una
duración: ``nights`` es siempre una entrada explícita. El corte de financiación es un
parámetro declarado y hoy está **sin verificar** (``cut_et: null``): asumir una hora de
corte fija está prohibido.

Fronteras declaradas (lo que este módulo **no** hace, con su issue):

- **No** es #8: no mide ni reproduce de nuevo la tabla declarada —la importa— ni toca
  ``config/cost_observations.yaml``.
- **No** es #13: no recorre sesiones, no pide *snapshots* *point-in-time*, no toca
  ``bid``/``ask``, no modela el *gap* de apertura ni simula precios.
- **No** decide ``R``: el tamaño de ``R`` es la decisión abierta 5 → **#60**.
- **No** calcula el ``p*`` de la puerta de Fase 0: ese cálculo es **#9**.
- **No** calcula el nocional, el *sizing*, el apalancamiento ni los límites de pérdida
  (``plan.md`` §3.4) → **#27**.
- **No** lee el almacén de datos, ni el calendario de sesiones, ni ningún precio.

**Aritmética exacta.** Todo el cálculo va en ``decimal.Decimal`` y los totales declarados
se reproducen **exactamente** (``==``, sin tolerancia). Los importes se publican como
cadenas decimales exactas, nunca como ``float``.

La CLI (``uv run python -m cfdtrader.backtest.costs``) es el **único** punto de
entrada/salida del módulo: sin ``--out-dir`` no escribe nada, y con él escribe el informe
**dentro** de ese directorio. Como el motor no consulta el reloj (es puro), la fecha del
informe hay que declararla con ``--as-of``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date, time
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, Self

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from cfdtrader.analysis.cost_audit import (
    BROKER_QUESTIONS,
    CARRY_LONG_PCT_PER_NIGHT,
    CARRY_LONG_USD_PER_NIGHT,
    CARRY_SHORT_PCT_PER_NIGHT,
    CARRY_SHORT_USD_PER_NIGHT,
    DECLARED_FINANCING_CUT,
    DECLARED_SETTLEMENT_CURRENCY,
    FX_COST_PCT,
    HOLDING_NIGHTS,
    R_ILLUSTRATIVE_PCT,
    REFERENCE_NOTIONAL_USD,
    SLIPPAGE_ASSUMPTION_DECIDED_ON,
    SLIPPAGE_ASSUMPTION_PCT_OF_R,
    SLIPPAGE_ASSUMPTION_PROVENANCE,
    SLIPPAGE_ASSUMPTION_R_ISSUE,
    SPREAD_PCT,
    SPREAD_USD,
    WINDOW_CONFIRMED_ON,
    MeasureState,
    Side,
    slippage_assumption_block,
)

__all__ = [
    "COST_MODEL_DOES_NOT_DO",
    "DECLARED_COMMISSION_PCT",
    "DECLARED_OVERNIGHT_REASON",
    "DECLARED_SPREAD_HALF_PCT",
    "DECLARED_SPREAD_HALF_SOURCE",
    "FOLLOW_UPS",
    "LIMITATIONS",
    "CostBreakdown",
    "CostError",
    "CostInputError",
    "CostModel",
    "CostModelError",
    "FinancingCut",
    "MeasureState",
    "Side",
    "SlippageParameter",
    "cost_breakdown",
    "declared_cost_model",
    "declared_slippage_assumption",
    "main",
    "render_markdown",
    "report_payload",
]

# ─────────────────────────────────────────────────────────────────────────────
# Unidades y procedencias (todo importe viaja con su unidad y su origen)
# ─────────────────────────────────────────────────────────────────────────────
#: Unidades declaradas. Sin unidad, una cifra de coste no es interpretable.
UNIT_USD: Final[str] = "$ sobre el nocional de la operación"
UNIT_PCT: Final[str] = "% del nocional"
UNIT_PCT_PER_NIGHT: Final[str] = "% del nocional por noche"
#: ``1 bp = 0,01 % del nocional``: la convención con la que #8 publica el equivalente en
#: bp del supuesto de *slippage* (20 bp = 0,2 % del nocional).
UNIT_BP: Final[str] = "bp del nocional (1 bp = 0,01 % del nocional, la convención de #8)"
RATIO_UNIT: Final[str] = "veces (bp de *slippage* / bp de diferencial)"
#: Cuantía de la ratio de dominancia (A21): cuatro decimales, suficientes para leerla.
RATIO_QUANTUM: Final[Decimal] = Decimal("0.0001")

#: Procedencia de la tabla declarada. Las **cifras** son de #8; aquí solo se citan.
SOURCE_DECLARED_TABLE: Final[str] = "plan.md §3.3 (documento del bróker) · reproducida por #8"
SOURCE_USER: Final[str] = "declaración del usuario (2026-09-18)"

#: Las **dos mitades** del diferencial declarado (A9): cada mitad con su procedencia.
DECLARED_SPREAD_HALF_PCT: Final[Decimal] = SPREAD_PCT / Decimal(2)
DECLARED_SPREAD_HALF_SOURCE: Final[str] = (
    f"plan.md §3.3 vía #8: SPREAD_PCT ({SPREAD_PCT}) = media entrada + media salida"
)

#: Comisión declarada: **0 con procedencia**, no un cero mudo (A11). La tabla declarada
#: de ``plan.md`` §3.3 no incluye comisión; el bróker definitivo es #59.
DECLARED_COMMISSION_PCT: Final[Decimal] = Decimal("0.00")
DECLARED_COMMISSION_SOURCE: Final[str] = (
    "plan.md §3.3 (la tabla declarada no incluye comisión) vía #8 · bróker: #59"
)
DECLARED_COMMISSION_REASON: Final[str] = (
    "la tabla declarada no incluye comisión: se declara 0 **con origen y motivo** en vez de "
    "omitirla; la comisión real del bróker (incluidas comisiones mínimas) es #59"
)

#: Motivo declarado de una noche de tenencia: el caso por defecto es el intradía puro.
DECLARED_OVERNIGHT_REASON: Final[str] = (
    "escenario declarado de tenencia de una noche (la tabla de #8): no es el caso por "
    "defecto. El caso por defecto es el intradía puro (0 noches) y el cierre manual "
    "obligatorio de la regla 16 evita que quede una noche abierta por descuido"
)

#: Justificación declarada de la dominancia del *slippage* (A21).
SLIPPAGE_DOMINANCE_JUSTIFICATION: Final[str] = (
    "plan.md §4.4: «20 bp de *slippage* pesan cincuenta veces más que el diferencial» "
    "(afirmación declarada, no medida)"
)

#: Qué **no** hace el módulo, legible por máquina (A25).
COST_MODEL_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_es_8",
        "issue": "#8",
        "statement": (
            "no mide ni reproduce de nuevo la tabla declarada —la **importa** de "
            "`cfdtrader.analysis.cost_audit`— ni toca `config/cost_observations.yaml`"
        ),
    },
    {
        "id": "no_es_13",
        "issue": "#13",
        "statement": (
            "no recorre sesiones, no pide *snapshots* *point-in-time*, no toca `bid`/`ask`, "
            "no modela el *gap* de apertura y no simula precios"
        ),
    },
    {
        "id": "no_decide_r",
        "issue": "#60",
        "statement": (
            "no decide `R`: lo consume y publica `null` mientras el tamaño de `R` no esté "
            "decidido (decisión abierta 5)"
        ),
    },
    {
        "id": "no_calcula_la_puerta_de_fase_0",
        "issue": "#9",
        "statement": (
            "no calcula `p* = (R + c) / 2R` ni emite veredicto de Fase 0: solo publica `c` "
            "en las unidades que #9 consume"
        ),
    },
    {
        "id": "no_calcula_sizing",
        "issue": "#27",
        "statement": (
            "no calcula el nocional, el *sizing*, el apalancamiento, el riesgo por operación, "
            "la distancia al stop ni los límites de pérdida (`plan.md` §3.4)"
        ),
    },
    {
        "id": "no_lee_almacen",
        "issue": "#13",
        "statement": (
            "no lee el almacén de datos, ni el calendario de sesiones, ni ningún precio: es "
            "una función pura"
        ),
    },
)

#: Limitaciones publicadas en el JSON y en el Markdown (A34). No se esconden.
LIMITATIONS: Final[tuple[str, ...]] = (
    "**La tabla declarada es declarativa, no una medición**: su origen es el documento del "
    "bróker recogido en `plan.md` §3.3 y la reproduce #8; el bróker definitivo sigue sin "
    "decidir (#59), y de él saldrán la comisión real, el diferencial real y la hora de "
    "corte de la financiación.",
    "**El corte de financiación sigue sin verificar** (`cut_et: null`, `state: "
    '"unmeasured"`) y asumir una hora de corte fija está prohibido (#8, #59): si el corte '
    "cae antes del cierre, el intradía puro paga tenencia igualmente. El motor no lo deduce "
    "de ningún *timestamp* y #12/#13 tampoco pueden inferirlo.",
    "**El *slippage* es un supuesto pesimista declarado** (20 % de `R`, "
    "`is_measurement: false`), **no** una medición: medirlo es #62 y el tamaño de `R` sigue "
    "pendiente (#60). El supuesto no cierra el total y ningún camino lo convierte en "
    "`measured`.",
    "**El diferencial se cobra constante**: el ensanchamiento por tramo de sesión y por "
    "tamaño que #8 mide **no** se cobra aquí, con el sesgo que eso introduce (#66).",
    "**El precio de ejecución real (bid/ask del `SPX500:CFD`) y la fuente intradía son "
    "#50**, y la verificación del diferencial sobre datos limpios es #52: aquí no se lee "
    "ningún precio.",
    "**Las decisiones declaradas aquí se documentan en `plan.md` y `tech_stack.md` desde "
    "#65** (incluida la frontera «#8 mide / #11 cobra»).",
)

#: Seguimientos abiertos (A28).
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#66",
        "topic": "cobrar el diferencial por tramo de sesión y por tamaño",
        "why": "hoy se cobra el declarado constante; #8 ya sabe medir el ensanchamiento",
    },
    {
        "issue": "#50",
        "topic": "fuente intradía y bid/ask real del `SPX500:CFD`",
        "why": "traería el diferencial asimétrico entrada/salida y el real por tramo",
    },
    {
        "issue": "#52",
        "topic": "verificación del `open` repetido de `^GSPC`",
        "why": "la muestra limpia sobre la que se sostienen las cifras declaradas",
    },
    {
        "issue": "#59",
        "topic": "bróker definitivo",
        "why": "de ahí salen la comisión real, el diferencial real y la hora de corte",
    },
    {
        "issue": "#60",
        "topic": "umbrales y tamaño de `R`",
        "why": "sin `R` decidido el supuesto de *slippage* no se puede cobrar",
    },
    {
        "issue": "#62",
        "topic": "medir el *slippage* real",
        "why": "10–15 ejecuciones en la apertura; es lo que reabre la mitad (b) de la puerta",
    },
    {
        "issue": "#65",
        "topic": "documentar estas decisiones en `plan.md`/`tech_stack.md`",
        "why": "incluida la frontera «#8 mide / #11 cobra»",
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados (A5): la raíz, el modelo y la entrada
# ─────────────────────────────────────────────────────────────────────────────
class CostError(Exception):
    """Raíz de los errores del motor de costes."""


class CostModelError(CostError):
    """El modelo de coste o el *slippage* están mal declarados (no es un fallo de entrada)."""


class CostInputError(CostError):
    """Los datos de **esta** operación (nocional, noches, motivo) no son admisibles."""


# ─────────────────────────────────────────────────────────────────────────────
# SlippageParameter: tres estados, nunca fusionados (A16, A17, A18)
# ─────────────────────────────────────────────────────────────────────────────
class SlippageParameter(BaseModel):
    """El *slippage* de ejecución, como parámetro obligatorio de primera clase.

    Tres estados **mutuamente distinguibles** que ninguna función, bloque ni informe
    fusiona ni normaliza:

    - ``measured``: hay medición. ``is_measurement is True`` y ``pct_of_notional`` con
      valor; el total se puede cerrar.
    - ``assumed``: hay un **supuesto declarado**. ``is_measurement is False``, con
      ``pct_of_r``, procedencia, motivo y fecha; ``pct_of_notional`` es ``None`` mientras
      ``R`` no esté decidido (#60) y por eso el total **no** se cierra.
    - ``unmeasured``: no hay nada. ``pct_of_notional is None`` y el motivo es obligatorio.
      Un ``Decimal(0)`` en este estado es un ``CostModelError``: «no medido» nunca se
      escribe como ``0``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: MeasureState = Field(description="measured / assumed / unmeasured")
    is_measurement: bool | None = Field(
        default=None,
        description="True solo si hay medición; False en un supuesto; None si no hay nada",
    )
    pct_of_notional: Decimal | None = Field(
        default=None, description="% del nocional, solo con state = measured"
    )
    pct_of_r: Decimal | None = Field(
        default=None, description="ratio sobre R declarada (solo con state = assumed)"
    )
    r_pct: Decimal | None = Field(
        default=None, description="tamaño de R en %: null mientras #60 no lo decida"
    )
    source: str | None = Field(default=None, description="procedencia declarada")
    reason: str = Field(description="motivo: obligatorio en los tres estados")
    decided_on: str | None = Field(default=None, description="fecha de la decisión (assumed)")
    follow_up_issue: str | None = Field(
        default=None, description="issue que cierra el hueco cuando el valor no es medido"
    )

    def model_post_init(self, _context: object, /) -> None:
        """Reglas de coherencia de los tres estados (A16, A17)."""
        if not {"state", "reason"} <= set(self.model_fields_set):
            return  # modelo construido sin validar: lo rechaza el motor (A4)
        if not self.reason.strip():
            raise CostModelError("slippage.reason: un *slippage* sin motivo no se admite")
        if self.state is MeasureState.MEASURED:
            if self.is_measurement is not True:
                raise CostModelError(
                    "slippage.is_measurement: con state = 'measured' hay una medición, así que "
                    "is_measurement tiene que ser True (los tres estados no se fusionan)"
                )
            if self.pct_of_notional is None:
                raise CostModelError(
                    "slippage.pct_of_notional: con state = 'measured' hace falta el valor en % del "
                    "nocional; sin él no hay medición que cobrar"
                )
            if not self.source:
                raise CostModelError(
                    "slippage.source: una medición exige procedencia (de dónde sale el número)"
                )
            return
        if self.state is MeasureState.ASSUMED:
            if self.is_measurement is not False:
                raise CostModelError(
                    "slippage.is_measurement: con state = 'assumed' hay un supuesto, no una "
                    "medición, así que is_measurement tiene que ser False"
                )
            if self.pct_of_r is None:
                raise CostModelError(
                    "slippage.pct_of_r: un supuesto se declara como ratio sobre `R`; sin ella no "
                    "hay nada declarado"
                )
            if self.pct_of_notional is not None:
                raise CostModelError(
                    "slippage.pct_of_notional: un supuesto **no** es un % del nocional; si `R` "
                    "está decidido, declara el término como medido (state = 'measured')"
                )
            if self.r_pct is not None:
                raise CostModelError(
                    "slippage.r_pct: el tamaño de `R` no se decide aquí (decisión abierta 5, #60): "
                    "el supuesto se declara como ratio sobre `R` y `R` viaja como null"
                )
            if not (self.source and self.decided_on):
                raise CostModelError(
                    "slippage.source/slippage.decided_on: un supuesto exige procedencia y fecha de "
                    "la decisión"
                )
            return
        # state = unmeasured
        if self.pct_of_notional is not None:
            raise CostModelError(
                "slippage.pct_of_notional: con state = 'unmeasured' no hay valor y no se puede "
                "rellenar con 0 ni con un número: «no medido» es null, nunca 0"
            )
        if self.pct_of_r is not None or self.is_measurement is not None:
            raise CostModelError(
                "slippage.pct_of_r/slippage.is_measurement: con state = 'unmeasured' no hay ni "
                "supuesto ni medición: los dos van a null"
            )

    @classmethod
    def measured(
        cls,
        *,
        pct_of_notional: Decimal,
        source: str,
        reason: str,
        follow_up_issue: str | None = None,
    ) -> Self:
        """Un *slippage* **medido** (el único estado que cierra el total)."""
        return cls.model_validate(
            {
                "state": MeasureState.MEASURED,
                "is_measurement": True,
                "pct_of_notional": pct_of_notional,
                "source": source,
                "reason": reason,
                "follow_up_issue": follow_up_issue,
            }
        )

    @classmethod
    def assumed(
        cls,
        *,
        pct_of_r: Decimal,
        source: str,
        reason: str,
        decided_on: str,
        follow_up_issue: str | None = None,
    ) -> Self:
        """Un **supuesto pesimista declarado**: ratio sobre ``R``, ``is_measurement`` False."""
        return cls.model_validate(
            {
                "state": MeasureState.ASSUMED,
                "is_measurement": False,
                "pct_of_r": pct_of_r,
                "source": source,
                "reason": reason,
                "decided_on": decided_on,
                "follow_up_issue": follow_up_issue,
            }
        )

    @classmethod
    def unmeasured(
        cls, *, reason: str, source: str | None = None, follow_up_issue: str | None = None
    ) -> Self:
        """Sin medición ni supuesto: valor ``null`` y motivo obligatorio."""
        return cls.model_validate(
            {
                "state": MeasureState.UNMEASURED,
                "reason": reason,
                "source": source,
                "follow_up_issue": follow_up_issue,
            }
        )


# ─────────────────────────────────────────────────────────────────────────────
# FinancingCut: el corte es un parámetro declarado y hoy está sin verificar (A14, A15)
# ─────────────────────────────────────────────────────────────────────────────
def _financing_cut_question() -> str:
    """La pregunta literal al bróker, **importada** de #8 (no se reescribe aquí)."""
    for item in BROKER_QUESTIONS:
        if item["id"] == "financing_cut":
            return item["question"]
    raise CostModelError(
        "BROKER_QUESTIONS: #8 ya no publica la pregunta del corte de financiación "
        "(id 'financing_cut'): no se reescribe aquí, se arregla en #8"
    )


class FinancingCut(BaseModel):
    """El corte de financiación como **entrada declarada**, jamás derivada.

    Hoy vale ``None`` con ``state: "unmeasured"``: la pregunta al bróker está sin
    contestar (#59) y asumir una hora de corte fija está prohibido. El módulo **no** tiene
    ninguna función que deduzca las noches a partir de *timestamps*, del corte o de una
    duración: ``nights`` es siempre una entrada explícita del llamante.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: MeasureState = Field(description="measured si el bróker contestó; unmeasured hoy")
    cut_et: time | None = Field(default=None, description="instante de corte en hora ET, o null")
    source: str | None = Field(default=None, description="procedencia del corte")
    reason: str = Field(description="por qué está en ese estado")
    broker_question: str = Field(description="la pregunta literal al bróker (#8)")
    note: str = Field(description="aviso de lo que no se puede inferir")

    @classmethod
    def unverified(cls) -> Self:
        """El corte **sin verificar**: ``cut_et: null`` y ``state: "unmeasured"`` (A14)."""
        declared: object = DECLARED_FINANCING_CUT
        if declared is not None:
            raise CostModelError(
                "DECLARED_FINANCING_CUT: #8 ya no lo declara como `None`; este módulo no "
                "adivina el corte (se arregla en #8 / #59)"
            )
        return cls.model_validate(
            {
                "state": MeasureState.UNMEASURED,
                "cut_et": None,
                "source": "#8 (cfdtrader.analysis.cost_audit.DECLARED_FINANCING_CUT = None)",
                "reason": "el corte de financiación sigue sin verificar: el bróker no ha "
                "contestado (#59) y asumir una hora de corte fija está prohibido",
                "broker_question": _financing_cut_question(),
                "note": (
                    "`cut_et` es `null` y **no** se asume: si el corte cae antes del cierre, el "
                    "intradía puro paga tenencia igualmente. Ni #12 (purga y embargo) ni #13 "
                    "(motor *walk-forward*) pueden inferirlo, y el motor no lo deduce de ningún "
                    "*timestamp*"
                ),
            }
        )

    @classmethod
    def verified(cls, *, cut_et: time, source: str, reason: str | None = None) -> Self:
        """Un corte **declarado** por el llamante (y entonces sí, ``state: "measured"``)."""
        return cls.model_validate(
            {
                "state": MeasureState.MEASURED,
                "cut_et": cut_et,
                "source": source,
                "reason": reason
                or "corte declarado por el llamante (el bróker contestó); no lo deduce el motor",
                "broker_question": _financing_cut_question(),
                "note": (
                    "el intradía puro solo queda libre de tenencia si este corte es posterior al "
                    "cierre de la sesión; el motor **no** lo comprueba ni lo deriva"
                ),
            }
        )


# ─────────────────────────────────────────────────────────────────────────────
# CostModel: los costes son siempre una entrada explícita (A3, A4, A9, A10, A11)
# ─────────────────────────────────────────────────────────────────────────────
_COST_MODEL_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "spread_entry_pct",
    "spread_exit_pct",
    "carry_long_pct_per_night",
    "carry_short_pct_per_night",
    "fx_pct",
    "commission_pct",
)


class CostModel(BaseModel):
    """Los costes declarados de una operación. **Ningún campo tiene valor por defecto.**

    ``spread_entry_pct`` y ``spread_exit_pct`` son **dos** mitades: el diferencial se cobra
    media al entrar y media al salir. Un diferencial **asimétrico** es declarable solo si
    cada mitad lleva su ``source`` (el bid/ask real del CFD es #50). Un ``fx_pct`` distinto
    de 0 exige ``fx_source`` y ``fx_reason`` (la exposición de divisa es #27).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    spread_entry_pct: Decimal = Field(description="mitad del diferencial que se cobra al entrar")
    spread_exit_pct: Decimal = Field(description="mitad del diferencial que se cobra al salir")
    carry_long_pct_per_night: Decimal = Field(description="tenencia en largo, % por noche")
    carry_short_pct_per_night: Decimal = Field(description="tenencia en corto, % por noche")
    fx_pct: Decimal = Field(description="coste de conversión de divisa")
    commission_pct: Decimal = Field(description="comisión explícita; nunca se rellena sola")
    name: str = Field(default="modelo de coste declarado", description="nombre del modelo")
    spread_entry_source: str | None = Field(
        default=None, description="procedencia de la mitad de entrada"
    )
    spread_exit_source: str | None = Field(
        default=None, description="procedencia de la mitad de salida"
    )
    carry_source: str = Field(
        default=SOURCE_DECLARED_TABLE, description="procedencia de la tenencia"
    )
    fx_state: MeasureState = Field(
        default=MeasureState.MEASURED, description="estado del coste de divisa"
    )
    fx_source: str | None = Field(default=None, description="procedencia del coste de divisa")
    fx_reason: str | None = Field(default=None, description="motivo del coste de divisa")
    commission_state: MeasureState = Field(
        default=MeasureState.MEASURED, description="estado de la comisión"
    )
    commission_source: str | None = Field(default=None, description="procedencia de la comisión")
    commission_reason: str | None = Field(default=None, description="motivo de la comisión")

    def model_post_init(self, _context: object, /) -> None:
        """Coherencia del modelo (A9, A10). No rellena ningún coste por su cuenta."""
        if not set(_COST_MODEL_REQUIRED_FIELDS) <= set(self.model_fields_set):
            return  # modelo construido sin validar: lo rechaza el motor (A4)
        for name in _COST_MODEL_REQUIRED_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                kind = type(value).__name__
                raise CostModelError(
                    f"model.{name}: los costes son `decimal.Decimal` exactos, no {kind}"
                )
        if self.spread_entry_pct != self.spread_exit_pct and not (
            self.spread_entry_source and self.spread_exit_source
        ):
            raise CostModelError(
                "model.spread_entry_source/model.spread_exit_source: un diferencial asimétrico "
                "(spread_entry_pct != spread_exit_pct) exige la `source` de **cada** mitad; el "
                "bid/ask real del CFD es #50"
            )
        if self.fx_pct != Decimal(0) and not (self.fx_source and self.fx_reason):
            raise CostModelError(
                "model.fx_pct: un coste de divisa distinto de 0 exige `fx_source` y `fx_reason`; "
                "la exposición de divisa (cuenta en EUR, nocional en USD) es #27"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CostBreakdown: el resultado del motor
# ─────────────────────────────────────────────────────────────────────────────
class CostBreakdown(BaseModel):
    """El coste de **una** operación, con cada término y su estado.

    Todo importe va en ``%`` del nocional y en ``$`` sobre el nocional declarado. ``null``
    es «no medido» y nunca se sustituye por ``0``: cuando el *slippage* no es un número, el
    total (``c_total_pct``/``c_total_usd``) es ``None`` y el hueco se publica en ``nulls``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    side: Side = Field(description="lado de la operación: la tenencia es asimétrica")
    nights: int = Field(description="noches declaradas, entrada explícita (0 = intradía puro)")
    overnight_reason: str | None = Field(default=None, description="motivo declarado de la noche")
    notional_usd: Decimal = Field(description="nocional declarado: el coste escala con él")
    spread_entry_pct: Decimal = Field(description="mitad del diferencial, entrada")
    spread_exit_pct: Decimal = Field(description="mitad del diferencial, salida")
    spread_entry_source: str | None = Field(
        default=None, description="procedencia de la mitad de entrada"
    )
    spread_exit_source: str | None = Field(
        default=None, description="procedencia de la mitad de salida"
    )
    spread_entry_usd: Decimal = Field(description="mitad del diferencial en $ sobre el nocional")
    spread_exit_usd: Decimal = Field(description="mitad del diferencial en $ sobre el nocional")
    spread_pct: Decimal = Field(description="diferencial total = entrada + salida")
    spread_usd: Decimal = Field(description="diferencial total en $")
    carry_pct_per_night: Decimal = Field(description="tenencia declarada por noche, con signo")
    carry_pct: Decimal = Field(description="tenencia total: por noche x noches")
    carry_usd: Decimal = Field(description="tenencia total en $")
    carry_state: MeasureState = Field(description="estado de la tenencia")
    carry_source: str = Field(description="procedencia de la tenencia")
    carry_reason: str = Field(description="motivo, citando la regla 6 cuando nights = 0")
    fx_pct: Decimal = Field(description="coste de divisa en %")
    fx_usd: Decimal = Field(description="coste de divisa en $")
    fx_state: MeasureState = Field(description="estado del coste de divisa")
    fx_source: str | None = Field(default=None, description="procedencia del coste de divisa")
    fx_reason: str | None = Field(default=None, description="motivo del coste de divisa")
    commission_pct: Decimal = Field(description="comisión explícita en %")
    commission_usd: Decimal = Field(description="comisión explícita en $")
    commission_state: MeasureState = Field(description="estado de la comisión")
    commission_source: str | None = Field(default=None, description="procedencia de la comisión")
    commission_reason: str | None = Field(default=None, description="motivo de la comisión")
    c_declared_pct: Decimal = Field(description="coste declarado de la operación (sin *slippage*)")
    c_declared_usd: Decimal = Field(description="coste declarado en $")
    slippage: SlippageParameter = Field(description="el *slippage* **tal cual** se declaró")
    slippage_pct: Decimal | None = Field(
        default=None, description="% del nocional, solo si es medido"
    )
    slippage_usd: Decimal | None = Field(
        default=None, description="$ sobre el nocional, solo si es medido"
    )
    c_total_pct: Decimal | None = Field(
        default=None, description="total; null mientras el *slippage* no sea medido"
    )
    c_total_usd: Decimal | None = Field(
        default=None, description="total en $; null por el mismo motivo"
    )
    slippage_over_spread_ratio: Decimal | None = Field(
        default=None,
        description="bp de *slippage* / bp de diferencial del tramo; null si no hay número",
    )
    slippage_dominates: bool | None = Field(
        default=None, description="True si el *slippage* pesa más que el diferencial"
    )
    financing_cut: FinancingCut = Field(description="el corte de financiación declarado")
    illustrative_equivalence: dict[str, str] | None = Field(
        default=None, description="equivalencia ilustrativa del supuesto; nunca alimenta el total"
    )
    nulls: tuple[dict[str, str], ...] = Field(
        default=(), description="términos del total que hoy son null, con su motivo"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Escenarios declarados (se construyen con las constantes **importadas** de #8)
# ─────────────────────────────────────────────────────────────────────────────
def declared_cost_model() -> CostModel:
    """El modelo de coste del escenario **declarado** (``plan.md`` §3.3, vía #8).

    Las cifras se toman **tal cual** de ``cfdtrader.analysis.cost_audit``: el diferencial
    se parte en sus dos mitades, la tenencia es la declarada (asimétrica), la divisa es el
    0 con motivo de #8 y la comisión se declara explícitamente en 0 con su procedencia.
    """
    return CostModel(
        name="tabla declarada de `plan.md` §3.3 (vía #8)",
        spread_entry_pct=DECLARED_SPREAD_HALF_PCT,
        spread_exit_pct=DECLARED_SPREAD_HALF_PCT,
        carry_long_pct_per_night=CARRY_LONG_PCT_PER_NIGHT,
        carry_short_pct_per_night=CARRY_SHORT_PCT_PER_NIGHT,
        fx_pct=FX_COST_PCT,
        commission_pct=DECLARED_COMMISSION_PCT,
        spread_entry_source=DECLARED_SPREAD_HALF_SOURCE,
        spread_exit_source=DECLARED_SPREAD_HALF_SOURCE,
        carry_source=SOURCE_DECLARED_TABLE,
        fx_state=MeasureState.MEASURED,
        fx_source=f"#8 (FX_COST_PCT) · {SOURCE_USER}",
        fx_reason=(
            f"nocional liquidado en {DECLARED_SETTLEMENT_CURRENCY}: el coste de conversión es 0 "
            f"**con motivo**, confirmado por el usuario el {WINDOW_CONFIRMED_ON.isoformat()} "
            "(procedencia: #8). La exposición de divisa es #27"
        ),
        commission_state=MeasureState.MEASURED,
        commission_source=DECLARED_COMMISSION_SOURCE,
        commission_reason=DECLARED_COMMISSION_REASON,
    )


def declared_slippage_assumption() -> SlippageParameter:
    """El supuesto pesimista **declarado** por el propietario el 2026-09-18 (#64).

    ``state = "assumed"``, ``is_measurement = False``, valor **100 % del margen de la
    puerta (b) = 20 % de ``R``**, ``pct_of_notional = None`` y ``r_pct = None``: el tamaño
    de ``R`` sigue pendiente (#60) y ningún camino convierte el supuesto en ``measured``.
    """
    return SlippageParameter.assumed(
        pct_of_r=SLIPPAGE_ASSUMPTION_PCT_OF_R,
        source=SLIPPAGE_ASSUMPTION_PROVENANCE,
        reason=(
            "el propietario decidió no medir el *slippage* todavía y seguir adelante con un "
            "**supuesto pesimista declarado** en vez de un valor de relleno: medirlo exige 10-15 "
            "ejecuciones reales en la apertura (#62). Es una asunción, **no** una medición"
        ),
        decided_on=SLIPPAGE_ASSUMPTION_DECIDED_ON,
        follow_up_issue=SLIPPAGE_ASSUMPTION_R_ISSUE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades exactas
# ─────────────────────────────────────────────────────────────────────────────
def _num(value: Decimal | None) -> str | None:
    """``Decimal`` → cadena decimal **exacta**, sin notación científica y sin ceros de relleno."""
    if value is None:
        return None
    if value == Decimal(0):
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _text(value: str | None) -> str:
    """``None`` → ``null`` (en el informe en prosa, sin inventar un 0)."""
    return "`null`" if value is None else value


def _usd(notional_usd: Decimal, pct: Decimal) -> Decimal:
    """El coste en $ **de este** nocional: ``usd = notional * pct / 100``, exacto."""
    return notional_usd * pct / Decimal(100)


def _require_decimal(value: object, *, field: str) -> Decimal:
    """Un importe tiene que ser ``Decimal`` (nada de ``float``) y estar declarado."""
    if not isinstance(value, Decimal):
        raise CostInputError(
            f"{field}: hay que declararlo como `decimal.Decimal` exacto (recibido "
            f"{type(value).__name__}); un `float` rompería la comparación exacta"
        )
    if not value.is_finite():
        raise CostInputError(f"{field}: un valor no finito no es un importe declarable")
    return value


def _require_positive(value: object, *, field: str) -> Decimal:
    """Un nocional (o un ``R`` ilustrativo) tiene que ser positivo."""
    amount = _require_decimal(value, field=field)
    if amount <= Decimal(0):
        raise CostInputError(f"{field}: tiene que ser mayor que 0 (recibido {_num(amount)})")
    return amount


def _require_int(value: object, *, field: str) -> int:
    """Las noches son un entero declarado, nunca un booleano ni un decimal disfrazado."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CostInputError(
            f"{field}: tiene que ser un entero explícito (recibido {type(value).__name__})"
        )
    return value


def _require_validated(
    model: CostModel | SlippageParameter, *, required: tuple[str, ...], field: str
) -> None:
    """Rechaza un modelo construido sin validar (``model_construct``) — A4."""
    missing = sorted(name for name in required if name not in model.model_fields_set)
    if missing:
        raise CostModelError(
            f"{field}: falta declarar {', '.join(missing)}; un modelo construido sin validar "
            "(`model_construct`) no entra al motor y ningún coste tiene valor de relleno"
        )


def _carry_reason(*, side: Side, nights: int, overnight_reason: str | None) -> str:
    """El motivo de la tenencia: la regla 6 cuando no hay noche, el declarado cuando la hay."""
    if nights == 0:
        return (
            "0 noches: no se cobra tenencia porque el intradía puro no pasa la noche "
            "(plan.md §12, regla 6) y la posición se cierra a mano en la sesión (regla 16)"
        )
    return (
        f"{nights} noche(s) declarada(s) en {side.value}, con motivo explícito del llamante "
        f"«{overnight_reason}»: la regla 6 prohíbe el *overnight* por defecto y la regla 16 "
        "obliga al cierre manual, así que cada noche hay que justificarla"
    )


def _dominance(
    slippage_pct: Decimal | None, spread_pct: Decimal
) -> tuple[Decimal | None, bool | None]:
    """Ratio *slippage*/diferencial en bp/bp y si el *slippage* domina (A21)."""
    if slippage_pct is None:
        return None, None
    if spread_pct == Decimal(0):
        return None, None
    ratio = (slippage_pct / spread_pct).quantize(RATIO_QUANTUM, rounding=ROUND_HALF_UP)
    return ratio, slippage_pct > spread_pct


def _slippage_null(slippage: SlippageParameter) -> dict[str, str]:
    """El hueco del *slippage* en la lista de ``nulls``, con su motivo (A19)."""
    if slippage.state is MeasureState.MEASURED:
        reason = "el *slippage* está declarado como medido pero sin valor: no debería ocurrir"
    elif slippage.state is MeasureState.ASSUMED:
        reason = (
            "el *slippage* es un supuesto declarado (no una medición) y su ratio sobre `R` no "
            "se puede cobrar mientras el tamaño de `R` no esté decidido (#60); por eso el "
            "total queda null"
        )
    else:
        reason = (
            "el *slippage* no está medido (no hay ninguna ejecución real) y no se rellena "
            "con 0 (#62)"
        )
    return {
        "field": "slippage.pct_of_notional",
        "state": slippage.state.value,
        "reason": reason,
        "blocks": "c_total_pct y c_total_usd",
        "follow_up_issue": slippage.follow_up_issue or "#62",
    }


def _illustrative_equivalence(r_illustrative_pct: Decimal, notional_usd: Decimal) -> dict[str, str]:
    """La equivalencia **ilustrativa** del supuesto: ``illustrative``, ``decision: false`` (A19).

    Nunca alimenta ``c_total``: es la lectura del supuesto bajo un ``R`` de ejemplo, con su
    issue (#60) y su etiqueta. La función es pura: cambiar el ``R`` ilustrativo cambia los
    bp y **no** toca el total.
    """
    pct = SLIPPAGE_ASSUMPTION_PCT_OF_R * r_illustrative_pct / Decimal(100)
    return {
        "illustrative": "true",
        "decision": "false",
        "is_measurement": "false",
        "r_issue": SLIPPAGE_ASSUMPTION_R_ISSUE,
        "r_pct": _num(r_illustrative_pct) or "",
        "pct_of_r": _num(SLIPPAGE_ASSUMPTION_PCT_OF_R) or "",
        "pct_of_notional": _num(pct) or "",
        "bp_of_notional": _num(pct * Decimal(100)) or "",
        "usd_on_notional": _num(_usd(notional_usd, pct)) or "",
        "notional_usd": _num(notional_usd) or "",
        "provenance": "plan.md §4.4 (equivalencia ilustrativa del supuesto declarado)",
        "warning": (
            "es una equivalencia **ilustrativa** bajo un `R` de ejemplo: no es una decisión de `R` "
            "(#60) y **no** entra en `c_total`"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# El motor: función pura y determinista (A4, A7-A22)
# ─────────────────────────────────────────────────────────────────────────────
def cost_breakdown(
    *,
    model: CostModel,
    slippage: SlippageParameter,
    notional_usd: Decimal,
    side: Side,
    nights: int = 0,
    overnight_reason: str | None = None,
    financing_cut: FinancingCut | None = None,
    r_illustrative_pct: Decimal | None = None,
) -> CostBreakdown:
    """El coste de ida y vuelta de **una** operación, exacto y sin red ni reloj.

    ``model`` y ``slippage`` son **obligatorios** (keyword-only, sin valor por defecto):
    sin coste declarado no hay resultado. ``notional_usd`` también es obligatorio y
    ``nights`` vale 0 por defecto (intradía puro). Con ``nights >= 1`` hace falta
    ``overnight_reason``.
    """
    _require_validated(model, required=_COST_MODEL_REQUIRED_FIELDS, field="model")
    _require_validated(slippage, required=("state", "reason"), field="slippage")
    notional = _require_positive(notional_usd, field="notional_usd")
    declared_nights = _require_int(nights, field="nights")
    if declared_nights < 0:
        raise CostInputError(
            f"nights: no puede ser negativo (recibido {declared_nights}); las noches de tenencia "
            "se declaran explícitamente y el caso por defecto es 0 (intradía puro)"
        )
    if declared_nights >= 1 and (overnight_reason is None or not overnight_reason.strip()):
        raise CostInputError(
            "overnight_reason: con nights >= 1 hay que declarar por qué se mantiene la posición "
            "durante una noche. El intradía puro es 0 noches (plan.md §12, regla 6) y el cierre "
            "manual obligatorio de la regla 16 evita la noche por descuido: cada noche se "
            "justifica explícitamente"
        )
    if r_illustrative_pct is not None:
        illustrative_r = _require_positive(r_illustrative_pct, field="r_illustrative_pct")
    else:
        illustrative_r = None
    cut = financing_cut if financing_cut is not None else FinancingCut.unverified()

    carry_per_night = (
        model.carry_short_pct_per_night if side is Side.SHORT else model.carry_long_pct_per_night
    )
    carry_pct = carry_per_night * Decimal(declared_nights)
    spread_pct = model.spread_entry_pct + model.spread_exit_pct
    c_declared_pct = spread_pct + carry_pct + model.fx_pct + model.commission_pct
    slippage_pct = slippage.pct_of_notional
    c_total_pct = None if slippage_pct is None else c_declared_pct + slippage_pct
    ratio, dominates = _dominance(slippage_pct, spread_pct)

    return CostBreakdown(
        side=side,
        nights=declared_nights,
        overnight_reason=overnight_reason,
        notional_usd=notional,
        spread_entry_pct=model.spread_entry_pct,
        spread_exit_pct=model.spread_exit_pct,
        spread_entry_source=model.spread_entry_source,
        spread_exit_source=model.spread_exit_source,
        spread_entry_usd=_usd(notional, model.spread_entry_pct),
        spread_exit_usd=_usd(notional, model.spread_exit_pct),
        spread_pct=spread_pct,
        spread_usd=_usd(notional, spread_pct),
        carry_pct_per_night=carry_per_night,
        carry_pct=carry_pct,
        carry_usd=_usd(notional, carry_pct),
        carry_state=MeasureState.MEASURED,
        carry_source=model.carry_source,
        carry_reason=_carry_reason(
            side=side, nights=declared_nights, overnight_reason=overnight_reason
        ),
        fx_pct=model.fx_pct,
        fx_usd=_usd(notional, model.fx_pct),
        fx_state=model.fx_state,
        fx_source=model.fx_source,
        fx_reason=model.fx_reason,
        commission_pct=model.commission_pct,
        commission_usd=_usd(notional, model.commission_pct),
        commission_state=model.commission_state,
        commission_source=model.commission_source,
        commission_reason=model.commission_reason,
        c_declared_pct=c_declared_pct,
        c_declared_usd=_usd(notional, c_declared_pct),
        slippage=slippage,
        slippage_pct=slippage_pct,
        slippage_usd=None if slippage_pct is None else _usd(notional, slippage_pct),
        c_total_pct=c_total_pct,
        c_total_usd=None if c_total_pct is None else _usd(notional, c_total_pct),
        slippage_over_spread_ratio=ratio,
        slippage_dominates=dominates,
        financing_cut=cut,
        illustrative_equivalence=(
            None if illustrative_r is None else _illustrative_equivalence(illustrative_r, notional)
        ),
        nulls=() if slippage_pct is not None else (_slippage_null(slippage),),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Informe legible por máquina (A25-A27, A34)
# ─────────────────────────────────────────────────────────────────────────────
def _amount_block(
    breakdown: CostBreakdown,
    pct: Decimal | None,
    usd: Decimal | None,
    *,
    unit_pct: str = UNIT_PCT,
) -> dict[str, Any]:
    """Un importe con su unidad y su nocional: sin eso no hay cifra interpretable."""
    return {
        "pct": _num(pct),
        "pct_unit": unit_pct,
        "usd": _num(usd),
        "usd_unit": UNIT_USD,
        "notional_usd": _num(breakdown.notional_usd),
    }


def _scenario_row(
    *,
    label: str,
    breakdown: CostBreakdown,
    slippage_label: str,
) -> dict[str, Any]:
    """Una fila de la tabla declarada, con sus términos, sus totales y su procedencia."""
    return {
        "label": label,
        "direction": breakdown.side.value,
        "nights": str(breakdown.nights),
        "spread": _amount_block(breakdown, breakdown.spread_pct, breakdown.spread_usd),
        "carry": _amount_block(breakdown, breakdown.carry_pct, breakdown.carry_usd),
        "fx": _amount_block(breakdown, breakdown.fx_pct, breakdown.fx_usd),
        "commission": _amount_block(breakdown, breakdown.commission_pct, breakdown.commission_usd),
        "c_declared": _amount_block(breakdown, breakdown.c_declared_pct, breakdown.c_declared_usd),
        "c_total": _amount_block(breakdown, breakdown.c_total_pct, breakdown.c_total_usd),
        "c_total_state": slippage_label,
        "source": SOURCE_DECLARED_TABLE,
    }


def _declared_scenarios() -> tuple[tuple[str, Side, int, str | None], ...]:
    """Los escenarios de la tabla declarada: 0 y 1 noche en las dos direcciones (A7, A8)."""
    nights = HOLDING_NIGHTS
    return (
        ("intradía puro (0 noches)", Side.SHORT, 0, None),
        ("intradía puro (0 noches)", Side.LONG, 0, None),
        (f"corto con {nights} noche", Side.SHORT, nights, DECLARED_OVERNIGHT_REASON),
        (f"largo con {nights} noche", Side.LONG, nights, DECLARED_OVERNIGHT_REASON),
    )


def _row_declared(rows: list[dict[str, Any]], *, side: Side | None, nights: int) -> dict[str, Any]:
    """La fila declarada de un lado y unas noches (o la primera de esas noches)."""
    matches = [
        row
        for row in rows
        if row["nights"] == str(nights) and (side is None or row["direction"] == side.value)
    ]
    if not matches:
        raise CostModelError(
            f"declared_table: no hay fila declarada para side = {side} y nights = {nights}"
        )
    return matches[0]["c_declared"]


def _slippage_block(slippage: SlippageParameter, notional_usd: Decimal) -> dict[str, Any]:
    """El bloque del *slippage* **tal cual** se declaró, con su estado y su motivo."""
    usd = None if slippage.pct_of_notional is None else _usd(notional_usd, slippage.pct_of_notional)
    return {
        "state": slippage.state.value,
        "is_measurement": slippage.is_measurement,
        "required": True,
        "pct_of_notional": _num(slippage.pct_of_notional),
        "pct_of_notional_unit": UNIT_PCT,
        "usd_on_notional": _num(usd),
        "usd_on_notional_unit": UNIT_USD,
        "pct_of_r": _num(slippage.pct_of_r),
        "pct_of_r_unit": "% de `R`",
        "r_pct": _num(slippage.r_pct),
        "r_issue": SLIPPAGE_ASSUMPTION_R_ISSUE,
        "source": slippage.source,
        "reason": slippage.reason,
        "decided_on": slippage.decided_on,
        "follow_up_issue": slippage.follow_up_issue,
        "notional_usd": _num(notional_usd),
        "never_filled_with_zero": (
            "con state = 'unmeasured' el valor es null con motivo: el motor no rellena un 0, "
            "y un 0 solo se admite con state = 'measured', fuente y motivo"
        ),
    }


def _slippage_measured_block(slippage: SlippageParameter, notional_usd: Decimal) -> dict[str, Any]:
    """El bloque del estado ``measured``, **separado** de los otros dos (A16)."""
    if slippage.state is MeasureState.MEASURED:
        return _slippage_block(slippage, notional_usd)
    return {
        "state": MeasureState.MEASURED.value,
        "is_measurement": True,
        "available": False,
        "pct_of_notional": None,
        "source": None,
        "reason": (
            "no hay ninguna ejecución real medida a 2026-09-18: medir el *slippage* "
            "(10-15 ejecuciones en la apertura) es #62. Hasta entonces ningún término "
            "numérico entra al total"
        ),
        "follow_up_issue": "#62",
    }


def _slippage_unmeasured_block() -> dict[str, Any]:
    """El bloque del estado ``unmeasured``, con ``null`` y motivo, **sin ningún 0** (A17)."""
    return {
        "state": MeasureState.UNMEASURED.value,
        "is_measurement": None,
        "pct_of_notional": None,
        "pct_of_r": None,
        "r_pct": None,
        "usd_on_notional": None,
        "source": None,
        "reason": (
            "el *slippage* no está medido: no existe ninguna ejecución real a 2026-09-18 y un "
            "supuesto no es una medición. «No medido» es null con motivo, nunca 0"
        ),
        "how_to_fill": (
            "plan.md §8.5: anotar el precio obtenido frente al de referencia en el instante de la "
            "orden, repetido 10-15 veces en la apertura, con el *timestamp* de cada intento (#62)"
        ),
        "follow_up_issue": "#62",
    }


def _spread_block(model: CostModel, notional_usd: Decimal) -> dict[str, Any]:
    """Las **dos mitades** del diferencial por separado, su suma y el contraste con #8 (A9)."""
    halves_usd = _usd(REFERENCE_NOTIONAL_USD, model.spread_entry_pct) + _usd(
        REFERENCE_NOTIONAL_USD, model.spread_exit_pct
    )
    return {
        "entry": {
            "pct": _num(model.spread_entry_pct),
            "usd": _num(_usd(notional_usd, model.spread_entry_pct)),
            "source": model.spread_entry_source,
        },
        "exit": {
            "pct": _num(model.spread_exit_pct),
            "usd": _num(_usd(notional_usd, model.spread_exit_pct)),
            "source": model.spread_exit_source,
        },
        "total_pct": _num(model.spread_entry_pct + model.spread_exit_pct),
        "total_usd": _num(_usd(notional_usd, model.spread_entry_pct + model.spread_exit_pct)),
        "pct_unit": UNIT_PCT,
        "usd_unit": UNIT_USD,
        "notional_usd": _num(notional_usd),
        "cross_check_against_8": {
            "halves_usd_on_reference_notional": _num(halves_usd),
            "declared_spread_usd": _num(SPREAD_USD),
            "matches_declared": str(halves_usd == SPREAD_USD).lower(),
            "reference_notional_usd": _num(REFERENCE_NOTIONAL_USD),
            "note": (
                "las dos mitades se suman **exactamente** al diferencial declarado en #8 sobre el "
                "nocional de referencia; si no coincide, el modelo no es el declarado"
            ),
        },
        "asymmetry_rule": (
            "un diferencial asimétrico (entrada != salida) es declarable **solo** si cada mitad "
            "lleva su `source`; el bid/ask real del `SPX500:CFD` es #50 y queda fuera de alcance"
        ),
    }


def _carry_block(breakdown: CostBreakdown) -> dict[str, Any]:
    """La tenencia: asimétrica, exacta y con la convención de signo publicada (A12)."""
    long_usd = CARRY_LONG_USD_PER_NIGHT
    short_usd = CARRY_SHORT_USD_PER_NIGHT
    return {
        "long_pct_per_night": _num(CARRY_LONG_PCT_PER_NIGHT),
        "short_pct_per_night": _num(CARRY_SHORT_PCT_PER_NIGHT),
        "long_usd_per_night": _num(long_usd),
        "short_usd_per_night": _num(short_usd),
        "pct_per_night_unit": UNIT_PCT_PER_NIGHT,
        "usd_per_night_unit": UNIT_USD,
        "sign_convention": "negativo = el lado **cobra**; positivo = el lado **paga**",
        "cross_check_against_8": {
            "long_usd_on_reference_notional": _num(
                _usd(REFERENCE_NOTIONAL_USD, CARRY_LONG_PCT_PER_NIGHT)
            ),
            "short_usd_on_reference_notional": _num(
                _usd(REFERENCE_NOTIONAL_USD, CARRY_SHORT_PCT_PER_NIGHT)
            ),
            "matches_declared": str(
                _usd(REFERENCE_NOTIONAL_USD, CARRY_LONG_PCT_PER_NIGHT) == long_usd
                and _usd(REFERENCE_NOTIONAL_USD, CARRY_SHORT_PCT_PER_NIGHT) == short_usd
            ).lower(),
            "reference_notional_usd": _num(REFERENCE_NOTIONAL_USD),
            "note": (
                "las dos representaciones declaradas en #8 (% por noche y $ sobre el nocional de "
                "referencia) tienen que coincidir exactamente: si no, el modelo no es el declarado"
            ),
        },
        "applied": _amount_block(breakdown, breakdown.carry_pct, breakdown.carry_usd),
        "state": breakdown.carry_state.value,
        "source": breakdown.carry_source,
        "reason": breakdown.carry_reason,
        "nights": str(breakdown.nights),
        "overnight_reason": breakdown.overnight_reason,
        "nights_note": (
            "`nights` es siempre una entrada explícita del llamante: el motor no lo deduce de "
            "ningún *timestamp*, del corte de financiación ni de una duración"
        ),
    }


def _financing_cut_block(cut: FinancingCut) -> dict[str, Any]:
    """El corte de financiación con su estado y lo que no se puede inferir (A14, A15)."""
    return {
        "state": cut.state.value,
        "cut_et": None if cut.cut_et is None else cut.cut_et.isoformat(),
        "source": cut.source,
        "reason": cut.reason,
        "broker_question": cut.broker_question,
        "note": cut.note,
        "sixteen_is_not_assumed": "asumir una hora de corte fija está prohibido",
    }


def _commission_block(model: CostModel, breakdown: CostBreakdown) -> dict[str, Any]:
    """La comisión explícita, con su procedencia: nunca la rellena el motor (A11)."""
    return {
        "pct": _num(model.commission_pct),
        "usd": _num(breakdown.commission_usd),
        "state": model.commission_state.value,
        "source": model.commission_source,
        "reason": model.commission_reason,
        "notional_usd": _num(breakdown.notional_usd),
        "filled_by_the_engine": False,
        "note": (
            "la comisión es un campo **obligatorio** del modelo: si falta, la construcción del "
            "`CostModel` no valida; el motor nunca la rellena por su cuenta"
        ),
    }


def _fx_block(model: CostModel, breakdown: CostBreakdown) -> dict[str, Any]:
    """La divisa: un 0 **con motivo**, nunca un cero mudo (A10)."""
    return {
        "pct": _num(model.fx_pct),
        "usd": _num(breakdown.fx_usd),
        "state": model.fx_state.value,
        "source": model.fx_source,
        "reason": model.fx_reason,
        "notional_usd": _num(breakdown.notional_usd),
        "settlement_currency": DECLARED_SETTLEMENT_CURRENCY,
        "confirmed_on": WINDOW_CONFIRMED_ON.isoformat(),
        "out_of_scope": (
            "la **exposición** de divisa (cuenta en EUR, nocional en USD) es distinta del coste de "
            "conversión y es #27"
        ),
    }


def _units_block(breakdown: CostBreakdown) -> dict[str, Any]:
    """Las unidades en las que #9 consume ``c``: % del nocional y fracción (A26)."""
    return {
        "usd_unit": UNIT_USD,
        "pct_unit": UNIT_PCT,
        "pct_per_night_unit": UNIT_PCT_PER_NIGHT,
        "bp_unit": UNIT_BP,
        "fraction_of_notional": "fracción del nocional (pct / 100), la unidad de `R` y de `c`",
        "ratio_unit": RATIO_UNIT,
        "c_pct_of_notional": _num(breakdown.c_declared_pct),
        "c_fraction_of_notional": _num(breakdown.c_declared_pct / Decimal(100)),
        "c_note": (
            "`c` son los costes declarados de esta operación; `c_total` añade el *slippage* "
            "**cuando está medido** y si no, queda `null`"
        ),
        "note": (
            "el cálculo de `p* = (R + c) / 2R` y el veredicto de Fase 0 son **#9**: aquí solo se "
            "publica `c` en las unidades que #9 consume (% del nocional y fracción)"
        ),
    }


def report_payload(
    *,
    model: CostModel | None = None,
    slippage: SlippageParameter | None = None,
    notional_usd: Decimal = REFERENCE_NOTIONAL_USD,
    side: Side | None = None,
    nights: int = 0,
    overnight_reason: str | None = None,
    financing_cut: FinancingCut | None = None,
    r_illustrative_pct: Decimal | None = None,
) -> dict[str, Any]:
    """El informe completo como *mapping* serializable con ``json.dumps`` (A25-A27, A34).

    Sin argumentos devuelve el escenario **declarado**: el modelo de #8 y el supuesto
    pesimista de #64 (``state: assumed``), con los tres estados del *slippage* publicados
    **por separado** y el total en ``null``.
    """
    cost_model = model if model is not None else declared_cost_model()
    term = slippage if slippage is not None else declared_slippage_assumption()
    cut = financing_cut if financing_cut is not None else FinancingCut.unverified()
    scenario_side = side if side is not None else Side.LONG
    scenario = cost_breakdown(
        model=cost_model,
        slippage=term,
        notional_usd=notional_usd,
        side=scenario_side,
        nights=nights,
        overnight_reason=overnight_reason,
        financing_cut=cut,
        r_illustrative_pct=r_illustrative_pct,
    )
    rows = [
        _scenario_row(
            label=label,
            slippage_label=term.state.value,
            breakdown=cost_breakdown(
                model=cost_model,
                slippage=term,
                notional_usd=notional_usd,
                side=row_side,
                nights=row_nights,
                overnight_reason=row_reason,
                financing_cut=cut,
            ),
        )
        for label, row_side, row_nights, row_reason in _declared_scenarios()
    ]
    ratio, dominates = scenario.slippage_over_spread_ratio, scenario.slippage_dominates
    return {
        "task": "#11",
        "title": (
            "Motor de costes: tabla declarada de `plan.md` §3.3 (importada de #8) y "
            "*slippage* explícito"
        ),
        "does_not_do": [dict(item) for item in COST_MODEL_DOES_NOT_DO],
        "units": _units_block(scenario),
        "scenario": {
            "direction": scenario.side.value,
            "nights": str(scenario.nights),
            "overnight_reason": scenario.overnight_reason,
            "notional_usd": _num(scenario.notional_usd),
            "notional_usd_unit": UNIT_USD,
            "slippage_state": term.state.value,
        },
        "declared_table": {
            "source": SOURCE_DECLARED_TABLE,
            "reference_notional_usd": _num(REFERENCE_NOTIONAL_USD),
            "holding_nights": str(HOLDING_NIGHTS),
            "scenarios": rows,
            "round_trip": {
                "label": "totales de ida y vuelta (corto y largo)",
                "intraday_pure": {
                    "night_pure": "0 noches en las dos direcciones",
                    "usd": _row_declared(rows, side=None, nights=0)["usd"],
                    "pct": _row_declared(rows, side=None, nights=0)["pct"],
                },
                "with_one_night": {
                    "short": _row_declared(rows, side=Side.SHORT, nights=HOLDING_NIGHTS),
                    "long": _row_declared(rows, side=Side.LONG, nights=HOLDING_NIGHTS),
                },
            },
            "note": (
                "la tabla es el documento del bróker recogido en `plan.md` §3.3 y la reproduce #8; "
                "aquí se **importa** para que no existan dos definiciones de «coste declarado»"
            ),
        },
        "spread": _spread_block(cost_model, notional_usd),
        "carry": _carry_block(scenario),
        "fx": _fx_block(cost_model, scenario),
        "commission": _commission_block(cost_model, scenario),
        "financing_cut": _financing_cut_block(cut),
        "slippage": _slippage_block(term, notional_usd),
        "slippage_measured": _slippage_measured_block(term, notional_usd),
        "slippage_assumption": slippage_assumption_block(),
        "slippage_unmeasured": _slippage_unmeasured_block(),
        "slippage_dominance": {
            "slippage_over_spread_ratio": _num(ratio),
            "ratio_unit": RATIO_UNIT,
            "ratio_quantum": _num(RATIO_QUANTUM),
            "slippage_dominates": dominates,
            "justification": SLIPPAGE_DOMINANCE_JUSTIFICATION,
            "note": (
                "la ratio se calcula con los términos **numéricos**: si el *slippage* no es un "
                "número (supuesto o no medido) la ratio y el booleano son `null`, no 0"
            ),
        },
        "illustrative_equivalence": scenario.illustrative_equivalence,
        "illustrative_equivalence_note": (
            "la equivalencia ilustrativa se publica **solo** si el llamante pasa "
            f"`r_illustrative_pct`; el `R` ilustrativo declarado en #8 es "
            f"{_num(R_ILLUSTRATIVE_PCT)} % y **no** se usa por defecto porque `R` no está "
            "decidido (#60)"
        ),
        "c_declared": _amount_block(scenario, scenario.c_declared_pct, scenario.c_declared_usd),
        "c_total": _amount_block(scenario, scenario.c_total_pct, scenario.c_total_usd),
        "nulls": [dict(item) for item in scenario.nulls],
        "nulls_note": (
            "`nulls` lista los **términos del total** que hoy son `null`, con su motivo: el "
            "corte de financiación y el supuesto del *slippage* se publican en sus propios "
            "bloques con su `state` y su motivo. Un `null` nunca se sustituye por `0`"
        ),
        "limitations": list(LIMITATIONS),
        "follow_ups": [dict(item) for item in FOLLOW_UPS],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Informe en prosa (A28)
# ─────────────────────────────────────────────────────────────────────────────
def _row_line(row: dict[str, Any]) -> str:
    """Una fila de la tabla declarada en Markdown, con unidades y procedencia."""
    c_total = f"{_text(row['c_total']['usd'])} $ · {_text(row['c_total']['pct'])} %"
    return (
        f"| {row['label']} | `{row['direction']}` | {row['nights']} | "
        f"{row['spread']['usd']} $ · {row['spread']['pct']} % | "
        f"{row['carry']['usd']} $ · {row['carry']['pct']} % | "
        f"{row['fx']['usd']} $ · {row['fx']['pct']} % | "
        f"{row['commission']['usd']} $ · {row['commission']['pct']} % | "
        f"**{row['c_declared']['usd']} $ · {row['c_declared']['pct']} %** | {c_total} | "
        f"`{row['c_total_state']}` | {row['source']} |"
    )


def _slippage_row(block: dict[str, Any]) -> str:
    """Una fila de la tabla de los tres estados del *slippage*."""
    value = block.get("pct_of_notional")
    if value is not None:
        rendered = f"{value} % · {_text(block.get('usd_on_notional'))} $"
    elif block.get("pct_of_r") is not None:
        rendered = f"{block['pct_of_r']} % de `R`"
    else:
        rendered = "`null` (no medido)"
    return (
        f"| `{block['state']}` | {rendered} | {_text(block.get('source'))} | "
        f"{block.get('reason')} | {_text(block.get('follow_up_issue'))} |"
    )


def _dominates_text(value: bool | None) -> str:
    """``True``/``False``/``null`` en minúsculas, porque un `null` no es un `False`."""
    return "null" if value is None else str(value).lower()


def _illustrative_text(payload: dict[str, Any]) -> str:
    """La equivalencia ilustrativa del supuesto, en una línea (o su ausencia declarada)."""
    block = payload["illustrative_equivalence"]
    if block is None:
        return "no publicada (no se ha pasado `r_illustrative_pct`)"
    return (
        f"{block['pct_of_notional']} % = {block['bp_of_notional']} bp = "
        f"{block['usd_on_notional']} $ con `R` = {block['r_pct']} % "
        f"(illustrative: true, decision: false, {block['r_issue']})"
    )


def render_markdown(payload: dict[str, Any] | None = None) -> str:
    """El informe en prosa: la tabla declarada, el *slippage* y lo que queda abierto (A28)."""
    report = report_payload() if payload is None else payload
    declared = report["declared_table"]
    spread = report["spread"]
    carry = report["carry"]
    cut = report["financing_cut"]
    dominance = report["slippage_dominance"]
    assertion = report["slippage_assumption"]
    lines: list[str] = [
        "# Motor de costes (#11): tabla declarada y *slippage* explícito",
        "",
        f"- **Escenario publicado:** `{report['scenario']['direction']}` · "
        f"{report['scenario']['nights']} noches · nocional "
        f"{report['scenario']['notional_usd']} {report['scenario']['notional_usd_unit']}",
        f"- **Fuente de la tabla:** {declared['source']}",
        f"- **Nocional de referencia:** {declared['reference_notional_usd']} $",
        "",
        "Este módulo **cobra**; medir y auditar los costes declarados es **#8**, de donde se "
        "importan las cifras. El bucle de sesiones es **#13** y el cálculo de `p*` es **#9**.",
        "",
        "## Tabla declarada (`plan.md` §3.3, reproducida por #8)",
        "",
        "Todos los importes en `$` sobre el nocional de la operación y en `%` del nocional. La "
        "comparación con la tabla declarada es **exacta** (`Decimal`), no aproximada.",
        "",
        "| escenario | lado | noches | diferencial | tenencia | divisa | comisión | "
        "coste declarado | total con *slippage* | estado del total | procedencia |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(_row_line(row) for row in declared["scenarios"])
    round_trip = declared["round_trip"]
    lines.extend(
        [
            "",
            f"**Totales de ida y vuelta:** intradía puro (0 noches) "
            f"{round_trip['intraday_pure']['usd']} $ · {round_trip['intraday_pure']['pct']} % · "
            f"corto con 1 noche {round_trip['with_one_night']['short']['usd']} $ · "
            f"{round_trip['with_one_night']['short']['pct']} % · largo con 1 noche "
            f"{round_trip['with_one_night']['long']['usd']} $ · "
            f"{round_trip['with_one_night']['long']['pct']} %.",
            "",
            "## Las dos mitades del diferencial",
            "",
            f"- Entrada: **{spread['entry']['pct']} %** ({spread['entry']['usd']} $) — "
            f"{_text(spread['entry']['source'])}",
            f"- Salida: **{spread['exit']['pct']} %** ({spread['exit']['usd']} $) — "
            f"{_text(spread['exit']['source'])}",
            f"- Suma: **{spread['total_pct']} %** ({spread['total_usd']} $) sobre el nocional de "
            f"la operación. Contraste con #8 sobre el nocional de referencia "
            f"({spread['cross_check_against_8']['reference_notional_usd']} $): "
            f"{spread['cross_check_against_8']['halves_usd_on_reference_notional']} $ = "
            f"{spread['cross_check_against_8']['declared_spread_usd']} $ "
            f"(`matches_declared`: `{spread['cross_check_against_8']['matches_declared']}`).",
            f"- {spread['asymmetry_rule']}",
            "",
            "## Tenencia (*carry*): asimétrica y con signo declarado",
            "",
            "| lado | % por noche | $ por noche |",
            "|---|---|---|",
            f"| corto | {carry['short_pct_per_night']} | {carry['short_usd_per_night']} |",
            f"| largo | {carry['long_pct_per_night']} | {carry['long_usd_per_night']} |",
            "",
            f"- Convención publicada: **{carry['sign_convention']}**.",
            f"- Aplicado en este escenario ({carry['nights']} noches): "
            f"{carry['applied']['pct']} {carry['applied']['pct_unit']} = "
            f"{carry['applied']['usd']} $ · estado `{carry['state']}`.",
            f"- {carry['reason']}",
            f"- {carry['nights_note']}",
            f"- Contraste con #8 sobre el nocional de referencia: "
            f"`matches_declared`: `{carry['cross_check_against_8']['matches_declared']}`.",
            "",
            "## Divisa, comisión y corte de financiación",
            "",
            f"- **Divisa:** {report['fx']['pct']} % · {report['fx']['usd']} $ · estado "
            f"`{report['fx']['state']}` — {report['fx']['reason']} "
            f"({_text(report['fx']['source'])})",
            f"  - {report['fx']['out_of_scope']}",
            f"- **Comisión:** {report['commission']['pct']} % · {report['commission']['usd']} $ · "
            f"estado `{report['commission']['state']}` — {report['commission']['reason']} "
            f"({_text(report['commission']['source'])})",
            f"- **Corte de financiación:** `state`: `{cut['state']}` · `cut_et`: "
            f"{_text(cut['cut_et'])} — {cut['reason']}",
            f"  - {cut['sixteen_is_not_assumed']} y el motor no lo deduce de ningún *timestamp*.",
            f"  - Pregunta literal al bróker: «{cut['broker_question']}»",
            "",
            "## Los tres estados del *slippage* (nunca se fusionan)",
            "",
            "| estado | valor | fuente | motivo | seguimiento |",
            "|---|---|---|---|---|",
        ]
    )
    lines.extend(
        _slippage_row(report[key])
        for key in ("slippage_measured", "slippage_assumption", "slippage_unmeasured")
    )
    lines.extend(
        [
            "",
            f"- El supuesto declarado del propietario (**#64**) es "
            f"`{assertion['value_pct_of_r']} {assertion['value_pct_of_r_unit']}` con "
            f"`is_measurement`: `{str(assertion['is_measurement']).lower()}` y `R` pendiente "
            f"(`r_pct`: `null`, {assertion['r_issue']}). {assertion['assumption_note']}.",
            "- Limitaciones declaradas de ese supuesto: 59 sesiones, **un solo régimen** y "
            "granularidad de **5 min**, medido sobre el índice `^GSPC` y **no** sobre el CFD.",
            f"- Equivalencia **ilustrativa**: {_illustrative_text(report)} — **nunca** "
            "alimenta el total.",
            "",
            "## Dominancia del *slippage*",
            "",
            f"- `slippage_over_spread_ratio`: **{_text(dominance['slippage_over_spread_ratio'])}** "
            f"({dominance['ratio_unit']}, cuantizada a {dominance['ratio_quantum']}).",
            f"- `slippage_dominates`: **{_dominates_text(dominance['slippage_dominates'])}**.",
            f"- Justificación declarada: {dominance['justification']}",
            f"- {dominance['note']}",
            "",
            "## Nulos declarados",
            "",
            f"`c_declared` = {report['c_declared']['usd']} $ · {report['c_declared']['pct']} % "
            f"({report['c_declared']['pct_unit']}). `c_total` = "
            f"{_text(report['c_total']['usd'])} $ · {_text(report['c_total']['pct'])} %.",
            "",
            "| campo | estado | motivo | bloquea | seguimiento |",
            "|---|---|---|---|---|",
        ]
    )
    for item in report["nulls"]:
        lines.append(
            f"| `{item['field']}` | `{item['state']}` | {item['reason']} | {item['blocks']} | "
            f"{item['follow_up_issue']} |"
        )
    if not report["nulls"]:
        lines.append("| — | — | no hay términos nulos: el total se cierra | — | — |")
    lines.extend(["", f"{report['nulls_note']}", "", "## Qué no hace este módulo", ""])
    for item in report["does_not_do"]:
        lines.append(f"- `{item['id']}` ({item['issue']}): {item['statement']}.")
    lines.extend(["", "## Limitaciones (declaradas, no escondidas)", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["", "## Seguimientos abiertos", ""])
    lines.extend(
        f"- `{item['issue']}` — {item['topic']}: {item['why']}." for item in report["follow_ups"]
    )
    lines.extend(
        [
            "",
            "## Unidades que consume #9",
            "",
            f"- `c` = {report['units']['c_pct_of_notional']} {report['units']['pct_unit']} = "
            f"{report['units']['c_fraction_of_notional']} "
            f"({report['units']['fraction_of_notional']}).",
            f"- {report['units']['note']}",
            "",
        ]
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI: el único punto de entrada/salida, y explícito (A29)
# ─────────────────────────────────────────────────────────────────────────────
def _decimal_from_text(text: str, *, field: str) -> Decimal:
    """Un número declarado en la línea de órdenes, parseado a ``Decimal`` exacto."""
    try:
        return Decimal(text)
    except InvalidOperation as error:
        raise CostInputError(f"{field}: no es un número decimal válido ({text!r})") from error


def _slippage_from_args(args: argparse.Namespace) -> SlippageParameter:
    """El *slippage* declarado en la CLI, con sus dos formas de fallar (A29)."""
    state = MeasureState(args.slippage_state)
    if state is MeasureState.MEASURED:
        if args.slippage_bp is None:
            raise CostInputError(
                "slippage: has pedido el término **medido** (--slippage-state measured) pero "
                "no has dado el número: pasa --slippage-bp, o declara el supuesto con "
                "--slippage-state assumed"
            )
        pct = _decimal_from_text(args.slippage_bp, field="--slippage-bp") / Decimal(100)
        return SlippageParameter.measured(
            pct_of_notional=pct,
            source="--slippage-bp (medición declarada por el llamante de la CLI)",
            reason=args.slippage_reason
            or "término declarado como medido en la CLI; el motor no comprueba la medición",
        )
    if args.slippage_bp is not None:
        raise CostInputError(
            f"slippage: has pasado un término numérico (--slippage-bp = {args.slippage_bp}) con "
            f"state = {state.value}: el motor solo admite un término numérico con "
            "state = measured, porque un supuesto o un «no medido» no tienen valor"
        )
    if state is MeasureState.ASSUMED:
        return declared_slippage_assumption()
    return SlippageParameter.unmeasured(
        reason="declarado como no medido en la CLI: no hay ninguna ejecución real (#62)"
    )


def _scenario_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Las entradas del motor, tal cual las declara el llamante de la CLI."""
    return {
        "model": declared_cost_model(),
        "slippage": _slippage_from_args(args),
        "notional_usd": _decimal_from_text(args.notional_usd, field="--notional-usd"),
        "side": None if args.side is None else Side(args.side),
        "nights": args.nights,
        "overnight_reason": args.overnight_reason,
        "r_illustrative_pct": (
            None
            if args.illustrative_r_pct is None
            else _decimal_from_text(args.illustrative_r_pct, field="--illustrative-r-pct")
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Imprime la tabla declarada y los tres estados del *slippage*.

    Códigos de salida: ``0`` = informe impreso (y escrito si se pidió ``--out-dir``);
    ``2`` = entrada inválida (noches negativas, nocional <= 0, una noche sin motivo, un
    término numérico que no está declarado como medido o un término medido sin número) ⇒
    **no** se escribe ningún fichero y el motivo sale por ``stderr``. Sin ``--out-dir`` no
    se escribe **nada**: el motor no consulta el reloj, así que para escribir hay que
    declarar la fecha con ``--as-of``.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.backtest.costs",
        description=(
            "Motor de costes: tabla declarada de `plan.md` §3.3 (importada de #8) y los tres "
            "estados del *slippage*. Sin --out-dir no escribe nada"
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="directorio donde escribir el informe"
    )
    parser.add_argument(
        "--as-of", default=None, help="fecha del informe (AAAA-MM-DD), para --out-dir"
    )
    parser.add_argument(
        "--nights", type=int, default=0, help="noches declaradas (0 = intradía puro)"
    )
    parser.add_argument("--overnight-reason", default=None, help="motivo, obligatorio si hay noche")
    parser.add_argument(
        "--notional-usd",
        default=_num(REFERENCE_NOTIONAL_USD),
        help="nocional declarado en $ (por defecto, el de referencia de #8)",
    )
    parser.add_argument("--side", choices=[side.value for side in Side], default=None, help="lado")
    parser.add_argument(
        "--slippage-state",
        choices=[state.value for state in MeasureState],
        default=MeasureState.ASSUMED.value,
        help="estado del *slippage* (por defecto, el supuesto declarado de #64)",
    )
    parser.add_argument(
        "--slippage-bp",
        default=None,
        help="término medido en bp del nocional (1 bp = 0,01 %), solo con measured",
    )
    parser.add_argument("--slippage-reason", default=None, help="motivo del término medido")
    parser.add_argument(
        "--illustrative-r-pct",
        default=None,
        help="`R` ilustrativo para la equivalencia del supuesto (nunca alimenta el total)",
    )
    args = parser.parse_args(argv)

    destination: Path | None = None
    day: date | None = None
    if args.out_dir is not None:
        if args.as_of is None:
            print(
                "--out-dir: para escribir el informe hay que declarar la fecha con --as-of "
                "(AAAA-MM-DD): este motor es puro y no consulta el reloj",
                file=sys.stderr,
            )
            return 2
        try:
            day = date.fromisoformat(args.as_of)
        except ValueError:
            print(
                f"--as-of: no es una fecha ISO válida (AAAA-MM-DD): {args.as_of!r}",
                file=sys.stderr,
            )
            return 2
        destination = Path(args.out_dir)

    try:
        payload = report_payload(**_scenario_from_args(args))
    except CostError as error:
        print(f"entrada inválida: {error}", file=sys.stderr)
        return 2

    markdown = render_markdown(payload)
    print(markdown)
    if destination is not None and day is not None:
        stem = f"costs_{day.isoformat()}"
        json_path = destination / f"{stem}.json"
        markdown_path = destination / f"{stem}.md"
        destination.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(markdown + "\n", encoding="utf-8")
        logger.info("informe de costes: {} y {}", json_path, markdown_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
