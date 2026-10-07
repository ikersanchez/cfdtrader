"""Listón **B** de primera clase: la serie de referencia «cierre a cierre» — tarea #70.

Frontera explícita: **este módulo declara la referencia B; no es un motor nuevo ni un
séptimo baseline**. `plan.md` §11.2 separa **tres listones** para «siempre largo» y
confundirlos invalida el análisis:

- **A.** `siempre largo` ``open`` -> cierre — implementado en ``cfdtrader.backtest.baselines``
  (``always_long``): mismo horizonte y mismo instrumento, **solo el diferencial**.
- **B.** `siempre largo` **``close`` -> ``close``** (aguantar el CFD) — **este módulo**. Carga
  con la **financiación** del CFD (``plan.md`` §11.2: ≈6,66 %/año) y por eso es el listón que
  revela si el sistema aporta valor frente a la alternativa real.
- **C.** índice puro ``^GSPC`` — **no invertible**, es una **referencia** y no un baseline
  (``#28``).

Por qué B **no** se corre por el motor de #13
---------------------------------------------

El motor de ``cfdtrader.backtest.engine`` (#13) es **intradía puro por construcción**
(``plan.md`` §12, regla 6): entra en el ``open`` de la subasta, sale **dentro de la misma
sesión** y llama a ``cost_breakdown`` con ``nights = 0`` y ``overnight_reason = None`` fijos.
De ahí el invariante de #13 (``tests/test_engine.py`` A17: ``nights=1`` **no** aparece en su
código). Una posición que **cruza la noche** —lo que B necesita— exige reabrir ese
invariante, y eso es una **decisión de dueño**, no una tarea de implementación. La issue #70
admite explícitamente la alternativa: *«…o bien una **serie de referencia ``close`` -> ``close``
calculada fuera de #13** y declarada como tal»*. Este módulo entrega esa serie como artefacto
**de primera clase** —tipado, puro, determinista y con la financiación **declarada**— y deja
declarado (en ``CLOSE_TO_CLOSE_DOES_NOT_DO``) que el listón B «por el motor» sigue siendo lo
único pendiente de #70.

Qué produce
-----------

Para **cada** sesión de *test*, una ``ReferenceRow`` (sin mirar el futuro de la sesión):

- ``gross_pct`` = ``100 · (close / close_previo - 1)`` — el retorno cierre a cierre, en %.
- ``financing_pct`` = la **tenencia declarada por noche** en largo, importada del
  ``CostModel`` del llamante: ``1`` noche por sesión, porque el tránsito de cierre a cierre
  **cruza el corte verificado de #87** (17:00 ET, posterior al cierre de las 16:00 ET).
- ``net_pct`` = ``gross_pct - financing_pct`` — la serie de referencia que se compara con A.

El **diferencial** (una vez, en la entrada y la salida) y la **tenencia de todo el tramo** no
se reparten por sesión: se publican **juntos** en ``declared_hold_cost``, el
``CostBreakdown`` **exacto** que produce el motor de costes de #11 con ``nights =
nights_total``. Así la serie por sesión no duplica el coste.

Equivalencia de la financiación (lo que #70 pide declarar)
----------------------------------------------------------

``plan.md`` §11.2 cita **6,66 %/año**; la tabla declarada de #11 (§3.3, vía #8) da
``+0,0182 %`` **por noche** en largo. No son la misma cifra: la anualización exacta de #8 es
**6,6647 %/año** (ratio ``366,2`` noches/año) y **no** se deriva de ``0,0182 · 365`` (que da
``6,6430``, distinto). Este módulo **usa el valor por-noche exacto** (el ``Decimal`` del
``CostModel``) y **declara** la equivalencia en ``FINANCING_EQUIVALENCE`` — con ``used =
"per_night"`` y la cifra de §11.2 marcada como redondeo—. No se reescribe ninguna cifra de la
tabla: la tenencia entra siempre por el ``CostModel`` del llamante.

Pureza y determinismo
---------------------

Solo biblioteca estándar y los contratos de #11/#13 (nada de ``polars``/``duckdb``/``numpy``,
**nada** de ``cfdtrader.analysis``/``cfdtrader.data``): no consulta el reloj, no toca el disco
y no lee el ``Store``. Dos entradas idénticas dan el mismo ``series_sha256`` byte a byte,
también con ``PYTHONHASHSEED`` distinto. El corte de financiación y el nocional son
**entradas explícitas**: el módulo no deduce una hora de corte ni un tamaño.

Qué **no** hace (cada frontera con su issue)
-------------------------------------------

Viaja, legible por máquina, en ``CLOSE_TO_CLOSE_DOES_NOT_DO`` y ``FOLLOW_UPS``: no es un
motor (#13), no calcula métricas netas (#15), no es la corrida real ni la tabla comparativa
(#18), no lee el almacén (#69), no implementa el listón C (#28) y —lo declarado— el listón B
«**por el motor**» (posiciones overnight con ``nights >= 1`` en #13) seguiría exigiendo
reabrir la regla 6, que es decisión de dueño.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal
from typing import Final, cast

from cfdtrader.backtest.costs import (
    CostBreakdown,
    CostModel,
    FinancingCut,
    Side,
    SlippageParameter,
    cost_breakdown,
)
from cfdtrader.backtest.engine import SessionInput

__all__ = [
    "CLOSE_TO_CLOSE",
    "CLOSE_TO_CLOSE_DOES_NOT_DO",
    "FINANCING_CUT_ET",
    "FINANCING_CUT_SOURCE",
    "FINANCING_EQUIVALENCE",
    "FOLLOW_UPS",
    "LIMITATIONS",
    "NIGHTS_PER_SESSION",
    "OVERNIGHT_REASON",
    "CloseToCloseError",
    "InvalidCloseToCloseParameterError",
    "ReferenceRow",
    "ReferenceSeries",
    "close_to_close_series",
    "declared_financing_cut",
]

#: El identificador estable del listón **B** de §11.2. **No** entra en ``BASELINE_IDS`` de #14:
#: los seis identificadores de #14 son solo A y las otras cinco reglas, y B «por el motor»
#: es la frontera declarada de #70.
CLOSE_TO_CLOSE: Final[str] = "close_to_close_long"

#: Cada sesión de la referencia B se mantiene **una** noche: el tránsito de cierre a cierre
#: cruza el corte de financiación verificado (#87). Es explícito, nunca deducido de un
#: *timestamp*.
NIGHTS_PER_SESSION: Final[int] = 1

#: Motivo declarado de la noche (lo exige ``cost_breakdown`` con ``nights >= 1``).
OVERNIGHT_REASON: Final[str] = (
    "listón B (plan.md §11.2): posición **larga** mantenida de cierre a cierre —se aguanta el "
    "CFD—, así que cada sesión cruza el corte de financiación verificado en #87 (17:00 ET, "
    "posterior al cierre de las 16:00 ET) y paga la tenencia declarada por noche"
)

#: Corte de financiación **declarado** (no deducido): las 17:00 ``America/New_York`` que el
#: propietario verificó en #87. El módulo no lee el YAML (es puro); ``declared_financing_cut``
#: lo entrega ya validado para que el llamante no asuma una hora.
FINANCING_CUT_ET: Final[time] = time(17, 0)
FINANCING_CUT_SOURCE: Final[str] = (
    "#87 (2026-10-07): el *swap* se cobra a las 17:00 ET, el cierre diario técnico de los "
    "futuros de EE. UU. Depositado en config/cost_observations.yaml (`financing_cut`)"
)

#: La equivalencia que #70 manda **declarar**, no derivar. §11.2 cita 6,66 %/año; la tabla
#: declarada (#11, vía #8) da +0,0182 %/noche, cuya anualización exacta es 6,6647 %/año
#: (ratio 366,2). El módulo usa el valor **por noche**.
FINANCING_EQUIVALENCE: Final[dict[str, str]] = {
    "per_night_pct": "0.0182",
    "per_year_pct": "6.6647",
    "plan_112_pct": "6.66",
    "nights_per_year": "366.2",
    "derived_from_daily": "false",
    "used": "per_night",
    "provenance": (
        "plan.md §3.3 y §11.2 vía #8 (cost_audit.ANNUALISED_LONG_PCT); 0,0182 % × 365 = "
        "6,6430 % ≠ 6,6647 %, así que la anualización **no** se deriva del valor por noche"
    ),
}

#: Qué **no** hace el módulo, legible por máquina. Cada frontera con su issue.
CLOSE_TO_CLOSE_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_es_un_motor",
        "issue": "#13",
        "statement": (
            "no es un motor: no recorre sesiones de *train*, no pide *snapshots* ni simula una "
            "operación intradía. Recorre las sesiones de *test* una vez y publica la serie de "
            "referencia; la única vía de ejecución intradía sigue siendo #13"
        ),
    },
    {
        "id": "no_es_el_liston_b_por_el_motor",
        "issue": "#70",
        "statement": (
            "el listón B **de primera clase con posiciones overnight *por el motor*** (``nights "
            ">= 1`` en #13) no se implementa aquí: #13 es intradía puro por construcción (regla "
            "6, A17) y reabrir ese invariante es decisión de dueño. Lo que se entrega es la "
            "**serie de referencia** close->close declarada, que la issue admite como alternativa"
        ),
    },
    {
        "id": "no_calcula_metricas",
        "issue": "#15",
        "statement": (
            "no calcula Sharpe, drawdown ni ningún estadístico: publica la serie y sus "
            "recuentos; las métricas netas, con intervalo *bootstrap*, son #15"
        ),
    },
    {
        "id": "no_es_la_corrida_real",
        "issue": "#18",
        "statement": (
            "no es la corrida real sobre el histórico ni la tabla comparativa de §11.2: es la "
            "serie de B que esa tabla consume; el informe de Fase 1 es #18"
        ),
    },
    {
        "id": "no_lee_el_almacen",
        "issue": "#69",
        "statement": (
            "no lee el Store ni construye los SessionInput: los recibe del llamante; sin el "
            "adaptador #69 solo se ejercita con entradas sintéticas"
        ),
    },
    {
        "id": "no_implementa_el_liston_c",
        "issue": "#28",
        "statement": (
            "no implementa el listón C (índice puro ^GSPC) porque no es invertible: es una "
            "referencia, no un baseline, y la separación alpha/beta es #28"
        ),
    },
)

#: Seguimientos abiertos que este módulo deja declarados.
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#15",
        "topic": "metricas netas de la serie de referencia",
        "why": "aquí solo se construye la serie; el rendimiento se mide allí",
    },
    {
        "issue": "#18",
        "topic": "tabla comparativa real (A y B juntas, C como referencia)",
        "why": "es el informe de Fase 1 y su puerta de salida; su prerrequisito duro es #69",
    },
    {
        "issue": "#69",
        "topic": "adaptador Store -> SessionInput",
        "why": "sin él, la serie solo se ejercita con entradas sintéticas",
    },
)

#: Limitaciones que el módulo publica. No se esconden.
LIMITATIONS: Final[tuple[str, ...]] = (
    "**B es una serie de referencia, no una corrida del motor**: sale **fuera** de #13 porque "
    "una posición que cruza la noche exige reabrir la regla 6 (decisión de dueño). El listón "
    "«por el motor» (posiciones overnight con `nights >= 1` en #13) es lo único que #70 deja "
    "abierto.",
    "**La serie por sesión resta la tenencia, no el diferencial**: el `financing_pct` por "
    "sesión es la tenencia declarada por noche en largo (una noche por sesión). El diferencial "
    "de entrada y salida —una sola vez, porque la posición se aguanta— y la tenencia del tramo "
    "completo van **juntos** en `declared_hold_cost` (el `CostBreakdown` de #11), para no "
    "duplicar el coste ni repartirlo sesión a sesión.",
    "**El nocional y el corte de financiación son entradas explícitas**: el módulo no deriva "
    "un tamaño ni asume una hora de corte. Igual que #11, ``nights`` nunca se deduce de un "
    "*timestamp*: cada sesión vale exactamente una noche porque el tránsito cierre a cierre "
    "cruza el corte verificado (#87).",
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados: nunca un resultado silencioso
# ─────────────────────────────────────────────────────────────────────────────
class CloseToCloseError(Exception):
    """Raíz de los errores de la serie de referencia del listón B."""


class InvalidCloseToCloseParameterError(CloseToCloseError):
    """Una entrada no declarable (tipo equivocado, sesiones desordenadas, nocional no positivo)."""


# ─────────────────────────────────────────────────────────────────────────────
# El resultado: una fila por sesión y la serie agregada
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ReferenceRow:
    """Una sesión de la serie B. Los términos sin precio o sin cierre previo van a ``None``.

    ``None`` es «no calculable con lo declarado», **nunca** un ``0``: una sesión sin cierre
    previo usable no se rellena. ``nights`` es ``1`` cuando la fila se mantiene una noche (hay
    cierre y cierre previo) y ``0`` cuando no hay tránsito que sostener.
    """

    session: date
    previous_close_px: float | None
    close_px: float | None
    nights: int
    gross_pct: float | None
    financing_pct: float | None
    net_pct: float | None
    note: str | None


@dataclass(frozen=True, slots=True)
class ReferenceSeries:
    """La serie de referencia B completa, con su coste declarado y su huella determinista.

    ``declared_hold_cost`` es el ``CostBreakdown`` **exacto** de #11 para la posición mantenida
    completa (diferencial una vez + tenencia por ``nights_total`` noches). ``gross_pct`` y
    ``net_pct`` van alineados con ``rows`` (un valor por fila, ``None`` donde no es calculable).
    """

    identifier: str
    rows: tuple[ReferenceRow, ...]
    gross_pct: tuple[float | None, ...]
    net_pct: tuple[float | None, ...]
    nights_total: int
    financing_pct_per_night: str
    financing_cut: FinancingCut
    financing_equivalence: dict[str, str]
    declared_hold_cost: CostBreakdown
    series_sha256: str


# ─────────────────────────────────────────────────────────────────────────────
# El corte declarado (no deducido) y las comprobaciones de entrada
# ─────────────────────────────────────────────────────────────────────────────
def declared_financing_cut() -> FinancingCut:
    """El corte de financiación **declarado** (17:00 ET, verificado en #87).

    El módulo es puro: no lee ``config/cost_observations.yaml`` ni inventa una hora. Entrega
    el corte ya validado para que el llamante no asuma uno; el motor de #13 recibe lo mismo
    como entrada explícita.
    """
    return FinancingCut.verified(
        cut_et=FINANCING_CUT_ET,
        source=FINANCING_CUT_SOURCE,
        reason=(
            "el propietario verificó el corte en #87 (2026-10-07); es **posterior** al cierre "
            "de las 16:00 ET, así que el intradía puro no paga tenencia y el tránsito de cierre "
            "a cierre del listón B sí paga una noche"
        ),
    )


def _require_session_input(value: object, *, field: str) -> SessionInput:
    """Un elemento de la secuencia es un ``SessionInput`` de #13, tal cual."""
    if not isinstance(value, SessionInput):
        raise InvalidCloseToCloseParameterError(
            f"{field}: se espera un SessionInput (recibido {type(value).__name__})"
        )
    return value


def _require_inputs(value: object) -> tuple[SessionInput, ...]:
    """Una secuencia **no vacía** de ``SessionInput`` con sesiones estrictamente crecientes."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InvalidCloseToCloseParameterError(
            "inputs: se espera una secuencia de SessionInput (recibido "
            f"{type(value).__name__}); el listón B exige el contrato de #13"
        )
    items = tuple(
        _require_session_input(item, field=f"inputs[{position}]")
        for position, item in enumerate(cast("Sequence[object]", value))
    )
    if not items:
        raise InvalidCloseToCloseParameterError(
            "inputs: la serie de referencia necesita al menos una sesión de *test*"
        )
    for position in range(len(items) - 1):
        if items[position].session >= items[position + 1].session:
            raise InvalidCloseToCloseParameterError(
                "inputs: las sesiones tienen que ser estrictamente crecientes y sin "
                f"duplicados; se rompe en la posicion {position + 1}"
            )
    return items


def _require_cost_model(value: object) -> CostModel:
    """El modelo de coste de #11 se consume tal cual: no se reescribe la tabla de §3.3."""
    if not isinstance(value, CostModel):
        raise InvalidCloseToCloseParameterError(
            f"cost_model: se espera el CostModel de #11 (recibido {type(value).__name__})"
        )
    return value


def _require_slippage(value: object) -> SlippageParameter:
    """El *slippage* de #11 viaja con su estado: el módulo no inventa un valor."""
    if not isinstance(value, SlippageParameter):
        raise InvalidCloseToCloseParameterError(
            f"slippage: se espera el SlippageParameter de #11 (recibido {type(value).__name__})"
        )
    return value


def _require_cut(value: object) -> FinancingCut:
    """El corte de financiación es una entrada declarada, nunca deducida."""
    if not isinstance(value, FinancingCut):
        raise InvalidCloseToCloseParameterError(
            f"financing_cut: se espera un FinancingCut (recibido {type(value).__name__})"
        )
    return value


def _require_notional(value: object) -> Decimal:
    """El nocional tiene que ser ``Decimal`` **positivo**: nada de ``float`` ni de relleno."""
    if not isinstance(value, Decimal):
        raise InvalidCloseToCloseParameterError(
            "notional_usd: hay que declararlo como `decimal.Decimal` exacto (recibido "
            f"{type(value).__name__})"
        )
    if not value.is_finite() or value <= Decimal(0):
        raise InvalidCloseToCloseParameterError(
            f"notional_usd: tiene que ser un importe finito > 0 (recibido {value})"
        )
    return value


def _require_price(value: object, *, field: str) -> float | None:
    """Un precio declarado es ``float`` finito y positivo, o ``None``; un ``0`` no es usable."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCloseToCloseParameterError(
            f"{field}: un precio es un número o `None` (recibido {type(value).__name__})"
        )
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number <= 0.0:
        raise InvalidCloseToCloseParameterError(
            f"{field}: un precio tiene que ser finito y > 0 (recibido {value!r})"
        )
    return number


# ─────────────────────────────────────────────────────────────────────────────
# La serie de referencia (pura y determinista)
# ─────────────────────────────────────────────────────────────────────────────
def close_to_close_series(
    inputs: Sequence[SessionInput],
    *,
    notional_usd: Decimal,
    cost_model: CostModel,
    slippage: SlippageParameter,
    financing_cut: FinancingCut,
    starting_close_px: float | None = None,
) -> ReferenceSeries:
    """Construye la serie de referencia del listón **B** (``close`` -> ``close``), en %.

    ``cost_model`` y ``slippage`` son **obligatorios, keyword-only y sin valor por defecto**
    (como en #13/#14): la tenencia declarada entra por el ``CostModel`` del llamante y el
    módulo no escribe ninguna cifra de la tabla de §3.3. ``financing_cut`` también es
    obligatorio y explícito (no se asume una hora). ``starting_close_px`` da el cierre previo
    de la **primera** sesión; sin él, esa fila queda declarada (no se inventa).
    """
    items = _require_inputs(inputs)
    notional = _require_notional(notional_usd)
    model = _require_cost_model(cost_model)
    slip = _require_slippage(slippage)
    cut = _require_cut(financing_cut)
    start = _require_price(starting_close_px, field="starting_close_px")

    carry_per_night = float(model.carry_long_pct_per_night) * NIGHTS_PER_SESSION
    carry_text = _exact(model.carry_long_pct_per_night)

    rows: list[ReferenceRow] = []
    for index, item in enumerate(items):
        previous = (
            start
            if index == 0
            else _require_price(items[index - 1].close_px, field=f"inputs[{index - 1}].close_px")
        )
        close = _require_price(item.close_px, field=f"inputs[{index}].close_px")
        if previous is None or close is None:
            missing = "cierre previo" if previous is None else "cierre"
            rows.append(
                ReferenceRow(
                    session=item.session,
                    previous_close_px=previous,
                    close_px=close,
                    nights=0,
                    gross_pct=None,
                    financing_pct=None,
                    net_pct=None,
                    note=f"sin {missing} declarado: no hay tránsito cierre a cierre que sostener",
                )
            )
            continue
        gross = 100.0 * (close / previous - 1.0)
        rows.append(
            ReferenceRow(
                session=item.session,
                previous_close_px=previous,
                close_px=close,
                nights=NIGHTS_PER_SESSION,
                gross_pct=gross,
                financing_pct=carry_per_night,
                net_pct=gross - carry_per_night,
                note=None,
            )
        )

    nights_total = sum(row.nights for row in rows)
    hold_cost = cost_breakdown(
        model=model,
        slippage=slip,
        notional_usd=notional,
        side=Side.LONG,
        nights=nights_total,
        overnight_reason=OVERNIGHT_REASON if nights_total >= 1 else None,
        financing_cut=cut,
    )
    equivalence = dict(FINANCING_EQUIVALENCE)
    equivalence["per_night_pct"] = carry_text
    return ReferenceSeries(
        identifier=CLOSE_TO_CLOSE,
        rows=tuple(rows),
        gross_pct=tuple(row.gross_pct for row in rows),
        net_pct=tuple(row.net_pct for row in rows),
        nights_total=nights_total,
        financing_pct_per_night=carry_text,
        financing_cut=cut,
        financing_equivalence=equivalence,
        declared_hold_cost=hold_cost,
        series_sha256=_series_sha256(
            rows=tuple(rows), financing_pct_per_night=carry_text, nights_total=nights_total
        ),
    )


def _exact(value: Decimal) -> str:
    """``Decimal`` -> cadena decimal **exacta**, sin ceros de relleno ni notación científica."""
    if value == Decimal(0):
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _series_sha256(
    *, rows: tuple[ReferenceRow, ...], financing_pct_per_night: str, nights_total: int
) -> str:
    """La huella determinista de la serie: sha256 del texto canónico de las filas declaradas."""
    payload = {
        "identifier": CLOSE_TO_CLOSE,
        "financing_pct_per_night": financing_pct_per_night,
        "nights_per_session": NIGHTS_PER_SESSION,
        "nights_total": nights_total,
        "rows": [
            {
                "session": row.session.isoformat(),
                "previous_close_px": row.previous_close_px,
                "close_px": row.close_px,
                "nights": row.nights,
                "gross_pct": row.gross_pct,
                "financing_pct": row.financing_pct,
                "net_pct": row.net_pct,
            }
            for row in rows
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
