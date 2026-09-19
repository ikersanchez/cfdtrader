"""Correccion por sobreajuste: Deflated Sharpe Ratio y PBO por CSCV (tarea #16).

`plan.md` §11.4 (linea 713) exige «Deflated Sharpe Ratio con el numero real de pruebas»
y «PBO < 20 %»; la referencia declarada (`plan.md` linea 1154) es Bailey y Lopez de Prado.
Este modulo es **puro**: stdlib (`math`, `statistics.NormalDist`) y `numpy`, mas los
contratos ya cerrados de #13 (`canonical_text`, que aqui no hace falta) y #15
(`sharpe_ratio`). No toca disco, no conoce el ``Store``, no importa `cfdtrader.data`,
`duckdb`, `polars` ni `scipy`, y **no** usa `cfdtrader.backtest.splits`: el CSCV del PBO
recombina bloques de la **matriz de retornos de las variantes**, no decide particiones de
entrenamiento (eso es #67). El unico modulo que escribe es
`cfdtrader.analysis.experiment_log`.

Formulas implementadas, literales del contrato estadistico declarado:

    DSR  = Z[ (SR - SR0) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR**2) ]
    SR0  = sqrt(V[SR]) * [ (1 - gamma) * Zinv(1 - 1/N) + gamma * Zinv(1 - 1/(N*e)) ]

donde ``Z`` es la normal estandar (`statistics.NormalDist`, exacta y determinista),
``Zinv`` su inversa, ``gamma`` la constante de Euler-Mascheroni, ``N`` el numero real de
variantes probadas (que **sale del registro de experimentos**, nunca de un parametro
suelto: la derivacion vive en `cfdtrader.analysis.experiment_log`), ``V[SR]`` la varianza
**muestral** (``ddof = 1``) de los Sharpe de esas variantes, ``T`` el numero de retornos,
``g3`` la asimetria y ``g4`` la curtosis **no excedente** (normal ⇒ 3). Los momentos son
**poblacionales**: ``m_k = (1/T) * suma((x - media)**k)``, ``g3 = m_3 / m_2**1.5`` y
``g4 = m_4 / m_2**2``.

Unidades, fijadas y sin ambiguedad: **todo entra por sesion**, nunca anualizado.
`sharpe_ratio(returns, annualization=1)` es exactamente eso. Anualizar no es inocuo: el
numerador escala y el denominador no, asi que mezclar un ``SR`` anualizado con una
``V[SR]`` por sesion produce un numero plausible y falso. `plan.md` §11.6 pre-registra
«Deflated Sharpe > 0 significativo»; el modulo lo lee como «significativo al nivel
declarado» (``DSR_CONFIDENCE_LEVEL = 0.95``) y lo **declara** en ``threshold_rule``,
porque «``dsr > 0``» lo cumple casi cualquier serie (el DSR es una probabilidad) y no
seria un criterio de *kill*: el literal se cita y se interpreta, no se reescribe.

El PBO se calcula con validacion cruzada combinatoriamente simetrica (CSCV): ``S`` bloques
disjuntos de igual tamano, ``S`` par y ``T mod S == 0`` (`DEFAULT_BLOCKS = 16`), las
``C(S, S/2)`` combinaciones, la variante de mayor ``SR`` en ``IS``, su rango ``OOS``
normalizado ``omega`` en ``(0, 1)`` y ``lambda = ln(omega / (1 - omega))``; el PBO es la
fraccion de combinaciones con ``lambda <= 0``. El presupuesto combinatorio es explicito
(`MAX_COMBINATIONS = 12_870 = C(16, 8)`): excederlo sin semilla es error tipado, y con
semilla explicita se sortea un subconjunto determinista con ``numpy.random.RandomState``
(el *stream* congelado entre versiones, la eleccion de #15) y el payload lo declara
(``method: "sampled"``). **Muestrear en silencio esta prohibido.**

Lo que este modulo **no** hace, legible por maquina, en :data:`OVERFITTING_DOES_NOT_DO`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

import numpy as np
import numpy.typing as npt

from cfdtrader.backtest.metrics import MAX_SEED, sharpe_ratio

__all__ = [
    "DEFAULT_BLOCKS",
    "DEFLATION_ESTIMATED",
    "DEFLATION_NONE",
    "DSR_CONFIDENCE_LEVEL",
    "EULER_MASCHERONI",
    "FOLLOW_UPS",
    "MAX_COMBINATIONS",
    "MAX_TRIALS",
    "METHOD_EXHAUSTIVE",
    "METHOD_SAMPLED",
    "NOISE_SEED",
    "OVERFITTING_DOES_NOT_DO",
    "PBO_MAX",
    "PER_SESSION",
    "SIGNAL_MEAN",
    "SIGNAL_SEED",
    "SYNTHETIC_OBSERVATIONS",
    "SYNTHETIC_SIGMA",
    "SYNTHETIC_VARIANTS",
    "THRESHOLD_RULE",
    "VERDICT_DETECTED",
    "VERDICT_NOT_DETECTED",
    "VERDICT_NOT_EVALUABLE",
    "VERDICT_NOT_SIGNIFICANT",
    "VERDICT_SIGNIFICANT",
    "BacktestOverfitting",
    "CombinationBudgetError",
    "DeflatedSharpe",
    "DegenerateMatrixError",
    "DegenerateSeriesError",
    "InsufficientObservationsError",
    "InsufficientTrialsError",
    "InsufficientVariantsError",
    "InvalidBlocksError",
    "InvalidConfidenceLevelError",
    "InvalidMatrixShapeError",
    "InvalidSamplingSeedError",
    "InvalidTrialsError",
    "InvalidVarianceError",
    "NonFiniteInputError",
    "NonPositiveDenominatorError",
    "OverfittingError",
    "SamplingExhaustedError",
    "deflated_sharpe_ratio",
    "noise_matrix",
    "probability_of_backtest_overfitting",
    "select_variant",
    "signal_matrix",
    "sr0_expected_max",
    "variant_sharpe_variance",
]

#: Numero de bloques por defecto del CSCV: ``C(16, 8) = 12_870`` combinaciones.
DEFAULT_BLOCKS: Final[int] = 16

#: Presupuesto combinatorio declarado: ``C(16, 8)``. Excederlo exige semilla explicita.
MAX_COMBINATIONS: Final[int] = 12_870

#: Umbral pre-registrado de `plan.md` §11.4 («PBO < 20 %»). No se ajusta desde el CLI.
PBO_MAX: Final[float] = 0.20

#: Umbral pre-registrado del DSR (`plan.md` §11.6). No se ajusta desde el CLI.
DSR_CONFIDENCE_LEVEL: Final[float] = 0.95

#: Constante de Euler-Mascheroni, la del contrato estadistico declarado.
EULER_MASCHERONI: Final[float] = 0.5772156649015329

#: Convencion de unidades: por sesion, nunca anualizada.
PER_SESSION: Final[str] = "per_session"

#: Veredictos declarados: **explicitos**, nunca una tabla de numeros sin conclusion.
VERDICT_SIGNIFICANT: Final[str] = "significant"
VERDICT_NOT_SIGNIFICANT: Final[str] = "not_significant"
VERDICT_DETECTED: Final[str] = "detected"
VERDICT_NOT_DETECTED: Final[str] = "not_detected"

#: Lo no estimable se declara, nunca se rellena con un numero (es el valor de #9).
VERDICT_NOT_EVALUABLE: Final[str] = "not_evaluable"

#: La deflacion se declara: ``estimated`` con varianza de variantes y ``none`` si es cero.
DEFLATION_ESTIMATED: Final[str] = "estimated"
DEFLATION_NONE: Final[str] = "none"

#: Metodos del PBO: exhaustivo o muestreado **declarado**.
METHOD_EXHAUSTIVE: Final[str] = "exhaustive"
METHOD_SAMPLED: Final[str] = "sampled"

#: `plan.md` §11.6 dice «Deflated Sharpe > 0 significativo»: se cita y se interpreta.
THRESHOLD_RULE: Final[str] = (
    "`plan.md` §11.6 (linea 735) pre-registra «Deflated Sharpe > 0 significativo»; se lee "
    "como «el estadistico deflactado es significativo al nivel declarado» "
    "(DSR >= DSR_CONFIDENCE_LEVEL = 0.95) y **no** como `dsr > 0`, que lo cumpliria casi "
    "cualquier serie porque el DSR es una probabilidad y no seria un criterio de *kill*: el "
    "literal no se reescribe, se cita y se interpreta"
)

#: Limite superior de ``n_trials``: por encima, ``1 - 1/(N*e)`` se redondea a 1 y la
#: correccion por normalidad deja de ser estimable. No se inventa un valor: es error tipado.
MAX_TRIALS: Final[int] = 2**52

#: Constantes de la validacion sintetica (A16, A17): fijas, con nombre y publicadas.
SYNTHETIC_OBSERVATIONS: Final[int] = 1_024
SYNTHETIC_VARIANTS: Final[int] = 20
SYNTHETIC_SIGMA: Final[float] = 0.01
SIGNAL_MEAN: Final[float] = 0.002
NOISE_SEED: Final[int] = 20_260_919
SIGNAL_SEED: Final[int] = 20_260_920

#: Multiplicador del presupuesto de sorteos: si no se alcanzan las combinaciones distintas
#: pedidas, es error tipado en vez de un bucle infinito.
_MAX_DRAW_MULTIPLIER: Final[int] = 64

#: Que **no** hace el modulo, legible por maquina (A30). Cada frontera con su issue.
OVERFITTING_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_es_la_suite_de_integridad",
        "issue": "#17",
        "reason": (
            "no es la suite de integridad (no-look-ahead de features, *golden dataset*, "
            "determinismo del gate): aqui el determinismo es una puerta del registro"
        ),
    },
    {
        "id": "no_mide_el_slippage",
        "issue": "#62",
        "reason": (
            "no mide el *slippage* real: sin medicion no hay Sharpe neto que deflactar, asi "
            "que el DSR y el PBO se calculan sobre el retorno declarado "
            "(`basis: declared_cost`, `is_validation: false`)"
        ),
    },
    {
        "id": "no_decide_r_ni_umbrales",
        "issue": "#60",
        "reason": (
            "no decide `R`, los umbrales ni el *sizing*: los umbrales son constantes "
            "pre-registradas que solo se citan, y `R` no se toca aqui"
        ),
    },
    {
        "id": "no_es_cpcv",
        "issue": "#67",
        "reason": (
            "no produce un `SplitPlan`, no purga ni embargo nada: el CSCV del PBO recombina "
            "bloques de la matriz de retornos de las variantes, no particiones de "
            "entrenamiento"
        ),
    },
    {
        "id": "no_toca_el_holdout",
        "issue": "#68",
        "reason": (
            "no reserva ni lee el periodo final intocable (`plan.md` §11.4 y §21 pregunta 10): "
            "aqui no hay constante de ultimos meses ni muestra reservada"
        ),
    },
    {
        "id": "no_entrena_modelos",
        "issue": "#28/#29",
        "reason": (
            "no entrena regresion logistica (#28) ni LightGBM (#29): recibe series de "
            "retornos ya producidas, tambien en la validacion sintetica"
        ),
    },
    {
        "id": "no_construye_features",
        "issue": "#19-#23",
        "reason": (
            "no construye ni versiona la matriz de features: el registro deja el campo de "
            "features declarado y guardara su version cuando exista"
        ),
    },
    {
        "id": "no_usa_el_store",
        "issue": "#2",
        "reason": (
            "no usa el `Store` ni `read_pit`: el almacen es para datos de mercado, no para "
            "experimentos; el registro escribe ficheros JSON/MD bajo `runs/<hash>/`"
        ),
    },
    {
        "id": "no_retencion_ops",
        "issue": "#44",
        "reason": (
            "no implementa retencion ni compactacion de `ops.*`: `runs/` crece con una "
            "carpeta por identidad y hoy nada la poda"
        ),
    },
    {
        "id": "no_publica_metricas_netas",
        "issue": "#15",
        "reason": (
            "no publica Sharpe/Sortino netos: `pnl_net_pct` es `null` con el supuesto de #64 "
            "y `calculate_metrics` los rechaza, asi que viaja `not_computable` con su motivo"
        ),
    },
    {
        "id": "no_valida_la_estrategia",
        "issue": "#18",
        "reason": (
            "no declara validada ninguna estrategia: publica una correccion por sobreajuste "
            "sobre datos declarados, con `is_validation: false`"
        ),
    },
)

#: Seguimientos enlazados (A30): cada frontera con su issue, nunca se borra en silencio.
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "integrity_suite",
        "issue": "#17",
        "reason": (
            "la suite de integridad (no-look-ahead, *golden dataset*, determinismo del gate) "
            "es la que blinda el arnes del que este modulo consume series"
        ),
    },
    {
        "id": "slippage_measurement",
        "issue": "#62",
        "reason": (
            "medir el *slippage* real es lo unico que convierte el retorno declarado en "
            "retorno neto y permite deflactar una serie validada"
        ),
    },
    {
        "id": "r_and_thresholds",
        "issue": "#60",
        "reason": (
            "el tamano de `R` y los umbrales de decision siguen sin decidir: hoy `R` no "
            "existe y el supuesto de #64 no se puede cobrar"
        ),
    },
    {
        "id": "cpcv_scheme",
        "issue": "#67",
        "reason": (
            "el CPCV como esquema alternativo de validacion (particiones, purga y embargo) no "
            "se implementa aqui; el CSCV del PBO no es un `SplitPlan`"
        ),
    },
    {
        "id": "final_holdout",
        "issue": "#68",
        "reason": (
            "el *holdout* final intocable se reserva y se gestiona en #68: aqui no se reserva "
            "ni se mira"
        ),
    },
    {
        "id": "phase2_logistic",
        "issue": "#28",
        "reason": (
            "el DSR y el PBO con variantes **reales** (regresion logistica) son Fase 2: aqui "
            "no se entrena ningun modelo"
        ),
    },
    {
        "id": "phase2_boosting",
        "issue": "#29",
        "reason": (
            "LightGBM y la decision documentada de umbrales son Fase 2, con el mismo registro "
            "de experimentos"
        ),
    },
    {
        "id": "features",
        "issue": "#19",
        "reason": (
            "la matriz de features (#19-#23) no existe todavia: el registro deja el campo "
            "declarado y guardara su version cuando exista"
        ),
    },
    {
        "id": "retention",
        "issue": "#44",
        "reason": (
            "la retencion y compactacion de `ops.*` (y la poda de `runs/`) es #44: hoy no hay "
            "politica declarada"
        ),
    },
    {
        "id": "documentation",
        "issue": "#65",
        "reason": (
            "documentar la divergencia `runs/<hash>/` frente a `ops.backtest_runs` de "
            "`tech_stack.md` §12 en `plan.md`/`tech_stack.md`"
        ),
    },
    {
        "id": "backlog_regularisation",
        "issue": "#65",
        "reason": (
            "regularizar el backlog: #67, #68 y #69 viven solo como issues y no estan en "
            "`_docs/tasks.md`"
        ),
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados (A19, A24): una clase distinta por caso, nunca una generica
# ─────────────────────────────────────────────────────────────────────────────
class OverfittingError(Exception):
    """Raiz de los errores de la correccion por sobreajuste."""


class InsufficientTrialsError(OverfittingError):
    """Menos de dos variantes: `V[SR]` no es estimable y `Zinv(1 - 1/N)` degenera."""


class InvalidTrialsError(OverfittingError):
    """Tantas variantes que la correccion por normalidad deja de ser estimable."""


class InvalidVarianceError(OverfittingError):
    """`V[SR]` no es un numero finito y no negativo: no se finge una deflacion."""


class InsufficientObservationsError(OverfittingError):
    """Menos de dos observaciones: no hay `sqrt(T - 1)` ni momentos."""


class InsufficientVariantsError(OverfittingError):
    """Menos de dos columnas: no hay nada que comparar ni rango que normalizar."""


class InvalidMatrixShapeError(OverfittingError):
    """La matriz de variantes no es rectangular, o no tiene forma `T x N`."""


class NonFiniteInputError(OverfittingError):
    """Un retorno o un parametro es `nan`/`inf`: el contrato exige entrada finita."""


class InvalidBlocksError(OverfittingError):
    """`S` no es par, no es al menos 2 o no divide a `T`."""


class CombinationBudgetError(OverfittingError):
    """`C(S, S/2)` excede `MAX_COMBINATIONS` y no se paso una semilla explicita."""


class InvalidSamplingSeedError(OverfittingError):
    """La semilla de muestreo no esta en `[0, MAX_SEED)`: el generador no la admite."""


class SamplingExhaustedError(OverfittingError):
    """No se alcanzaron las combinaciones distintas pedidas dentro del presupuesto."""


class InvalidConfidenceLevelError(OverfittingError):
    """El nivel de confianza no esta estrictamente entre 0 y 1."""


class DegenerateSeriesError(OverfittingError):
    """Serie de desviacion cero: los momentos de orden 3 y 4 no son estimables."""


class DegenerateMatrixError(OverfittingError):
    """Ninguna columna de la matriz tiene desviacion: el rango no existe."""


class NonPositiveDenominatorError(OverfittingError):
    """`1 - g3*SR + (g4 - 1)/4 * SR**2 <= 0`: la correccion por no normalidad no es valida."""


# ─────────────────────────────────────────────────────────────────────────────
# Validacion de entradas
# ─────────────────────────────────────────────────────────────────────────────
def _as_floats(values: Sequence[float], *, where: str) -> tuple[float, ...]:
    """Copia a `float` validando finitud: `nan`/`inf` son error tipado, no un resultado."""
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise NonFiniteInputError(f"{where}: hay un valor no finito (`nan`/`inf`) (A19)")
    return result


def _validate_confidence(confidence_level: float) -> None:
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise InvalidConfidenceLevelError(
            f"confidence_level debe estar estrictamente entre 0 y 1; llego {confidence_level!r}"
        )


def _validate_trials(n_trials: int) -> None:
    if n_trials < 2:
        raise InsufficientTrialsError(
            f"n_trials >= 2 es obligatorio; con N = {n_trials} no hay varianza de variantes "
            "estimable y `Zinv(1 - 1/N)` degenera (A3)"
        )
    if n_trials > MAX_TRIALS:
        raise InvalidTrialsError(
            f"n_trials <= {MAX_TRIALS} es el limite publicable: por encima, "
            "`1 - 1/(N*e)` se redondea a 1 y la correccion no es estimable (A19)"
        )


def _validate_variance(sr_variance: float) -> None:
    if not math.isfinite(sr_variance) or sr_variance < 0.0:
        raise InvalidVarianceError(
            f"sr_variance debe ser finita y >= 0; llego {sr_variance!r}: sin varianza no se "
            "finge una deflacion, se declara (A19)"
        )


def _validate_seed(sampling_seed: int) -> None:
    if not 0 <= sampling_seed < MAX_SEED:
        raise InvalidSamplingSeedError(
            f"sampling_seed debe estar en [0, {MAX_SEED}); {sampling_seed} no lo esta: el "
            "generador declarado no admite ese valor"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sharpes por sesion y varianza de las variantes (A9, #15)
# ─────────────────────────────────────────────────────────────────────────────
def _as_matrix(returns_matrix: Sequence[Sequence[float]], *, where: str) -> npt.NDArray[np.float64]:
    """Matriz `T x N` (filas = observaciones, columnas = variantes) validada y finita."""
    rows = [tuple(row) for row in returns_matrix]
    if not rows:
        raise InsufficientObservationsError(f"{where}: la matriz esta vacia (A19)")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise InvalidMatrixShapeError(
            f"{where}: la matriz no es rectangular; se ven anchos {sorted(widths)} (A19)"
        )
    n_observations = len(rows)
    n_variants = len(rows[0])
    if n_observations < 2:
        raise InsufficientObservationsError(
            f"{where}: se necesitan al menos 2 observaciones; hay {n_observations} (A19)"
        )
    if n_variants < 2:
        raise InsufficientVariantsError(
            f"{where}: se necesitan al menos 2 variantes; hay {n_variants} (A19)"
        )
    matrix = np.asarray(rows, dtype=np.float64)
    if not bool(np.isfinite(matrix).all()):
        raise NonFiniteInputError(f"{where}: hay un valor no finito (`nan`/`inf`) (A19)")
    spread = matrix.max(axis=0) - matrix.min(axis=0)
    if not bool((np.asarray(spread) != 0.0).any()):
        raise DegenerateMatrixError(
            f"{where}: ninguna columna tiene desviacion, asi que no hay rango OOS que "
            "normalizar ni variante que elegir; no se inventa un numero (A19)"
        )
    return matrix


def variant_sharpe_variance(sharpes: Sequence[float]) -> float:
    """Varianza **muestral** (``ddof = 1``) de los Sharpe de las variantes (A3, A9).

    Es la `V[SR]` de la formula del DSR y sale del registro de experimentos, nunca de un
    parametro suelto. Una serie constante devuelve ``0.0`` **exacto** (el test ``max - min``
    de #15): un residuo de coma flotante convertiria una deflacion nula en una deflacion
    inventada.
    """
    values = _as_floats(sharpes, where="sharpes de las variantes")
    if len(values) < 2:
        raise InsufficientTrialsError(
            f"se necesitan al menos 2 variantes para estimar V[SR]; hay {len(values)} (A3)"
        )
    if max(values) == min(values):
        return 0.0
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _population_moments(values: tuple[float, ...]) -> tuple[float, float]:
    """Asimetria y curtosis **poblacionales** (no excedente) de una serie (A2).

    Momentos poblacionales literalmente: ``m_k = (1/T) * suma((x - media)**k)``,
    ``g3 = m_3 / m_2**1.5`` y ``g4 = m_4 / m_2**2``. Una serie constante no tiene momentos
    de orden 3 y 4: es error tipado, no un cero inventado.
    """
    if max(values) == min(values):
        raise DegenerateSeriesError(
            "la serie es constante: `m_2 = 0` y la asimetria y la curtosis no son "
            "estimables; no se rellenan con 0 (A19)"
        )
    size = len(values)
    mean = math.fsum(values) / size
    centred = [value - mean for value in values]
    second = math.fsum(value * value for value in centred) / size
    if second <= 0.0:
        raise DegenerateSeriesError("la serie tiene `m_2 <= 0`: los momentos no son estimables")
    third = math.fsum(value**3 for value in centred) / size
    fourth = math.fsum(value**4 for value in centred) / size
    return third / second**1.5, fourth / second**2


def select_variant(returns_matrix: Sequence[Sequence[float]]) -> int:
    """Indice de la variante de mayor Sharpe por sesion sobre la muestra completa.

    Es la regla de seleccion declarada: el Sharpe de cada columna sale de
    `sharpe_ratio(..., annualization=1)` y **los empates se resuelven al indice mas bajo**
    (``argmax`` determinista), nunca por orden de iteracion de un contenedor sin orden.
    """
    matrix = _as_matrix(returns_matrix, where="matriz de variantes")
    best = 0
    best_ratio = sharpe_ratio(matrix[:, 0].tolist(), annualization=1)
    for index in range(1, int(matrix.shape[1])):
        ratio = sharpe_ratio(matrix[:, index].tolist(), annualization=1)
        if ratio > best_ratio:
            best, best_ratio = index, ratio
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Deflated Sharpe Ratio (A2-A5, A8)
# ─────────────────────────────────────────────────────────────────────────────
def sr0_expected_max(*, n_trials: int, sr_variance: float) -> float:
    """`SR0`: el Sharpe esperado del **mejor** de `N` intentos sin habilidad real.

    Formula literal del contrato: ``sqrt(V[SR]) * [(1 - gamma) * Zinv(1 - 1/N) + gamma *
    Zinv(1 - 1/(N*e))]``, con la normal estandar exacta de ``statistics.NormalDist``. Si
    ``V[SR] = 0`` devuelve ``0.0`` y el payload lo declara como ``deflation: "none"``: sin
    varianza no se finge deflacion, se publica (A3).
    """
    _validate_variance(sr_variance)
    _validate_trials(n_trials)
    if sr_variance == 0.0:
        return 0.0
    normal = NormalDist()
    first = 1.0 - 1.0 / n_trials
    second = 1.0 - 1.0 / (n_trials * math.e)
    return math.sqrt(sr_variance) * (
        (1.0 - EULER_MASCHERONI) * normal.inv_cdf(first) + EULER_MASCHERONI * normal.inv_cdf(second)
    )


@dataclass(frozen=True, slots=True)
class DeflatedSharpe:
    """El resultado del DSR con su veredicto **explicito** (A4) y su convencion de unidades."""

    dsr: float
    sr_observed: float
    sr0_expected_max: float
    n_trials: int
    sr_variance: float
    n_observations: int
    skewness: float
    kurtosis: float
    confidence_level: float
    deflation: str
    verdict: str
    non_normality_denominator: float
    units: str = PER_SESSION
    threshold_rule: str = THRESHOLD_RULE
    note: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Payload JSON puro: sin `nan`/`inf` y con el veredicto al lado del numero."""
        return {
            "dsr": self.dsr,
            "sr_observed": self.sr_observed,
            "sr0_expected_max": self.sr0_expected_max,
            "n_trials": self.n_trials,
            "sr_variance": self.sr_variance,
            "n_observations": self.n_observations,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "confidence_level": self.confidence_level,
            "deflation": self.deflation,
            "verdict": self.verdict,
            "non_normality_denominator": self.non_normality_denominator,
            "units": self.units,
            "threshold_rule": self.threshold_rule,
            "note": self.note,
        }


def deflated_sharpe_ratio(
    returns: Sequence[float],
    *,
    n_trials: int,
    sr_variance: float,
    confidence_level: float = DSR_CONFIDENCE_LEVEL,
) -> DeflatedSharpe:
    """Deflated Sharpe Ratio de una serie, con el numero **real** de variantes probadas.

    `returns` son los retornos de la variante **seleccionada**, en unidades por sesion;
    `n_trials` y `sr_variance` se derivan del registro de experimentos (nunca de un campo
    del informe ni de una opcion del CLI). El veredicto es explicito: ``significant`` si
    ``dsr >= confidence_level``.
    """
    values = _as_floats(returns, where="returns")
    if len(values) < 2:
        raise InsufficientObservationsError(
            f"se necesitan al menos 2 retornos; hay {len(values)}: `sqrt(T - 1)` no existe (A19)"
        )
    _validate_confidence(confidence_level)
    sr0 = sr0_expected_max(n_trials=n_trials, sr_variance=sr_variance)
    observed = sharpe_ratio(values, annualization=1)
    skewness, kurtosis = _population_moments(values)
    denominator = 1.0 - skewness * observed + (kurtosis - 1.0) / 4.0 * observed * observed
    if denominator <= 0.0:
        raise NonPositiveDenominatorError(
            "el denominador de la correccion por no normalidad es <= 0 "
            f"({denominator!r}): el DSR no es estimable con esta serie (A19)"
        )
    statistic = (observed - sr0) * math.sqrt(len(values) - 1) / math.sqrt(denominator)
    dsr = NormalDist().cdf(statistic)
    deflation = DEFLATION_NONE if sr_variance == 0.0 else DEFLATION_ESTIMATED
    note = (
        "V[SR] = 0: todas las variantes comparten Sharpe, asi que SR0 = 0 y no se finge "
        "deflacion; el numero se publica igual (A3)"
        if sr_variance == 0.0
        else None
    )
    return DeflatedSharpe(
        dsr=dsr,
        sr_observed=observed,
        sr0_expected_max=sr0,
        n_trials=n_trials,
        sr_variance=sr_variance,
        n_observations=len(values),
        skewness=skewness,
        kurtosis=kurtosis,
        confidence_level=confidence_level,
        deflation=deflation,
        verdict=VERDICT_SIGNIFICANT if dsr >= confidence_level else VERDICT_NOT_SIGNIFICANT,
        non_normality_denominator=denominator,
        note=note,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PBO por CSCV (A6, A7)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _BlockStats:
    """Momentos precalculados por bloque del CSCV: suma, suma de cuadrados, maximo y minimo."""

    totals: npt.NDArray[np.float64]
    squares: npt.NDArray[np.float64]
    maxima: npt.NDArray[np.float64]
    minima: npt.NDArray[np.float64]


def _subset_sharpes(
    stats: _BlockStats, index: npt.NDArray[np.intp], *, observations: int
) -> npt.NDArray[np.float64]:
    """Sharpe por sesion de cada columna sobre la union de los bloques `index`.

    Forma de **una pasada** (``suma(x)``, ``suma(x*x)``) sobre los momentos ya precalculados
    de cada bloque: es la unica manera de recorrer las 12.870 combinaciones del CSCV en
    tiempo util, y el estimador puntual que se **publica** (`sr_observed`) sigue saliendo del
    `sharpe_ratio` de #15. La regla de la varianza cero es la de #15: ``max - min == 0`` en la
    union de bloques, que aqui sale del maximo de los maximos y el minimo de los minimos, y la
    varianza se acota en ``0`` para que una cancelacion en coma flotante no publique un
    ``nan``.
    """
    totals = stats.totals[index].sum(axis=0)
    squares = stats.squares[index].sum(axis=0)
    spread = stats.maxima[index].max(axis=0) - stats.minima[index].min(axis=0)
    mean = totals / observations
    variance = np.maximum((squares - observations * mean * mean) / (observations - 1), 0.0)
    deviation = np.sqrt(variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = mean / deviation
    return np.asarray(np.where((spread == 0.0) | (np.asarray(deviation) == 0.0), 0.0, ratios))


def _block_stats(matrix: npt.NDArray[np.float64], *, blocks: int) -> _BlockStats:
    """Momentos por bloque de la matriz `T x N`: suma, suma de cuadrados, maximo y minimo."""
    rows = int(matrix.shape[0]) // blocks
    shaped = matrix.reshape(blocks, rows, int(matrix.shape[1]))
    return _BlockStats(
        totals=np.asarray(shaped.sum(axis=1)),
        squares=np.asarray((shaped * shaped).sum(axis=1)),
        maxima=np.asarray(shaped.max(axis=1)),
        minima=np.asarray(shaped.min(axis=1)),
    )


def _select_column(sharpes: npt.NDArray[np.float64]) -> int:
    """Indice de la columna de mayor Sharpe, con empates al indice mas bajo.

    Es la misma regla que :func:`select_variant` (`numpy.argmax` devuelve el primer maximo),
    sin volver a validar la matriz en cada una de las combinaciones del CSCV.
    """
    best = 0
    for index in range(1, int(sharpes.shape[0])):
        if float(sharpes[index]) > float(sharpes[best]):
            best = index
    return best


def _combinations(
    *, blocks: int, sampling_seed: int | None
) -> tuple[tuple[tuple[int, ...], ...], str]:
    """Combinaciones a evaluar y el metodo declarado: exhaustivo o muestreado (A7).

    Con ``C(S, S/2) <= MAX_COMBINATIONS`` se recorren **todas** (orden creciente, el de
    `itertools.combinations`). Por encima hace falta una semilla **explicita**: se sortean
    combinaciones distintas con ``numpy.random.RandomState`` y el payload declara
    ``method: "sampled"``, ``n_combinations_drawn`` y la semilla. Muestrear en silencio
    esta prohibido: sin semilla, el exceso de presupuesto es error tipado.
    """
    total = math.comb(blocks, blocks // 2)
    if total <= MAX_COMBINATIONS:
        return tuple(itertools.combinations(range(blocks), blocks // 2)), METHOD_EXHAUSTIVE
    if sampling_seed is None:
        raise CombinationBudgetError(
            f"C({blocks}, {blocks // 2}) = {total} excede MAX_COMBINATIONS = "
            f"{MAX_COMBINATIONS}: o se reduce `blocks` o se pasa `sampling_seed` explicito "
            "(el muestreo en silencio esta prohibido, A7)"
        )
    _validate_seed(sampling_seed)
    rng = np.random.RandomState(sampling_seed)
    seen: set[tuple[int, ...]] = set()
    drawn: list[tuple[int, ...]] = []
    attempts = 0
    limit = MAX_COMBINATIONS * _MAX_DRAW_MULTIPLIER
    while len(drawn) < MAX_COMBINATIONS and attempts < limit:
        attempts += 1
        pick = tuple(
            sorted(int(value) for value in rng.choice(blocks, size=blocks // 2, replace=False))
        )
        if pick in seen:
            continue
        seen.add(pick)
        drawn.append(pick)
    if len(drawn) < MAX_COMBINATIONS:
        raise SamplingExhaustedError(
            f"no se alcanzaron {MAX_COMBINATIONS} combinaciones distintas en {attempts} sorteos "
            f"con sampling_seed = {sampling_seed}: el sorteo no cubre el presupuesto declarado"
        )
    return tuple(drawn), METHOD_SAMPLED


def _average_ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Rango 1..N de cada columna, con **rangos medios** en los empates (regla declarada).

    El rango de los empates se promedia, asi que la regla es determinista y no depende del
    orden de las columnas; sin empates el reparto es el mismo que asignaria un `argsort`
    doble, sin recorrer ninguna lista en Python.
    """
    size = int(values.shape[0])
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    if len(np.unique(sorted_values)) == size:
        ranks = np.argsort(order, kind="stable").astype(np.float64) + 1.0
        return ranks
    ranks = np.empty(size, dtype=np.float64)
    start = 0
    while start < size:
        stop = start
        while stop + 1 < size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


@dataclass(frozen=True, slots=True)
class BacktestOverfitting:
    """El resultado del PBO por CSCV, con su metodo declarado y su veredicto (A6, A7)."""

    pbo: float
    blocks: int
    n_observations: int
    n_variants: int
    n_combinations: int
    n_combinations_drawn: int
    logit_median: float
    omega_median: float
    best_oos_sharpe_median: float
    method: str
    verdict: str
    sampling_seed: int | None
    pbo_max: float = PBO_MAX
    units: str = PER_SESSION
    omega_rule: str = (
        "omega = rango OOS de la variante elegida en IS entre las N, normalizado como "
        "rango/(N + 1) para que caiga en (0, 1); lambda = ln(omega / (1 - omega))"
    )
    tie_rule: str = (
        "empates: la mejor en IS es la de menor indice de columna; los empates de rango OOS "
        "se resuelven con rangos medios"
    )
    note: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Payload JSON puro, con `method` y el veredicto explicito."""
        return {
            "pbo": self.pbo,
            "blocks": self.blocks,
            "n_observations": self.n_observations,
            "n_variants": self.n_variants,
            "n_combinations": self.n_combinations,
            "n_combinations_drawn": self.n_combinations_drawn,
            "logit_median": self.logit_median,
            "omega_median": self.omega_median,
            "best_oos_sharpe_median": self.best_oos_sharpe_median,
            "method": self.method,
            "sampling_seed": self.sampling_seed,
            "verdict": self.verdict,
            "pbo_max": self.pbo_max,
            "units": self.units,
            "omega_rule": self.omega_rule,
            "tie_rule": self.tie_rule,
            "note": self.note,
        }


def probability_of_backtest_overfitting(
    returns_matrix: Sequence[Sequence[float]],
    *,
    blocks: int = DEFAULT_BLOCKS,
    sampling_seed: int | None = None,
) -> BacktestOverfitting:
    """PBO por CSCV sobre una matriz `T x N` de retornos de las variantes (A6, A7).

    `blocks` (``S``) debe ser par, al menos 2 y dividir a ``T``; cada combinacion parte los
    ``S`` bloques en ``S/2`` para *in-sample* y ``S/2`` para *out-of-sample*, elige la
    variante de mayor Sharpe por sesion en ``IS`` y mide su rango ``OOS``. ``sampling_seed``
    sin valor por defecto util: ``None`` significa «no se muestrea», y exceder
    ``MAX_COMBINATIONS`` con ``None`` es error tipado.
    """
    matrix = _as_matrix(returns_matrix, where="matriz de variantes")
    n_observations = int(matrix.shape[0])
    n_variants = int(matrix.shape[1])
    if blocks < 2 or blocks % 2 != 0:
        raise InvalidBlocksError(f"blocks debe ser par y al menos 2; llego {blocks} (A6, A19)")
    if n_observations % blocks != 0:
        raise InvalidBlocksError(
            f"blocks debe dividir al numero de observaciones; {blocks} no divide a "
            f"{n_observations} (A6, A19)"
        )
    combos, method = _combinations(blocks=blocks, sampling_seed=sampling_seed)
    rows_per_block = n_observations // blocks
    stats = _block_stats(matrix, blocks=blocks)
    every_block = tuple(range(blocks))
    omega = np.empty(len(combos), dtype=np.float64)
    best_oos = np.empty(len(combos), dtype=np.float64)
    for position, combo in enumerate(combos):
        in_sample = np.asarray(combo, dtype=np.intp)
        out_sample = np.asarray(
            tuple(block for block in every_block if block not in combo), dtype=np.intp
        )
        chosen = _select_column(
            _subset_sharpes(stats, in_sample, observations=len(combo) * rows_per_block)
        )
        out_sharpes = _subset_sharpes(
            stats, out_sample, observations=(blocks - len(combo)) * rows_per_block
        )
        omega[position] = float(_average_ranks(out_sharpes)[chosen]) / (n_variants + 1)
        best_oos[position] = float(out_sharpes[chosen])
    with np.errstate(divide="ignore", invalid="ignore"):
        logits = np.log(omega / (1.0 - omega))
    pbo = float(np.count_nonzero(logits <= 0.0)) / len(combos)
    return BacktestOverfitting(
        pbo=pbo,
        blocks=blocks,
        n_observations=n_observations,
        n_variants=n_variants,
        n_combinations=math.comb(blocks, blocks // 2),
        n_combinations_drawn=len(combos),
        logit_median=float(np.median(logits)),
        omega_median=float(np.median(omega)),
        best_oos_sharpe_median=float(np.median(best_oos)),
        method=method,
        verdict=VERDICT_NOT_DETECTED if pbo < PBO_MAX else VERDICT_DETECTED,
        sampling_seed=sampling_seed if method == METHOD_SAMPLED else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validacion sintetica (A16, A17): ruido puro y senal plantada, con semilla fija
# ─────────────────────────────────────────────────────────────────────────────
def noise_matrix(
    *,
    seed: int = NOISE_SEED,
    n_variants: int = SYNTHETIC_VARIANTS,
    n_observations: int = SYNTHETIC_OBSERVATIONS,
) -> tuple[tuple[float, ...], ...]:
    """Ruido puro: `N` variantes iid sin media, con semilla fija (A16).

    El DSR de la mejor variante debe salir ``not_significant`` y el PBO alto (≈ 0.5). Si el
    ruido «pasa», el modulo miente. El generador es ``numpy.random.RandomState`` (el
    *stream* congelado entre versiones, la eleccion de #15) y la semilla es **explicita**:
    misma semilla ⇒ mismos numeros, byte a byte, entre procesos. Sin
    ``numpy.random.Generator``.
    """
    if n_variants < 2:
        raise InsufficientVariantsError(f"se necesitan al menos 2 variantes; hay {n_variants}")
    rng = np.random.RandomState(seed)
    draws = rng.normal(loc=0.0, scale=SYNTHETIC_SIGMA, size=(n_observations, n_variants))
    return tuple(tuple(float(value) for value in row) for row in draws)


def signal_matrix(
    *,
    seed: int = SIGNAL_SEED,
    effect: float = SIGNAL_MEAN,
    n_variants: int = SYNTHETIC_VARIANTS,
    n_observations: int = SYNTHETIC_OBSERVATIONS,
) -> tuple[tuple[float, ...], ...]:
    """Senal plantada: la **primera** columna tiene media real y el resto es ruido (A17).

    La semilla y el tamano del efecto son constantes con nombre: el caso es determinista,
    no una corrida de esperanza, y su umbral no se ajusta despues de ver el resultado.
    """
    matrix = noise_matrix(seed=seed, n_variants=n_variants, n_observations=n_observations)
    planted = tuple(
        tuple(value + effect if column == 0 else value for column, value in enumerate(row))
        for row in matrix
    )
    return planted
