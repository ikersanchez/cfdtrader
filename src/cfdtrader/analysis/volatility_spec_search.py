"""Busqueda pre-registrada de especificaciones del forecast de volatilidad (tarea #58).

**Que entrega.** La busqueda que se saco de #7 para no caer en *data snooping*: ocho
especificaciones **declaradas en el codigo antes de medir** (HAR con cuarticidad, HAR con
saltos, HAR con la *term structure* del VIX, ridge y lasso sobre el HAR, regresion por
cuantiles, EGARCH y apARCH con `arch`), medidas en el **mismo** *walk-forward* que #7
--importado, no reimplementado-- y corregidas por comparaciones multiples con el DSR y el PBO
de #16, con el numero **real** de variantes probadas derivado de un registro de ensayos.

**La metrica y los folds no se tocan.** `MIN_TRAIN`, `REFIT_EVERY`, `RELATIVE_TOLERANCE`,
`TARGETS`, `TARGET_COLUMNS`, `garch_fold_bounds`, `select_candidate` y la evaluacion
(`evaluate_forecast`) se **importan** de :mod:`cfdtrader.analysis.volatility_forecast`; el unico
cambio en ese modulo es el alias publico de una funcion. Las cuatro filas de #7 de la tabla se
recalculan llamando a su :func:`walk_forward`: si la metrica o los folds se hubieran movido,
esas filas dejarian de coincidir (y eso es lo que comprueba un test).

**Las unidades del DSR se declaran, porque no son un P&L.** La serie de cada variante es el
**diferencial de perdida QLIKE por sesion frente a `rw`** (``d_t = L_rw,t - L_i,t``, positivo =
mejor que la persistencia), sobre el objetivo **primario** de #7 (Parkinson). El "Sharpe" de
diferenciales es un estadistico de ventaja estandarizada, no un retorno: por eso el payload lo
declara literalmente, el `basis` de los ensayos registrados lo dice, y `rw` entra como **linea
de referencia y no como trial** (su diferencial consigo misma es identicamente cero y no tiene
Sharpe que deflactar).

**Todo lo probado se registra, y en su propia raiz.** Cada variante evaluada se anota con
:func:`record_experiment` (#16) en ``runs_volatility/`` por defecto: `model_comparison` enumera
**todas** las entradas de ``runs/``, asi que registrar ahi los ensayos de volatilidad romperia
la cadena de modelos (es el acoplamiento que #158 dejo documentado).

**Lo que NO hace** (fronteras declaradas)

- **No** cambia produccion: `DECLARED_CANDIDATE`, `features/volatility.py`, `features/regime.py`
  y `delivery.run_daily` quedan intactos, y el informe de #7 no se sobreescribe.
- **No** adopta la especificacion ganadora: eso lo decide §19.16 (#63). Aqui se mide.
- **No** evalua HAR con RV intradia: depende de #57, bloqueado; se declara con su motivo.
- **No** toca la metrica, los folds ni el corte de muestra de #7: eso es de #7.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, cast

import duckdb
import numpy as np
import polars as pl
from arch import arch_model
from loguru import logger
from sklearn.linear_model import Lasso, QuantileRegressor, Ridge

from cfdtrader.analysis.baseline_report import (
    _as_utc,  # pyright: ignore[reportPrivateUsage]
    _json_text,  # pyright: ignore[reportPrivateUsage]
    _report_digest,  # pyright: ignore[reportPrivateUsage]
)
from cfdtrader.analysis.experiment_log import (
    ExperimentConfig,
    ExperimentLogError,
    ExperimentRecord,
    ExperimentResult,
    Registry,
    load_registry,
    record_experiment,
    require_trials_match_registry,
)
from cfdtrader.analysis.volatility_forecast import (
    DECLARED_CANDIDATE,
    MIN_TRAIN,
    REFIT_EVERY,
    RELATIVE_TOLERANCE,
    SERIES_ID,
    TARGET_COLUMNS,
    TARGETS,
    Candidate,
    CandidateResult,
    Metrics,
    Status,
    Target,
    build_sample,
    evaluate_forecast,
    relative_qlike_gap,
    walk_forward,
)
from cfdtrader.backtest.metrics import sharpe_ratio
from cfdtrader.backtest.overfitting import (
    DSR_CONFIDENCE_LEVEL,
    PBO_MAX,
    VERDICT_NOT_DETECTED,
    VERDICT_NOT_EVALUABLE,
    VERDICT_SIGNIFICANT,
    BacktestOverfitting,
    DeflatedSharpe,
    OverfittingError,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store, UnknownDatasetError
from cfdtrader.features.volatility import (
    HAR_MIN_TRAIN,
    HarCoefficients,
    fit_log_har,
    garch_fold_bounds,
)

__all__ = [
    "BASIS_LOSS_DIFFERENTIAL",
    "BLOCKS_MAX",
    "CLI_NAME",
    "CORRECTION_UNITS_NOTE",
    "DEFAULT_TRIALS_ROOT",
    "EVALUABLE_SPECS",
    "NOT_A_PNL_NOTE",
    "REFERENCE",
    "REPORT_DOES_NOT_DO",
    "REPORT_HASH_FORMAT",
    "REPORT_PREFIX",
    "SPECS",
    "VERDICT_IMPROVES",
    "VERDICT_INCONCLUSIVE",
    "VERDICT_KEEPS_7",
    "VIX3M_SERIES_ID",
    "VIX9D_SERIES_ID",
    "Spec",
    "SpecSearchError",
    "SpecSearchReport",
    "analyse",
    "blocks_for",
    "build_spec_frame",
    "load_term_structure",
    "loss_units",
    "main",
    "render_markdown",
    "spec_forecasts",
    "term_structure_regressors",
]

#: Nombre del CLI, para los mensajes de ``stderr``.
CLI_NAME: Final[str] = "cfdtrader.analysis.volatility_spec_search"

#: Prefijo del informe: ``volatility_spec_search_<AAAA-MM-DD>.{json,md}`` (no pisa el de #7).
REPORT_PREFIX: Final[str] = "volatility_spec_search"

#: Formato estable del ``report_sha256``: identico al de #24/#26/#158 (prefijo dentro del valor).
REPORT_HASH_FORMAT: Final[str] = (
    "`sha256:<64 hex>` del sha256(UTF-8) de cfdtrader.backtest.engine.canonical_text(payload), "
    "**sin** la clave report_sha256 (un informe no se hashea a si mismo). El payload es JSON puro "
    "y no lleva ninguna ruta absoluta: el hash no depende de `--reports-dir` ni de `--trials-root`"
)

#: Raiz por defecto del registro de ensayos. **Nunca** `runs/`: `model_comparison` enumera todas
#: las entradas de ahi y una variante que no sabe reconstruir deja su matriz incompleta (es el
#: acoplamiento que #158 documento en `REGISTRY_COUPLING_NOTE`).
DEFAULT_TRIALS_ROOT: Final[str] = "runs_volatility"

#: Las dos series de la *term structure* del VIX (A2).
VIX9D_SERIES_ID: Final[str] = "^VIX9D"

VIX3M_SERIES_ID: Final[str] = "^VIX3M"

#: Linea de referencia del diferencial: la persistencia de #7. **No** es un trial (A6).
REFERENCE: Final[str] = Candidate.RW.value

#: `basis` que declaran los ensayos registrados: **no** es el coste declarado de una estrategia,
#: es un diferencial de perdida. Escribir `declared_cost` aqui seria mentir sobre las unidades.
BASIS_LOSS_DIFFERENTIAL: Final[str] = "loss_differential_vs_rw"

#: Declaracion literal de las unidades de la correccion, en el payload (A6).
CORRECTION_UNITS_NOTE: Final[str] = (
    "La serie de cada variante **no** es un P&L: es el diferencial de perdida QLIKE por sesion "
    "frente a `rw` (`d_t = L_rw,t - L_i,t`, positivo = mejor que la persistencia), con el QLIKE "
    "de #7 (`L = ln(v_hat) + v / v_hat`) sobre el objetivo **primario** (Parkinson). El `Sharpe` "
    "que consume el DSR es, por tanto, una **ventaja estandarizada por sesion**, no un retorno: "
    "`units = per_session` y `annualization = 1`, igual que en #16. `rw` es la linea de "
    "referencia, no un trial: su diferencial consigo misma es identicamente cero."
)

#: Lo que el informe **no** hace, con su seguimiento (A9).
NOT_A_PNL_NOTE: Final[str] = (
    "La correccion se aplica sobre **diferenciales de perdida**, no sobre el P&L de una "
    "estrategia: un `pbo` bajo aqui significa «la mejor especificacion en muestra no conserva su "
    "ventaja fuera de muestra», no «la estrategia es rentable». Acoplar esto a la cadena de "
    "modelos y a sus unidades de coste es #26"
)

REPORT_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "what": "adoptar la especificacion ganadora",
        "where": "#63 (§19.16: la declaracion del candidato de volatilidad)",
    },
    {
        "what": "evaluar HAR con RV intradia",
        "where": "#57 (bloqueado por la decision de compra de datos finos)",
    },
    {
        "what": "cambiar la metrica, los folds o el corte de muestra de #7",
        "where": "#7 (no se mueve desde aqui)",
    },
    {
        "what": "traducir el resultado a unidades de P&L o al Sharpe de una estrategia",
        "where": "#26",
    },
)

#: Veredictos del bloque de correccion + regla (A8).
VERDICT_IMPROVES: Final[str] = "improves"
VERDICT_KEEPS_7: Final[str] = "keeps_declared_candidate"
VERDICT_INCONCLUSIVE: Final[str] = "inconclusive"

#: Bloques (`S`) que se prueban, de mayor a menor, para el CSCV de #16 (A7). La regla es
#: **declarada** y se calcula de `T`: el mayor divisor par de `T` que no pasa de `BLOCKS_MAX`.
BLOCKS_MAX: Final[int] = 16

# ─────────────────────────────────────────────────────────────────────────────
# Hiperparametros fijos (A1): ninguno se ajusta contra la muestra
# ─────────────────────────────────────────────────────────────────────────────
#: Penalizacion de la ridge sobre los retardos del HAR, en log-varianza.
RIDGE_ALPHA: Final[float] = 1.0

#: Penalizacion del lasso sobre los retardos del HAR, en log-varianza.
LASSO_ALPHA: Final[float] = 0.001

#: Cuantil de la regresion por cuantiles: la **mediana**, declarada (no un barrido).
QUANTILE: Final[float] = 0.5

#: Termino de regularizacion L1 de la regresion por cuantiles: 0 (sin penalizacion).
QUANTILE_ALPHA: Final[float] = 0.0

#: Ordenes de las dos familias `arch`: p = 1, o = 1 (asimetria), q = 1.
ARCH_P: Final[int] = 1
ARCH_O: Final[int] = 1
ARCH_Q: Final[int] = 1

#: Media y distribucion de las dos familias `arch`, iguales que las de #7.
ARCH_MEAN: Final[str] = "Zero"
ARCH_DIST: Final[str] = "normal"

#: Semilla **declarada** de la busqueda. Ni `arch` ni los estimadores lineales la usan; se
#: registra para que la configuracion del ensayo declare una semilla en vez de dejarla vacia.
SPEC_SEED: Final[int] = 20_261_011

#: Momento absoluto de una normal estandar: ``E|z| = sqrt(2/pi)`` (recursion de EGARCH).
NORMAL_ABS_MOMENT: Final[float] = math.sqrt(2.0 / math.pi)

#: Columnas del frame que necesita cada especificacion (A4).
_SPEC_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "harq": ("har_lag1", "har_lag4", "har_lag17", "har_quarticity"),
    "har_jump": ("har_lag1", "har_lag4", "har_lag17", "har_jump"),
    "har_vix_ts": (
        "har_lag1",
        "har_lag4",
        "har_lag17",
        "vix9d_level",
        "vix3m_level",
        "vix_ts_slope",
    ),
    "ridge_har": ("har_lag1", "har_lag4", "har_lag17"),
    "lasso_har": ("har_lag1", "har_lag4", "har_lag17"),
    "quantile_har": ("har_lag1", "har_lag4", "har_lag17"),
}


@dataclass(frozen=True, slots=True)
class Spec:
    """Una especificacion **pre-registrada** del catalogo (A1).

    ``estimator`` selecciona la familia de ajuste; ``regressors`` las columnas del diseño (vacio
    para las familias que trabajan sobre la serie de retornos); ``hyperparameters`` los valores
    **fijos** que se registran con el ensayo. Nada de esto se elige mirando el resultado.
    """

    name: str
    estimator: str
    formula: str
    regressors: tuple[str, ...]
    hyperparameters: Mapping[str, object]
    evaluable: bool = True
    reason: str | None = None
    follow_up: str | None = None

    def to_payload(self) -> dict[str, object]:
        """La declaracion del catalogo como objeto JSON puro."""
        return {
            "name": self.name,
            "estimator": self.estimator,
            "formula": self.formula,
            "regressors": list(self.regressors),
            "hyperparameters": dict(self.hyperparameters),
            "evaluable": self.evaluable,
            "status": Status.OK.value if self.evaluable else Status.UNAVAILABLE.value,
            "reason": self.reason,
            "follow_up": self.follow_up,
            "source": "#58",
        }


#: Catalogo **pre-registrado**, en el orden en que se desempata (A1). Escrito en el codigo
#: antes de medir y sin ninguna busqueda de hiperparametros: cada parametro es una constante.
SPECS: Final[tuple[Spec, ...]] = (
    Spec(
        name="harq",
        estimator="log_lstsq",
        formula=(
            "ln(rv_t) ~ 1 + har_lag1 + har_lag4 + har_lag17 + rv_quarticity, con "
            "rv_quarticity = ln((4/3) * rv_{t-1}^2): la cuarticidad del proxy de Parkinson "
            "**en la misma escala logaritmica** que el objetivo"
        ),
        regressors=_SPEC_COLUMNS["harq"],
        hyperparameters={"estimator": "log_lstsq", "target": "ln(parkinson_rv)"},
    ),
    Spec(
        name="har_jump",
        estimator="log_lstsq",
        formula=(
            "ln(rv_t) ~ 1 + har_lag1 + har_lag4 + har_lag17 + har_jump, con "
            "har_jump = max(0, r_{t-1}^2 - rv_{t-1}) (parte del retorno al cuadrado que el "
            "recorrido de Parkinson no explica)"
        ),
        regressors=_SPEC_COLUMNS["har_jump"],
        hyperparameters={"estimator": "log_lstsq", "target": "ln(parkinson_rv)"},
    ),
    Spec(
        name="har_vix_ts",
        estimator="log_lstsq",
        formula=(
            "ln(rv_t) ~ 1 + har_lag1 + har_lag4 + har_lag17 + vix9d_level + vix3m_level + "
            "vix_ts_slope, con vix_ts_slope = vix3m / vix9d - 1 (los tres de t-1)"
        ),
        regressors=_SPEC_COLUMNS["har_vix_ts"],
        hyperparameters={
            "estimator": "log_lstsq",
            "target": "ln(parkinson_rv)",
            "vix9d_series": VIX9D_SERIES_ID,
            "vix3m_series": VIX3M_SERIES_ID,
        },
    ),
    Spec(
        name="ridge_har",
        estimator="ridge",
        formula="ln(rv_t) ~ 1 + har_lag1 + har_lag4 + har_lag17, ajustado con ridge (alpha fijo)",
        regressors=_SPEC_COLUMNS["ridge_har"],
        hyperparameters={"estimator": "ridge", "alpha": RIDGE_ALPHA, "target": "ln(parkinson_rv)"},
    ),
    Spec(
        name="lasso_har",
        estimator="lasso",
        formula="ln(rv_t) ~ 1 + har_lag1 + har_lag4 + har_lag17, ajustado con lasso (alpha fijo)",
        regressors=_SPEC_COLUMNS["lasso_har"],
        hyperparameters={"estimator": "lasso", "alpha": LASSO_ALPHA, "target": "ln(parkinson_rv)"},
    ),
    Spec(
        name="quantile_har",
        estimator="quantile",
        formula=(
            "Q_0.5(ln(rv_t) | har_lag1, har_lag4, har_lag17): regresion por cuantiles (mediana) "
            "en vez de media condicional"
        ),
        regressors=_SPEC_COLUMNS["quantile_har"],
        hyperparameters={
            "estimator": "quantile",
            "quantile": QUANTILE,
            "alpha": QUANTILE_ALPHA,
            "target": "ln(parkinson_rv)",
        },
    ),
    Spec(
        name="egarch",
        estimator="egarch",
        formula=(
            "EGARCH(1,1,1) sobre r_t (media cero, errores normales): "
            "ln(sigma^2_t) = omega + beta * ln(sigma^2_{t-1}) + alpha * (|z| - E|z|) + gamma * z"
        ),
        regressors=(),
        hyperparameters={
            "estimator": "arch",
            "vol": "EGARCH",
            "p": ARCH_P,
            "o": ARCH_O,
            "q": ARCH_Q,
            "mean": ARCH_MEAN,
            "dist": ARCH_DIST,
        },
    ),
    Spec(
        name="aparch",
        estimator="aparch",
        formula=(
            "apARCH(1,1,1) sobre r_t (media cero, errores normales): "
            "sigma^delta_t = omega + alpha * (|eps| - gamma * eps)^delta + beta * sigma^delta_{t-1}"
        ),
        regressors=(),
        hyperparameters={
            "estimator": "arch",
            "vol": "APARCH",
            "p": ARCH_P,
            "o": ARCH_O,
            "q": ARCH_Q,
            "mean": ARCH_MEAN,
            "dist": ARCH_DIST,
        },
    ),
    Spec(
        name="har_intraday_rv",
        estimator="none",
        formula=(
            "ln(rv_t) ~ 1 + har_lag1 + har_lag4 + har_lag17 + rv_intradia_{t-1}, con la RV de "
            "barras de 5 minutos sumada en la ventana open->close"
        ),
        regressors=(),
        hyperparameters={},
        evaluable=False,
        reason=(
            "la RV intradia no existe en el almacen con la cobertura que el *walk-forward* "
            "necesita (~60 sesiones de `market_intraday`): no se inventa un numero"
        ),
        follow_up="#57",
    ),
)

#: Las especificaciones **evaluables**: las que producen un pronostico y entran en la correccion.
EVALUABLE_SPECS: Final[tuple[Spec, ...]] = tuple(spec for spec in SPECS if spec.evaluable)

#: Las variantes de #7 que **no** son la referencia. Entran en el registro porque fueron probadas
#: (`plan.md` §2, principio 6) y el `n_trials` del DSR las cuenta; su aritmetica **no** se
#: re-declara aqui, es la de #7 (importada). `rw` no aparece: es la linea del diferencial.
REFERENCE_SPECS: Final[tuple[Spec, ...]] = (
    Spec(
        name="har",
        estimator="log_lstsq",
        formula="ln(rv_t) ~ 1 + har_lag1 + har_lag4 + har_lag17 (el HAR de #7)",
        regressors=("har_lag1", "har_lag4", "har_lag17"),
        hyperparameters={"estimator": "log_lstsq", "target": "ln(parkinson_rv)", "source": "#7"},
    ),
    Spec(
        name="har_vix",
        estimator="log_lstsq",
        formula=("HAR + vix_level, vix_zscore y vix_percentile de t-1 (el HAR-VIX de #7)"),
        regressors=(
            "har_lag1",
            "har_lag4",
            "har_lag17",
            "vix_level",
            "vix_zscore",
            "vix_percentile",
        ),
        hyperparameters={"estimator": "log_lstsq", "target": "ln(parkinson_rv)", "source": "#7"},
    ),
    Spec(
        name="garch",
        estimator="arch_garch",
        formula=(
            "GARCH(1,1) media cero, errores normales, sin exogenas (el candidato **declarado** "
            "de #7): no es una especificacion nueva, es el liston"
        ),
        regressors=(),
        hyperparameters={
            "estimator": "arch",
            "vol": "GARCH",
            "p": ARCH_P,
            "q": ARCH_Q,
            "source": "#7",
        },
    ),
)

#: Todas las variantes que se registran: las tres de #7 que no son la referencia + las nuevas.
TRIAL_SPECS: Final[tuple[Spec, ...]] = REFERENCE_SPECS + EVALUABLE_SPECS


def trial_spec(name: str) -> Spec:
    """Declaracion registrada de esa variante: la del catalogo o la heredada de #7 (A5)."""
    for spec in TRIAL_SPECS:
        if spec.name == name:
            return spec
    raise SpecSearchError(f"`{name}` no es una variante declarada del catalogo ni de #7")


class SpecSearchError(Exception):
    """Raiz de los errores de la busqueda de especificaciones del forecast de volatilidad."""


# ─────────────────────────────────────────────────────────────────────────────
# Datos: term structure del VIX y regresores nuevos (A2, A4)
# ─────────────────────────────────────────────────────────────────────────────
def _literal(value: str) -> str:
    """Literal SQL seguro: los identificadores vienen del registro validado."""
    return "'" + value.replace("'", "''") + "'"


def load_term_structure(
    store: Store,
    *,
    short_series_id: str = VIX9D_SERIES_ID,
    mid_series_id: str = VIX3M_SERIES_ID,
) -> pl.DataFrame | None:
    """Cierres de ``^VIX9D`` y ``^VIX3M`` por sesion, o ``None`` si falta alguna (A2).

    Que falte una de las dos **no** aborta el estudio: ``har_vix_ts`` se declara ``unavailable``
    con el motivo y el informe se escribe igualmente (misma doctrina que ``load_vix`` de #7).
    Se usa ``store.sql()`` y **no** ``read_pit``, por la misma razon que #7: aqui se quiere «que
    datos existian», no «que sabiamos en T».
    """
    frames: list[pl.DataFrame] = []
    for series_id, column in ((short_series_id, "vix9d_close"), (mid_series_id, "vix3m_close")):
        query = (
            f"SELECT as_of, close AS {column} FROM raw.market_daily "  # noqa: S608
            f"WHERE series_id = {_literal(series_id)} AND close IS NOT NULL ORDER BY as_of"
        )
        try:
            frame = store.sql(query)
        except (UnknownDatasetError, duckdb.Error) as error:
            logger.warning(
                "sin {} en el almacen ({}): har_vix_ts no se podra estimar", series_id, error
            )
            return None
        if frame.height < 2:
            return None
        frames.append(frame)
    joined = frames[0].join(frames[1], on="as_of", how="inner")
    return joined.with_columns(
        pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("session")
    ).select("session", "vix9d_close", "vix3m_close")


def term_structure_regressors(
    frame: pl.DataFrame, *, columns: tuple[str, str] = ("vix9d_close", "vix3m_close")
) -> pl.DataFrame:
    """Anade ``vix9d_level``, ``vix3m_level`` y ``vix_ts_slope``, **todos de ``t-1``** (A2).

    ``vix_ts_slope = vix3m / vix9d - 1`` (pendiente de la *term structure*, adimensional), la
    formula declarada en el catalogo. ``^VIX9D`` no existe antes de 2011: esas sesiones quedan
    con los tres regresores a ``NULL`` --sin ventana parcial y sin relleno-- y ``har_vix_ts``
    simplemente no las usa.
    """
    short, mid = columns
    return frame.with_columns(
        pl.col(short).shift(1).alias("vix9d_level"),
        pl.col(mid).shift(1).alias("vix3m_level"),
        ((pl.col(mid) / pl.col(short)) - 1.0).shift(1).alias("vix_ts_slope"),
    )


def build_spec_frame(frame: pl.DataFrame, *, term_structure: pl.DataFrame | None) -> pl.DataFrame:
    """Anade a la muestra de #7 los regresores que necesita esta busqueda (A4).

    - ``har_quarticity`` = ``ln((4/3) · rv_{t-1}²)``: la cuarticidad del proxy, **en la escala
      logaritmica del objetivo**. En nivel, ``rv²`` esta diez ordenes de magnitud por debajo de
      ``rv`` y el ajuste por minimos cuadrados extrapola a varianzas absurdas: se midio antes de
      fijar el catalogo definitivo y se descarto por **unidades**, no por resultado (el porque
      queda escrito aqui, que es donde se puede comprobar).
    - ``har_jump`` = ``max(0, r_{t-1}² - rv_{t-1})``: la parte del retorno al cuadrado que el
      recorrido no explica. Es un **proxy diario** del salto (la descomposicion de verdad pide
      RV y bipower intradia, que es #57): se declara como proxy, no como salto medido.
    - los tres regresores de la *term structure*, si sus series estan en el almacen.
    """
    rv = pl.col("parkinson_rv")
    with_extra = frame.with_columns(
        ((4.0 / 3.0) * rv.pow(2)).log().shift(1).alias("har_quarticity"),
        pl.max_horizontal(pl.lit(0.0), pl.col("ret_sq").shift(1) - rv.shift(1)).alias("har_jump"),
    )
    if term_structure is None:
        return with_extra.sort("session")
    joined = with_extra.join(term_structure, on="session", how="left")
    # El `join` no garantiza el orden de las filas: se reordena por sesion para que los indices
    # de este marco y los de la muestra de #7 sigan siendo el mismo (de eso dependen los folds).
    return term_structure_regressors(joined).sort("session")


def _column(frame: pl.DataFrame, name: str) -> np.ndarray:
    """Columna como array de ``numpy``, con los nulos convertidos en ``NaN``."""
    return np.asarray(
        frame.get_column(name).cast(pl.Float64).fill_null(float("nan")).to_numpy(), dtype=float
    )


def _design_matrix(frame: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Matriz de diseño (una columna por regresor) o error si falta alguna columna."""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise SpecSearchError(
            f"faltan regresores en el marco de trabajo: {', '.join(missing)}. La especificacion "
            "queda no evaluable; no se rellena con ceros"
        )
    return np.column_stack([_column(frame, name) for name in columns])


def _log_target(
    frame: pl.DataFrame, *, column: str = TARGET_COLUMNS[Target.PARKINSON]
) -> np.ndarray:
    """``ln(rv)`` del objetivo primario, con ``NaN`` donde la varianza no es positiva."""
    variance = _column(frame, column)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(variance > 0.0, np.log(variance), np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Ajuste y pronostico de cada familia (A4)
# ─────────────────────────────────────────────────────────────────────────────
#: Iteraciones maximas del lasso (constante declarada: no se busca, se fija).
LASSO_MAX_ITER: Final[int] = 10_000

#: Limite del logaritmo de la varianza antes de que ``exp`` desborde; misma guarda que el
#: ``HarCoefficients`` de #7. Fuera de ahi no hay pronostico, nunca ``inf``.
_MAX_LOG_VARIANCE: Final[float] = 700.0


def _coefficients(intercept: float, slopes: Sequence[float]) -> HarCoefficients | None:
    """Envuelve unos coeficientes ajustados en el ``HarCoefficients`` de #7, o ``None``.

    Se reutiliza el objeto de #7 a proposito: su ``predict`` deshace el logaritmo y devuelve la
    varianza **positiva por construccion** (y ``None`` si desborda), asi que las familias
    lineales de esta busqueda comparten con el HAR de #7 esa garantia, no una copia.
    """
    values = (float(intercept), *(float(value) for value in slopes))
    if not all(math.isfinite(value) for value in values):
        return None
    return HarCoefficients(intercept=values[0], slopes=tuple(values[1:]))


def _standardise(design: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estandariza cada columna con la media y la desviacion de la **ventana de entrenamiento**.

    Una columna constante (desviacion cero) se deja como esta: su efecto lo absorbe el
    intercepto, no se inventa una escala.
    """
    mean = design.mean(axis=0)
    std = design.std(axis=0, ddof=0)
    safe = np.where(std > 0.0, std, 1.0)
    return (design - mean) / safe, mean, safe


def _unstandardise(
    coefficients: HarCoefficients | None, mean: np.ndarray, std: np.ndarray
) -> HarCoefficients | None:
    """Devuelve unos coeficientes estandarizados al espacio **original** de los regresores.

    Es una transformacion afin: ``b_original = b_estandar / std`` y
    ``a_original = a_estandar - Σ b_estandar · mean / std``. Con esto el pronostico se calcula
    siempre sobre los regresores crudos, igual que en las familias no penalizadas, y los
    coeficientes que se publican siguen siendo los del modelo en las unidades de #7.
    """
    if coefficients is None:
        return None
    slopes = np.asarray(coefficients.slopes, dtype=float)
    raw_slopes = slopes / std
    raw_intercept = coefficients.intercept - float(np.sum(slopes * mean / std))
    return _coefficients(raw_intercept, [float(value) for value in raw_slopes])


def _fit_lstsq(design: np.ndarray, log_target: np.ndarray) -> HarCoefficients | None:
    """Minimos cuadrados en ``ln(rv)``: **la misma** funcion que usa el HAR de #7.

    Los regresores se estandarizan antes de ajustar y los coeficientes se devuelven al espacio
    original: para minimos cuadrados el hiperplano es **el mismo** (la transformacion es afin),
    asi que esta spec no puede desviarse del HAR de #7 por culpa del escalado.
    """
    scaled, mean, std = _standardise(design)
    return _unstandardise(fit_log_har(scaled, log_target, min_train=HAR_MIN_TRAIN), mean, std)


def _fit_sklearn(
    design: np.ndarray, log_target: np.ndarray, *, estimator: str
) -> HarCoefficients | None:
    """Ajusta ridge, lasso o cuantiles en log-varianza con las mismas guardas que #7.

    Sin ``HAR_MIN_TRAIN`` filas, con valores no finitos o con un ajuste no finito **no hay
    ajuste** (``None``): nunca uno inventado. Las penalizaciones y el cuantil son constantes del
    modulo, no se eligen contra la muestra.

    Los regresores **se estandarizan** (media 0, desviacion 1) dentro de la ventana de
    entrenamiento: para la regresion por cuantiles es indiferente --el hiperplano es el mismo-- y
    para ridge y lasso es la convencion obligada, porque una penalizacion ``alpha`` sobre
    coeficientes medidos en escalas distintas (aqui, varianzas del orden de ``1e-5``) castiga mas
    al regresor con unidades mas pequenas y deja de ser la penalizacion que se declaro.
    """
    if design.shape[0] < HAR_MIN_TRAIN or design.shape[1] < 1:
        return None
    if not (bool(np.isfinite(design).all()) and bool(np.isfinite(log_target).all())):
        return None
    scaled, mean, std = _standardise(design)
    try:
        if estimator == "ridge":
            model: Any = Ridge(alpha=RIDGE_ALPHA)
        elif estimator == "lasso":
            model = Lasso(alpha=LASSO_ALPHA, max_iter=LASSO_MAX_ITER)
        else:
            model = QuantileRegressor(quantile=QUANTILE, alpha=QUANTILE_ALPHA, solver="highs")
        fitted = model.fit(scaled, log_target)
    except Exception as error:  # el solver falla de muchas formas distintas
        logger.warning("ajuste {} no estimable: {}: {}", estimator, type(error).__name__, error)
        return None
    slopes = [float(value) for value in fitted.coef_]
    return _unstandardise(_coefficients(float(fitted.intercept_), slopes), mean, std)


def _linear_forecasts(
    frame: pl.DataFrame,
    *,
    columns: Sequence[str],
    bounds: Sequence[tuple[int, int]],
    estimator: str,
) -> tuple[np.ndarray, str | None]:
    """Walk-forward de una especificacion lineal en log-varianza (A4).

    El esquema es el de #7, sin tocar: los parametros se ajustan con **todas** las sesiones
    anteriores al tramo y quedan **congelados** dentro de el (el reajuste es cada
    ``REFIT_EVERY`` sesiones); el pronostico de ``t`` usa solo los regresores de ``t``, que son
    todos de ``t-1`` o anteriores. Donde no hay ajuste el pronostico queda ``NaN``.
    """
    sessions = frame.height
    forecasts = np.full(sessions, np.nan)
    design = _design_matrix(frame, columns)
    target = _log_target(frame)
    predictable = np.isfinite(design).all(axis=1)
    trainable = predictable & np.isfinite(target)
    reason: str | None = None
    for start, end in bounds:
        train = np.flatnonzero(trainable[:start])
        if train.size < HAR_MIN_TRAIN:
            continue
        fitted = (
            _fit_lstsq(design[train], target[train])
            if estimator == "log_lstsq"
            else _fit_sklearn(design[train], target[train], estimator=estimator)
        )
        if fitted is None:
            # Un tramo que no se puede ajustar deja sus sesiones a NaN y el estudio **sigue**:
            # abortar por un tramo degenerado (una columna constante en esa ventana, p. ej.)
            # tiraria los tramos que si se pueden ajustar. El motivo se publica si no hay
            # ningun pronostico; si hay, la fila es `ok` y las sesiones que faltan se declaran.
            reason = reason or (
                f"la especificacion `{estimator}` no es estimable con las {train.size} filas de "
                f"entrenamiento del tramo [{start}, {end})"
            )
            continue
        for index in range(start, end):
            if not predictable[index]:
                continue
            predicted = fitted.predict(*(float(value) for value in design[index]))
            if predicted is not None:
                forecasts[index] = predicted
    if bool(np.isfinite(forecasts).any()):
        return forecasts, None
    return forecasts, reason


@dataclass(frozen=True, slots=True)
class ArchState:
    """Estado de la recursion de una familia ``arch``: varianza y ultimo residuo, en fraccion."""

    sigma2: float
    epsilon: float


@dataclass(frozen=True, slots=True)
class ArchFit:
    """Parametros ajustados de EGARCH/apARCH y el estado al final del entrenamiento."""

    params: Mapping[str, float]
    state: ArchState


def fit_arch(returns: np.ndarray, *, vol: str) -> ArchFit:
    """Ajusta EGARCH(1,1,1) o apARCH(1,1,1) con ``arch``, con la receta de #7.

    Media cero, errores normales, sin regresores exogenos y ``rescale=False``: los parametros
    quedan en las unidades de la serie (**fracciones**), que es lo que permite recursar la
    varianza condicional paso a paso. Un ajuste no finito o con varianza no admisible es un
    ``ValueError``: el llamante decide si eso es un fallo o un pronostico que no existe.
    """
    model = arch_model(
        returns,
        mean=ARCH_MEAN,
        vol=cast("Any", vol),
        p=ARCH_P,
        o=ARCH_O,
        q=ARCH_Q,
        dist=ARCH_DIST,
        rescale=False,
    )
    result = cast("Any", model.fit(disp="off", show_warning=False))
    params = {name: float(value) for name, value in result.params.items()}
    conditional = np.asarray(result.conditional_volatility, dtype=float)
    state = ArchState(sigma2=float(conditional[-1]) ** 2, epsilon=float(returns[-1]))
    if not all(math.isfinite(value) for value in (state.sigma2, state.epsilon)):
        raise ValueError(f"el ajuste {vol} no ha dado un estado finito")
    if state.sigma2 <= 0.0:
        raise ValueError(f"el ajuste {vol} no ha dado una varianza admisible")
    return ArchFit(params=params, state=state)


#: Nombres de los parametros que la recursion necesita de ``arch`` (p = o = q = 1).
_ARCH_PARAMETERS: Final[dict[str, tuple[str, ...]]] = {
    "EGARCH": ("omega", "alpha[1]", "gamma[1]", "beta[1]"),
    "APARCH": ("omega", "alpha[1]", "gamma[1]", "beta[1]", "delta"),
}


def arch_step(fit: ArchFit, *, vol: str) -> float:
    """Varianza pronosticada a un paso (fraccion²) con los parametros del ajuste.

    ``EGARCH``: ``ln sigma^2_t = omega + beta * ln sigma^2_{t-1} + alpha * (|z| - E|z|) +
    gamma * z``, con ``z = eps / sigma`` y ``E|z| = sqrt(2/pi)`` (normal estandar, la
    distribucion declarada).

    ``APARCH``: ``sigma^delta_t = omega + alpha * (|eps| - gamma * eps)^delta +
    beta * sigma^delta_{t-1}``.

    Las dos recursiones se contrastan en los tests contra ``forecast(horizon=1)`` de ``arch``
    sobre el mismo ajuste: si divergieran, el numero publicado seria plausible y falso.
    """
    missing = [name for name in _ARCH_PARAMETERS[vol] if name not in fit.params]
    if missing:
        return float("nan")
    state = fit.state
    if vol == "EGARCH":
        z = state.epsilon / math.sqrt(state.sigma2)
        log_variance = (
            fit.params["omega"]
            + fit.params["beta[1]"] * math.log(state.sigma2)
            + fit.params["alpha[1]"] * (abs(z) - NORMAL_ABS_MOMENT)
            + fit.params["gamma[1]"] * z
        )
        if not math.isfinite(log_variance) or log_variance > _MAX_LOG_VARIANCE:
            return float("nan")
        return math.exp(log_variance)
    delta = fit.params["delta"]
    if not math.isfinite(delta) or delta <= 0.0:
        return float("nan")
    powered = (
        fit.params["omega"]
        + fit.params["alpha[1]"]
        * (abs(state.epsilon) - fit.params["gamma[1]"] * state.epsilon) ** delta
        + fit.params["beta[1]"] * state.sigma2 ** (delta / 2.0)
    )
    if not math.isfinite(powered) or powered <= 0.0:
        return float("nan")
    variance = powered ** (2.0 / delta)
    return variance if math.isfinite(variance) and variance > 0.0 else float("nan")


def _arch_forecasts(
    returns: np.ndarray, *, bounds: Sequence[tuple[int, int]], vol: str
) -> tuple[np.ndarray, str | None]:
    """Walk-forward de EGARCH/apARCH con los parametros congelados dentro de cada tramo.

    Es el mismo esquema que ``garch_forecasts`` de #7: reajuste al empezar el tramo y recursion
    con los retornos realizados dentro de el. Un paso no finito deja esa sesion a ``NaN`` y
    **conserva** la varianza anterior, para que un desborde puntual no envenene el resto.
    """
    sessions = returns.shape[0]
    forecasts = np.full(sessions, np.nan)
    reason: str | None = None
    for start, end in bounds:
        try:
            fit = fit_arch(returns[:start], vol=vol)
        except Exception as error:  # `arch` falla de muchas formas distintas
            reason = reason or f"{vol} no estimable: {type(error).__name__}: {error}"
            continue
        for index in range(start, end):
            variance = arch_step(fit, vol=vol)
            if math.isfinite(variance) and variance > 0.0:
                forecasts[index] = variance
                current = variance
            else:
                current = fit.state.sigma2
            fit = ArchFit(
                params=fit.params, state=ArchState(sigma2=current, epsilon=float(returns[index]))
            )
    if bool(np.isfinite(forecasts).any()):
        return forecasts, None
    return forecasts, reason


def spec_forecasts(
    frame: pl.DataFrame, *, spec: Spec, bounds: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, str | None]:
    """Pronosticos de una especificacion del catalogo, o el motivo por el que no los hay (A4).

    Nunca lanza por causas del dato: una especificacion no evaluable, un regresor que falta o un
    ajuste imposible devuelven un vector de ``NaN`` **con su motivo**, jamas un numero.
    """
    if not spec.evaluable:
        return np.full(frame.height, np.nan), spec.reason or "declarada no evaluable"
    if spec.estimator in {"egarch", "aparch"}:
        return _arch_forecasts(_column(frame, "ret_log"), bounds=bounds, vol=spec.estimator.upper())
    try:
        return _linear_forecasts(
            frame, columns=spec.regressors, bounds=bounds, estimator=spec.estimator
        )
    except SpecSearchError as error:
        return np.full(frame.height, np.nan), str(error)


# ─────────────────────────────────────────────────────────────────────────────
# Perdida, diferenciales y la matriz de la correccion (A6, A7)
# ─────────────────────────────────────────────────────────────────────────────
def qlike_losses(
    forecast: np.ndarray, target: np.ndarray, *, mask: np.ndarray | None = None
) -> np.ndarray:
    """Perdida QLIKE **por sesion** (``ln v_hat + v / v_hat``), ``NaN`` donde no es evaluable.

    Es el sumando del que sale el QLIKE medio de #7 (``mean`` de esta serie es su ``Metrics.qlike``
    sobre las mismas filas): por eso se puede comparar por sesion sin reimplementar la metrica.
    """
    loss = np.full(forecast.shape, np.nan)
    valid = np.isfinite(forecast) & (forecast > 0.0) & np.isfinite(target) & (target > 0.0)
    if mask is not None:
        valid &= mask
    predicted = forecast[valid]
    observed = target[valid]
    loss[valid] = np.log(predicted) + observed / predicted
    return loss


def loss_differentials(reference_losses: np.ndarray, losses: np.ndarray) -> np.ndarray:
    """Diferencial de perdida por sesion: ``d_t = L_ref,t - L_i,t`` (A6).

    ``d_t > 0`` significa que la especificacion pierde **menos** que la referencia, o sea que es
    mejor. Es la serie que consumen el DSR y el PBO: un diferencial, no un P&L.
    """
    return reference_losses - losses


def blocks_for(sessions: int, *, maximum: int = BLOCKS_MAX) -> int | None:
    """Bloques del CSCV: el mayor numero **par** que no pasa de ``maximum`` ni de ``T`` (A7).

    El CSCV exige bloques **pares** y de igual tamano. ``T`` no tiene por que ser divisible -- con
    las 2435 sesiones comunes del almacen vigente **ningun** par divide a ``T``--, asi que la regla
    no es «un divisor de ``T``» sino «el mayor par admisible»: quien construye la matriz
    **recorta** las sesiones mas antiguas hasta el mayor multiplo de los bloques que no pasa de
    ``T`` (el informe publica cuantas). El recorte es deterministico y declarado, y con ``T < 2``
    no hay bloques: el PBO queda ``not_evaluable``, nunca un ``pbo: 0``.
    """
    top = maximum if maximum % 2 == 0 else maximum - 1
    blocks = min(top, sessions - (sessions % 2))
    return blocks if blocks >= 2 else None


def spec_row(
    *,
    spec: Spec,
    forecast: np.ndarray,
    reason: str | None,
    targets: Mapping[str, np.ndarray],
    evaluated: np.ndarray,
) -> CandidateResult:
    """Fila de la tabla para una especificacion: metricas de #7 o ``unavailable`` con su motivo.

    Las metricas salen de :func:`evaluate_forecast` (la de #7, importada) sobre la **misma**
    mascara de evaluacion y los **mismos** objetivos. Si no hay ninguna metrica, la fila es
    ``unavailable``: no se publica un cero.
    """
    metrics: dict[str, Metrics | None] = {
        name: evaluate_forecast(forecast, target, mask=evaluated)
        for name, target in targets.items()
    }
    available = [value for value in metrics.values() if value is not None]
    sigma = forecast[evaluated]
    sigma = sigma[np.isfinite(sigma) & (sigma > 0.0)]
    status = Status.OK if available else Status.UNAVAILABLE
    detail = reason
    if status is Status.UNAVAILABLE and detail is None:
        detail = "no se ha podido producir ningun pronostico evaluable"
    return CandidateResult(
        name=spec.name,
        status=status.value,
        reason=detail if status is Status.UNAVAILABLE else None,
        metrics=metrics,
        median_sigma_bp=float(np.median(np.sqrt(sigma))) * 10_000.0 if sigma.size else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registro de los ensayos (A5): la identidad es el contenido y va a su propia raiz
# ─────────────────────────────────────────────────────────────────────────────
#: Prefijo del ``variant_id`` de los ensayos: el mismo vocabulario que el barrido de #82
#: (``lightgbm_search_v1#<nombre>``), para que se vea de un vistazo que son de una busqueda.
VOL_SPEC_PREFIX: Final[str] = "volatility_spec_v1#"


def spec_variant_id(name: str) -> str:
    """``variant_id`` del ensayo de esa especificacion."""
    return f"{VOL_SPEC_PREFIX}{name}"


def spec_config(
    *,
    spec: Spec,
    summary: Mapping[str, object],
    sessions: int,
    evaluated_sessions: int,
    folds: int,
    reference: str,
) -> ExperimentConfig:
    """Configuracion registrada de una especificacion (A5).

    Declara lo que `plan.md` §11.4 exige anotar por variante --features, hiperparametros,
    semilla, serie y ventana-- y ademas **las unidades**: ``units`` dice que el numero es un
    diferencial de perdida, ``target`` que el objetivo es el primario de #7 y ``reference``
    frente a quien se mide. La ventana es la de #7, sin cambios.
    """
    return ExperimentConfig(
        variant_id=spec_variant_id(spec.name),
        features=spec.regressors if spec.regressors else ("ret_log",),
        hyperparameters=dict(spec.hyperparameters),
        seed=SPEC_SEED,
        series_id=SERIES_ID,
        window={
            "sample": "muestra limpia de #52, la de #7 sin cambios",
            "clean_from": summary.get("clean_from"),
            "clean_sessions": summary.get("clean_sessions"),
            "sessions": sessions,
            "evaluated_sessions": evaluated_sessions,
            "folds": folds,
            "min_train": MIN_TRAIN,
            "refit_every": REFIT_EVERY,
            "horizon": 1,
            "target": TARGET_COLUMNS[Target.PARKINSON],
            "metric": "qlike",
            "reference": reference,
            "units": BASIS_LOSS_DIFFERENTIAL,
        },
    )


def register_spec(
    *,
    trials_root: Path,
    config: ExperimentConfig,
    sharpe_per_session: float,
    n_observations: int,
    as_of: datetime,
    write: bool,
) -> ExperimentRecord:
    """Registra el ensayo de una especificacion (#16): identidad = contenido, nunca sobrescribe.

    El ``basis`` declarado es :data:`BASIS_LOSS_DIFFERENTIAL`, no ``declared_cost``: el numero es
    un diferencial de perdida y escribirlo como un Sharpe de estrategia seria mentir sobre las
    unidades. El bloque ``net_metrics`` que trae el formato de #16 se publica igual, declarando
    que las metricas netas no son computables: es el mismo fichero que lee ``load_registry``.
    """
    return record_experiment(
        runs_root=trials_root,
        config=config,
        result=ExperimentResult(
            sharpe_per_session=sharpe_per_session,
            n_observations=n_observations,
            basis=BASIS_LOSS_DIFFERENTIAL,
        ),
        as_of=as_of,
        write=write,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Correccion por comparaciones multiples (A6, A7) y veredicto (A8)
# ─────────────────────────────────────────────────────────────────────────────
def loss_units(series: np.ndarray, *, where: str = "serie") -> np.ndarray:
    """Reescala un diferencial de perdida al dominio que ``sharpe_ratio`` acepta (A6).

    ``sharpe_ratio`` (#13) valida sus entradas como **retornos**: rechaza cualquier valor
    ``<= -1`` («un retorno no puede perder mas del 100 %»). Un diferencial de perdida **no** es un
    retorno y puede valer ``1e281``, asi que se divide por ``2 · max|d|``, una constante
    **positiva y comun a toda la serie**. El ``Sharpe`` es invariante a un escalado positivo
    (``c·mean / (c·std)``), asi que el numero que se publica es **el mismo** que el del
    diferencial sin escalar: lo unico que cambia es el dominio en el que se puede calcular. Se
    divide por el doble para que ningun valor caiga justo en el limite que la validacion rechaza.

    Una serie degenerada (magnitud cero o no finita) se devuelve tal cual: de ahi sale un Sharpe
    ``0`` por la convencion de #13, nunca un ratio inventado.
    """
    magnitude = float(np.max(np.abs(series))) if series.size else 0.0
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        logger.warning(f"{where}: el diferencial no tiene magnitud utilizable; no se reescala")
        return series
    return series / (2.0 * magnitude)


def correction_inputs(
    *,
    names: Sequence[str],
    differentials: Mapping[str, np.ndarray],
    evaluated: np.ndarray,
    skipped: Mapping[str, str],
) -> tuple[list[str], np.ndarray, dict[str, float], dict[str, str]]:
    """Variantes registrables, sesiones **comunes** y su Sharpe de diferencial (A6).

    ``common`` exige que **todas** las variantes tengan diferencial finito en esa sesion: comparar
    variantes en ventanas distintas no seria comparar. Una variante cuyo diferencial es degenerado
    (varianza cero) no se puede registrar: se declara y sale del conjunto en vez de inventar un
    Sharpe.
    """
    usable = list(names)
    skips = dict(skipped)
    while usable:
        common = evaluated.copy()
        for name in usable:
            common &= np.isfinite(differentials[name])
        sharpes: dict[str, float] = {}
        degenerate: list[str] = []
        for name in usable:
            series = [float(value) for value in loss_units(differentials[name][common])]
            try:
                sharpes[name] = sharpe_ratio(series, annualization=1)
            except Exception as error:  # una serie que no se puede medir
                degenerate.append(name)
                skips[name] = (
                    f"diferencial no medible ({type(error).__name__}: {error}): no se registra"
                )
        if not degenerate:
            return usable, common, sharpes, skips
        usable = [name for name in usable if name not in degenerate]
    raise SpecSearchError(
        "ninguna variante tiene un diferencial con Sharpe: sin registro no hay DSR ni PBO, y un "
        "informe sin correccion no responderia la pregunta de la tarea. No se escribe nada"
    )


def deflated_block(
    *,
    selected: str,
    differentials: Mapping[str, np.ndarray],
    common: np.ndarray,
    registry: Registry,
) -> tuple[DeflatedSharpe | None, str | None]:
    """DSR de la especificacion seleccionada, con el ``n_trials`` **derivado** del registro (A6).

    :func:`require_trials_match_registry` es la puerta de #16: el ``n_trials`` y la ``V[SR]`` que
    entran en la formula son los del registro de ensayos, nunca un campo del informe ni una
    opcion del CLI. Sin registro (o sin dos variantes) no hay DSR: se declara el motivo y **no**
    se publica un numero.
    """
    try:
        require_trials_match_registry(
            n_trials=registry.n_trials, sr_variance=registry.sr_variance, registry=registry
        )
    except (ExperimentLogError, OverfittingError) as error:
        return None, f"el registro de ensayos no permite derivar el DSR: {error}"
    series = loss_units(differentials[selected][common])
    try:
        block = deflated_sharpe_ratio(
            [float(value) for value in series],
            n_trials=registry.n_trials,
            sr_variance=registry.sr_variance,
        )
    except OverfittingError as error:
        return None, f"DSR no evaluable: {error}"
    return block, None


def pbo_block(
    *, names: Sequence[str], differentials: Mapping[str, np.ndarray], common: np.ndarray
) -> tuple[BacktestOverfitting | None, str | None]:
    """PBO por CSCV sobre la matriz ``T x N`` de diferenciales, con ``S`` calculado de ``T`` (A7).

    La matriz se construye sobre las sesiones **comunes** (todas las columnas finitas): comparar
    variantes en ventanas distintas no seria comparar. Cuantas sesiones y cuantas columnas entran
    lo publica el bloque, y si no hay divisor par valido el bloque sale ``not_evaluable``.
    """
    if len(names) < 2:
        return None, (
            "el PBO necesita al menos dos variantes con diferencial finito; hay "
            f"{len(names)}: no se inventa un `pbo: 0`"
        )
    blocks = blocks_for(int(common.sum()))
    if blocks is None:
        return None, (
            f"con {int(common.sum())} sesiones comunes no hay ni dos bloques del CSCV: el PBO "
            "queda `not_evaluable`"
        )
    # Se recortan las sesiones **mas antiguas** hasta el mayor multiplo de los bloques: los
    # bloques del CSCV tienen que ser contiguos y de igual tamano.
    used = int(common.sum()) - (int(common.sum()) % blocks)
    rows = np.flatnonzero(common)[-used:]
    columns = [loss_units(differentials[name][rows], where=name) for name in names]
    # La matriz se arma fila a fila (sesion a sesion) y como listas de `float`: es lo que la
    # funcion de #16 acepta, y deja el orden explicito (filas = observaciones, columnas =
    # variantes).
    matrix: list[list[float]] = [
        [float(column[index]) for column in columns] for index in range(used)
    ]
    try:
        block = probability_of_backtest_overfitting(matrix, blocks=blocks)
    except OverfittingError as error:
        return None, f"PBO no evaluable: {error}"
    return block, None


def best_new_spec(rows: Sequence[CandidateResult], *, names: Sequence[str]) -> str | None:
    """La mejor de las especificaciones nuevas, con el desempate **declarado** (A8).

    Orden: menor QLIKE del objetivo primario; a igualdad, menor MSE del logaritmo del primario; a
    igualdad otra vez, el orden del catalogo (escrito antes de medir, asi que es estable).
    Devuelve ``None`` si ninguna de las nuevas tiene metrica primaria.
    """
    primary = Target.PARKINSON.value
    best: str | None = None
    best_key: tuple[float, float] | None = None
    for row in rows:
        if row.name not in names or row.status != Status.OK.value:
            continue
        metric = row.metrics.get(primary)
        if metric is None:
            continue
        key = (metric.qlike, metric.mse_log)
        if best_key is None or key < best_key:
            best, best_key = row.name, key
    return best


def improvement_verdict(
    *,
    reference: str,
    best_new: str | None,
    gaps: Mapping[str, float | None],
    deflated: DeflatedSharpe | None,
    pbo: BacktestOverfitting | None,
) -> dict[str, object]:
    """Las **tres** condiciones de A8, en un solo sitio y con su aritmetica publicada.

    ``improves`` solo es ``True`` si (a) la mejor nueva bate a la referencia por mas de
    ``RELATIVE_TOLERANCE`` en los **dos** objetivos, (b) el DSR es ``significant`` y (c) el PBO es
    ``not_detected``. Si el DSR o el PBO no son evaluables el veredicto es ``inconclusive`` y
    ``improves`` es ``False``: un ``not_evaluable`` nunca se escribe como aprobado.
    """
    primary, secondary = Target.PARKINSON.value, Target.RET_SQ.value
    first, second = gaps.get(primary), gaps.get(secondary)
    beats_reference = bool(
        first is not None
        and second is not None
        and first < -RELATIVE_TOLERANCE
        and second < -RELATIVE_TOLERANCE
    )
    dsr_ok = deflated is not None and deflated.verdict == VERDICT_SIGNIFICANT
    pbo_ok = pbo is not None and pbo.verdict == VERDICT_NOT_DETECTED
    conditions: dict[str, object] = {
        "beats_reference_beyond_tolerance": beats_reference,
        "deflated_sharpe_significant": dsr_ok,
        "pbo_not_detected": pbo_ok,
        "tolerance": RELATIVE_TOLERANCE,
        "dsr_confidence_level": DSR_CONFIDENCE_LEVEL,
        "pbo_max": PBO_MAX,
        "gap_primary": first,
        "gap_secondary": second,
    }
    if deflated is None or pbo is None:
        if not beats_reference:
            # Con la condicion (a) ya medida y fallando, el veredicto **no** es una inferencia: la
            # brecha no llega a la tolerancia pre-registrada y ninguna correccion puede convertir
            # eso en una mejora. Se dice, aunque falte un bloque.
            return {
                "verdict": VERDICT_KEEPS_7,
                "improves": False,
                "best_new": best_new,
                "conditions": conditions,
                "reason": (
                    f"`{best_new}` no bate a `{reference}` mas alla de la tolerancia declarada en "
                    f"los dos objetivos: se queda el candidato de #7 (`{reference}`), y eso no "
                    "depende de la correccion"
                ),
            }
        missing = "el DSR" if deflated is None else "el PBO"
        return {
            "verdict": VERDICT_INCONCLUSIVE,
            "improves": False,
            "best_new": best_new,
            "conditions": conditions,
            "reason": (
                f"sin {missing} no hay veredicto posible: con `{VERDICT_NOT_EVALUABLE}` no se "
                f"puede decir que una especificacion bata a `{reference}`, y no se dice"
            ),
        }
    if beats_reference and dsr_ok and pbo_ok:
        return {
            "verdict": VERDICT_IMPROVES,
            "improves": True,
            "best_new": best_new,
            "conditions": conditions,
            "reason": (
                f"`{best_new}` bate a `{reference}` por mas de {RELATIVE_TOLERANCE:.0%} en los dos "
                "objetivos y sobrevive a la correccion (DSR significativo y PBO no detectado): "
                "merece que §19.16 lo considere, y esa decision es de #63"
            ),
        }
    if not beats_reference:
        detail = f"`{best_new}` no bate a `{reference}` mas alla de la tolerancia declarada"
    elif not dsr_ok:
        detail = (
            f"`{best_new}` bate a `{reference}` en la muestra, pero el DSR no es significativo con "
            f"{deflated.n_trials} variantes probadas: la ventaja no se distingue del azar"
        )
    else:
        detail = (
            f"`{best_new}` bate a `{reference}`, pero el PBO ({pbo.pbo:.4f}) no esta por debajo de "
            f"{PBO_MAX:.0%}: lo mejor en muestra y lo mejor fuera de muestra no coinciden"
        )
    return {
        "verdict": VERDICT_KEEPS_7,
        "improves": False,
        "best_new": best_new,
        "conditions": conditions,
        "reason": f"{detail}. Se queda el candidato de #7 (`{reference}`)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Informe (A9)
# ─────────────────────────────────────────────────────────────────────────────
#: Limitaciones declaradas del informe. Ni una se esconde y ninguna se arregla aqui.
LIMITATIONS: Final[tuple[str, ...]] = (
    "La correccion se aplica sobre **diferenciales de perdida QLIKE**, no sobre el P&L de una "
    "estrategia: el `Sharpe` del DSR es una ventaja estandarizada por sesion, no un retorno (A6).",
    "El objetivo es el **proxy diario** de #7 (Parkinson y `r^2`), no la RV intradia: `har_jump` "
    "usa un **proxy diario** del salto y `har_intraday_rv` esta declarada no evaluable (#57).",
    "`^VIX9D` empieza en 2011 (la fuente no tiene mas) y los dos contratos del VIX son indices, "
    "no el CFD: la *term structure* entra como regresor exogeno, no como coste.",
    "Las variantes se comparan en las sesiones **comunes** a todas ellas: una sesion sin "
    "diferencial finito en alguna queda fuera de la matriz y se declara cuantas y por que.",
    "El registro de ensayos vive en su propia raiz y **no** entra en `runs/`: la cadena de "
    "modelos enumera todas las entradas de alli (acoplamiento documentado en #158).",
)

#: Notas de procedimiento (A9).
NOTES: Final[tuple[str, ...]] = (
    "Las cuatro filas de #7 se recalculan llamando a su `walk_forward` sobre la misma muestra: "
    "no se copian de su informe congelado, y por eso el informe de #7 no se toca.",
    "Ninguna especificacion busca hiperparametros: las penalizaciones, el cuantil y los ordenes "
    "de `arch` son constantes del modulo, escritas antes de medir (A1).",
    "La busqueda no se amplia despues de ver el resultado: anadir una variante es otra tarea, "
    "con su propia correccion (plan.md §3.5).",
    "Ninguna especificacion se estabiliza, se acota ni se suaviza despues de ver el resultado: si "
    "un filtro se va al suelo en un tramo, su QLIKE lo refleja y se publica tal cual.",
)


@dataclass(frozen=True, slots=True)
class SpecSearchReport:
    """Resultado completo de la busqueda, listo para el informe."""

    as_of: datetime
    report_date: date
    payload: dict[str, object]
    report_sha256: str
    rows: tuple[CandidateResult, ...]
    registry: Registry
    records: tuple[ExperimentRecord, ...]

    def published(self) -> dict[str, object]:
        """El documento que se escribe: el payload **con** su propio `report_sha256`."""
        return {**self.payload, "report_sha256": self.report_sha256}


def _gap(row: CandidateResult, reference: CandidateResult | None, target: Target) -> float | None:
    """Brecha relativa de QLIKE de una fila frente a otra, o ``None`` si falta alguna metrica."""
    if reference is None:
        return None
    left = row.metrics.get(target.value)
    right = reference.metrics.get(target.value)
    if left is None or right is None:
        return None
    return relative_qlike_gap(left.qlike, right.qlike)


def _row_payload(
    row: CandidateResult,
    *,
    differential_reference: CandidateResult | None,
    declared: CandidateResult | None,
    has_differential: bool,
    differential_sharpe: float | None,
    skip_reason: str | None,
) -> dict[str, object]:
    """Una fila de la tabla con sus dos brechas relativas, su metrica y su estado (A9)."""
    return {
        "status": row.status,
        "reason": row.reason,
        "median_sigma_bp": row.median_sigma_bp,
        "is_differential_reference": row.name == REFERENCE,
        "in_correction": has_differential,
        "differential_sharpe": differential_sharpe,
        "skip_reason": skip_reason,
        "gap_vs_rw_primary": _gap(row, differential_reference, Target.PARKINSON),
        "gap_vs_rw_secondary": _gap(row, differential_reference, Target.RET_SQ),
        "gap_vs_declared_primary": _gap(row, declared, Target.PARKINSON),
        "gap_vs_declared_secondary": _gap(row, declared, Target.RET_SQ),
        "metrics": {
            target.value: None
            if row.metrics.get(target.value) is None
            else asdict(cast("Metrics", row.metrics[target.value]))
            for target in TARGETS
        },
    }


def _payload(
    *,
    moment: datetime,
    summary: Mapping[str, object],
    sessions: int,
    evaluated_sessions: int,
    first_evaluated: str | None,
    last_evaluated: str | None,
    folds: int,
    rows: Sequence[CandidateResult],
    differentials: Mapping[str, np.ndarray],
    names: Sequence[str],
    common: np.ndarray,
    skipped: Mapping[str, str],
    sharpes: Mapping[str, float],
    registry: Registry,
    deflated: DeflatedSharpe | None,
    deflated_reason: str | None,
    pbo: BacktestOverfitting | None,
    pbo_reason: str | None,
    verdict: Mapping[str, object],
) -> dict[str, object]:
    """El payload del informe, JSON puro y sin rutas absolutas (A9, A10)."""
    by_name = {row.name: row for row in rows}
    declared = by_name.get(DECLARED_CANDIDATE.candidate)
    reference_row = by_name.get(REFERENCE)
    per_row_gaps = {
        name: {target.value: _gap(row, by_name.get(REFERENCE), target) for target in TARGETS}
        for name, row in by_name.items()
    }
    return {
        "as_of": moment.isoformat(),
        "series_id": SERIES_ID,
        "vix_term_structure_series": [VIX9D_SERIES_ID, VIX3M_SERIES_ID],
        "catalogue": [spec.to_payload() for spec in SPECS],
        "catalogue_size": len(SPECS),
        "evaluable_specs": len(EVALUABLE_SPECS),
        "not_evaluable_specs": [spec.name for spec in SPECS if not spec.evaluable],
        "sample": dict(summary),
        "walk_forward": {
            "window": "expanding",
            "min_train": MIN_TRAIN,
            "refit_every": REFIT_EVERY,
            "horizon": 1,
            "folds": folds,
            "sessions": sessions,
            "evaluated_sessions": evaluated_sessions,
            "first_evaluated": first_evaluated,
            "last_evaluated": last_evaluated,
            "rule": "el de #7, importado: no se cambia desde aqui",
        },
        "references": {
            "differential": REFERENCE,
            "differential_reason": (
                "`rw` es la persistencia de #7 y la linea contra la que se mide el diferencial; "
                "**no** es un trial: su diferencial consigo misma es identicamente cero"
            ),
            "declared_candidate": DECLARED_CANDIDATE.candidate,
            "declaration": {
                "decided_on": DECLARED_CANDIDATE.decided_on.isoformat(),
                "decided_by": DECLARED_CANDIDATE.decided_by,
                "source": DECLARED_CANDIDATE.source,
                "reason": DECLARED_CANDIDATE.reason,
            },
            "target_of_the_verdict": "el candidato declarado de #7",
        },
        "rows": {
            name: _row_payload(
                row,
                differential_reference=reference_row,
                declared=declared,
                has_differential=name in differentials and name in sharpes,
                differential_sharpe=sharpes.get(name),
                skip_reason=skipped.get(name),
            )
            for name, row in by_name.items()
        },
        "correction": {
            "units_note": CORRECTION_UNITS_NOTE,
            "basis": BASIS_LOSS_DIFFERENTIAL,
            "n_variants": len(names),
            "variant_ids": list(names),
            "common_sessions": int(common.sum()),
            "session_rule": (
                "la matriz y el DSR usan las sesiones **comunes**: todas las variantes con "
                "diferencial finito en esa sesion"
            ),
            "dropped_sessions": {
                name: int(
                    (~np.isfinite(differentials[name]) & (differentials[name].size > 0)).sum()
                )
                for name in differentials
            },
            "skipped": dict(skipped),
            "deflated_sharpe": None if deflated is None else deflated.to_payload(),
            "deflated_sharpe_reason": deflated_reason,
            "deflated_sharpe_state": VERDICT_NOT_EVALUABLE if deflated is None else "evaluated",
            "pbo": None if pbo is None else pbo.to_payload(),
            "pbo_reason": pbo_reason,
            "pbo_state": VERDICT_NOT_EVALUABLE if pbo is None else "evaluated",
            "blocks_rule": (
                f"el mayor numero de bloques **par** que no pasa de {BLOCKS_MAX} ni de T; las "
                "sesiones que sobran del principio se recortan y se declaran en "
                "`trimmed_sessions` (A7)"
            ),
            "trimmed_sessions": None if pbo is None else int(common.sum()) - pbo.n_observations,
            "sessions_per_block": None if pbo is None else pbo.n_observations // pbo.blocks,
            "registry": registry.to_payload(),
            "registry_root": "propia raiz (`--trials-root`), nunca `runs/`",
        },
        "verdict": dict(verdict),
        "per_row_gaps_vs_reference": per_row_gaps,
        "not_a_pnl_note": NOT_A_PNL_NOTE,
        "report_does_not_do": [dict(item) for item in REPORT_DOES_NOT_DO],
        "limitations": list(LIMITATIONS),
        "notes": list(NOTES),
        "hash_format": REPORT_HASH_FORMAT,
        "report_hash_format": REPORT_HASH_FORMAT,
    }


def _fmt(value: object, *, digits: int = 6) -> str:
    """Numero con `digits` decimales, o `n/d` si no lo hay: nunca un cero que parezca un dato.

    Los valores de magnitud extrema (un QLIKE puede desbordarse cuando un filtro se va al suelo:
    ``5.9e281``) salen en notacion cientifica. El dato **no** se toca —el `.json` lleva el valor
    exacto—: lo que se evita es una celda de trescientos digitos en la tabla.
    """
    if value is None:
        return "n/d"
    if isinstance(value, bool):
        return "si" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/d"
        if value != 0.0 and (abs(value) >= 1e6 or abs(value) < 1e-4):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: SpecSearchReport) -> str:
    """Version legible del informe: catalogo, tabla, correccion y veredicto (A9)."""
    payload = report.payload
    correction = cast("Mapping[str, object]", payload["correction"])
    verdict = cast("Mapping[str, object]", payload["verdict"])
    sample = cast("Mapping[str, object]", payload["sample"])
    rows = cast("Mapping[str, Mapping[str, object]]", payload["rows"])
    registry = cast("Mapping[str, object]", correction["registry"])
    lines = [
        "# Busqueda de especificaciones del forecast de volatilidad (#58)",
        "",
        f"- fecha del informe: `{report.report_date.isoformat()}`",
        f"- `report_sha256`: `{report.report_sha256}`",
        f"- serie: `{payload['series_id']}`; *term structure*: "
        f"`{VIX9D_SERIES_ID}` y `{VIX3M_SERIES_ID}`",
        f"- muestra: la de #7 **sin cambios** ({sample.get('clean_sessions')} sesiones limpias "
        f"desde `{sample.get('clean_from')}`)",
        f"- walk-forward: ventana expansiva, `min_train = {MIN_TRAIN}`, reajuste cada "
        f"{REFIT_EVERY} sesiones, horizonte 1 (el de #7, importado)",
        "",
        "## Catalogo pre-registrado (escrito antes de medir)",
        "",
        "| especificacion | estimador | regresores | evaluable |",
        "|---|---|---|---|",
    ]
    for spec in SPECS:
        regressors = ", ".join(f"`{name}`" for name in spec.regressors) or "`ret_log`"
        evaluable = "si" if spec.evaluable else "**no**"
        lines.append(f"| `{spec.name}` | `{spec.estimator}` | {regressors} | {evaluable} |")
    lines.extend(
        ["", "Formulas exactas en `catalogue` del `.json`; ninguna busca hiperparametros.", ""]
    )
    for spec in SPECS:
        if not spec.evaluable:
            lines.append(
                f"- `{spec.name}`: **no evaluable** - {spec.reason} "
                f"(seguimiento `{spec.follow_up}`)"
            )
    lines.extend(
        [
            "",
            "## Tabla comparativa",
            "",
            "Brechas **relativas** de QLIKE: positivo = peor que la referencia "
            "(`(candidato - referencia) / |referencia|`). QLIKE en varianza, el de #7.",
            "",
            "| especificacion | QLIKE pri | QLIKE sec | MSE log pri | sigma med (bp) | "
            "vs `rw` pri | vs declarado pri | en correccion | Sharpe diferencial |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for name, row in rows.items():
        metrics = cast("Mapping[str, Mapping[str, object] | None]", row["metrics"])
        primary = metrics.get(Target.PARKINSON.value)
        secondary = metrics.get(Target.RET_SQ.value)
        lines.append(
            f"| `{name}` | {_fmt(None if primary is None else primary['qlike'])} | "
            f"{_fmt(None if secondary is None else secondary['qlike'])} | "
            f"{_fmt(None if primary is None else primary['mse_log'])} | "
            f"{_fmt(row['median_sigma_bp'], digits=1)} | "
            f"{_fmt(row['gap_vs_rw_primary'], digits=4)} | "
            f"{_fmt(row['gap_vs_declared_primary'], digits=4)} | "
            f"{'si' if row['in_correction'] else 'no'} | "
            f"{_fmt(row['differential_sharpe'], digits=4)} |"
        )
    return _markdown_tail(lines, correction=correction, verdict=verdict, registry=registry)


def _markdown_tail(
    lines: list[str],
    *,
    correction: Mapping[str, object],
    verdict: Mapping[str, object],
    registry: Mapping[str, object],
) -> str:
    """Cierra el Markdown: la correccion de #16, el veredicto, las limitaciones y las notas."""
    variants = ", ".join(f"`{name}`" for name in cast("Sequence[str]", correction["variant_ids"]))
    deflated = correction.get("deflated_sharpe")
    pbo = correction.get("pbo")
    lines.extend(
        [
            "",
            "## Correccion por comparaciones multiples (#16)",
            "",
            f"- variantes en la correccion: **{correction['n_variants']}** ({variants})",
            f"- sesiones comunes (`T`): **{correction['common_sessions']}**",
            f"- `n_trials` **derivado** del registro de ensayos: **{registry['n_trials']}** "
            f"(`registry_sha256`: `{registry['registry_sha256']}`)",
            "",
            "Unidades: " + CORRECTION_UNITS_NOTE,
            "",
        ]
    )
    if isinstance(deflated, Mapping):
        dsr = cast("Mapping[str, object]", deflated)
        lines.extend(
            [
                "| DSR | valor |",
                "|---|---|",
                f"| `dsr` | {_fmt(dsr['dsr'])} |",
                f"| `sr_observed` | {_fmt(dsr['sr_observed'])} |",
                f"| `sr0_expected_max` | {_fmt(dsr['sr0_expected_max'])} |",
                f"| `n_trials` | {dsr['n_trials']} |",
                f"| `sr_variance` | {_fmt(dsr['sr_variance'])} |",
                f"| `T` | {dsr['n_observations']} |",
                f"| asimetria / curtosis | {_fmt(dsr['skewness'])} / {_fmt(dsr['kurtosis'])} |",
                f"| **veredicto** | `{dsr['verdict']}` |",
            ]
        )
    else:
        lines.append(f"- **DSR no evaluable**: {correction.get('deflated_sharpe_reason')}")
    lines.append("")
    if isinstance(pbo, Mapping):
        backtest = cast("Mapping[str, object]", pbo)
        lines.extend(
            [
                "| PBO (CSCV) | valor |",
                "|---|---|",
                f"| `pbo` | {_fmt(backtest['pbo'])} |",
                f"| `blocks` | {backtest['blocks']} |",
                f"| `n_variants` | {backtest['n_variants']} |",
                f"| `n_combinations` | {backtest['n_combinations']} |",
                f"| `logit_median` | {_fmt(backtest['logit_median'])} |",
                f"| **veredicto** | `{backtest['verdict']}` |",
            ]
        )
    else:
        lines.append(f"- **PBO no evaluable**: {correction.get('pbo_reason')}")
    lines.extend(
        [
            "",
            "## Veredicto",
            "",
            f"- linea del diferencial: `{REFERENCE}` (la persistencia de #7, **no** un trial); a "
            f"quien hay que batir: `{DECLARED_CANDIDATE.candidate}` (§19.16)",
            f"- mejor especificacion nueva: `{verdict.get('best_new')}`",
            f"- las tres condiciones: `{verdict.get('conditions')}`",
            f"- **veredicto: `{verdict['verdict']}`** "
            f"(`improves = {'true' if verdict.get('improves') else 'false'}`)",
            "",
            f"{verdict.get('reason')}",
            "",
            NOT_A_PNL_NOTE,
            "",
            "## Limitaciones (declaradas, no escondidas)",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in LIMITATIONS)
    lines.extend(["", "## Notas", ""])
    lines.extend(f"- {item}" for item in NOTES)
    lines.extend(["", "## Lo que este informe no hace", ""])
    lines.extend(f"- {item['what']} → `{item['where']}`" for item in REPORT_DOES_NOT_DO)
    lines.append("")
    return "\n".join(lines)


def write_report(report: SpecSearchReport, directory: Path) -> tuple[Path, Path]:
    """Escribe el `.json` y el `.md` del informe en ``directory`` (A9)."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{REPORT_PREFIX}_{report.report_date.isoformat()}"
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    json_path.write_text(_json_text(report.published()), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def analyse(
    *,
    store: Store,
    reports_dir: Path,
    trials_root: Path,
    as_of: datetime,
    write: bool = True,
) -> SpecSearchReport:
    """Mide el catalogo pre-registrado, lo corrige por comparaciones multiples y escribe (A9).

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta lee el reloj.
    ``write=False`` no escribe **nada**, ni el informe ni la carpeta del ensayo. El candidato
    declarado de #7, su informe y ``runs/`` quedan intactos: el registro va a ``--trials-root``.
    """
    moment = _as_utc(as_of)
    frame, summary = build_sample(store)
    working = build_spec_frame(frame, term_structure=load_term_structure(store))
    bounds = garch_fold_bounds(working.height)
    if not bounds:
        raise SpecSearchError(
            f"no se puede estimar ninguna especificacion: {working.height} sesiones utilizables y "
            f"el minimo de entrenamiento es {MIN_TRAIN}"
        )
    evaluated = np.zeros(working.height, dtype=bool)
    evaluated[bounds[0][0] :] = True
    targets = {target.value: _column(working, TARGET_COLUMNS[target]) for target in TARGETS}
    primary = Target.PARKINSON.value

    # 1) Las cuatro filas de #7: su `walk_forward`, importado, sobre su propia muestra.
    study = walk_forward(frame)
    forecasts: dict[str, np.ndarray] = dict(study.forecasts)
    rows: list[CandidateResult] = list(study.candidates)

    # 2) Las nuevas, con el mismo esquema de folds y la misma metrica.
    reasons: dict[str, str | None] = {}
    for spec in SPECS:
        if spec.name in forecasts:
            continue
        forecast, reason = spec_forecasts(working, spec=spec, bounds=bounds)
        forecasts[spec.name], reasons[spec.name] = forecast, reason
    rows.extend(
        spec_row(
            spec=spec,
            forecast=forecasts[spec.name],
            reason=reasons.get(spec.name),
            targets=targets,
            evaluated=evaluated,
        )
        for spec in SPECS
    )

    # 3) Diferenciales de perdida frente a la persistencia y correccion por comparaciones (A5-A7).
    reference_losses = qlike_losses(forecasts[REFERENCE], targets[primary], mask=evaluated)
    differentials: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    for row in rows:
        if row.name == REFERENCE:
            continue
        if row.status != Status.OK.value or row.metrics.get(primary) is None:
            skipped[row.name] = row.reason or "sin metrica primaria: no entra en la correccion"
            continue
        differentials[row.name] = loss_differentials(
            reference_losses, qlike_losses(forecasts[row.name], targets[primary], mask=evaluated)
        )
    names, common, sharpes, skipped = correction_inputs(
        names=list(differentials),
        differentials=differentials,
        evaluated=evaluated,
        skipped=skipped,
    )
    records = tuple(
        register_spec(
            trials_root=trials_root,
            config=spec_config(
                spec=trial_spec(name),
                summary=summary,
                sessions=working.height,
                evaluated_sessions=int(evaluated.sum()),
                folds=len(bounds),
                reference=REFERENCE,
            ),
            sharpe_per_session=sharpes[name],
            n_observations=int(common.sum()),
            as_of=moment,
            write=write,
        )
        for name in names
    )
    registry = load_registry(trials_root, extra=records)
    best = best_new_spec(rows, names=[spec.name for spec in EVALUABLE_SPECS])
    if best is not None and best in sharpes:
        deflated, deflated_reason = deflated_block(
            selected=best, differentials=differentials, common=common, registry=registry
        )
    else:
        deflated, deflated_reason = (
            None,
            (
                "ninguna especificacion nueva tiene diferencial registrable: sin candidata que "
                "deflactar, el DSR queda `not_evaluable`"
            ),
        )
    pbo, pbo_reason = pbo_block(names=names, differentials=differentials, common=common)
    by_name = {row.name: row for row in rows}
    declared = DECLARED_CANDIDATE.candidate
    gaps = {
        target.value: (None if best is None else _gap(by_name[best], by_name.get(declared), target))
        for target in TARGETS
    }
    verdict = improvement_verdict(
        reference=declared, best_new=best, gaps=gaps, deflated=deflated, pbo=pbo
    )
    sessions_column = working.get_column("session")
    payload = _payload(
        moment=moment,
        summary=summary,
        sessions=working.height,
        evaluated_sessions=int(evaluated.sum()),
        first_evaluated=str(sessions_column[bounds[0][0]]),
        last_evaluated=str(sessions_column[working.height - 1]),
        folds=len(bounds),
        rows=rows,
        differentials=differentials,
        names=names,
        common=common,
        skipped=skipped,
        sharpes=sharpes,
        registry=registry,
        deflated=deflated,
        deflated_reason=deflated_reason,
        pbo=pbo,
        pbo_reason=pbo_reason,
        verdict=verdict,
    )
    report = SpecSearchReport(
        as_of=moment,
        report_date=moment.date(),
        payload=payload,
        report_sha256=_report_digest(payload),
        rows=tuple(rows),
        registry=registry,
        records=records,
    )
    if write:
        json_path, markdown_path = write_report(report, reports_dir)
        logger.info("busqueda de especificaciones: {} y {}", json_path, markdown_path)
    return report


def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado de la CLI: **obligatorio**, y nunca tomado del reloj."""
    if value is None:
        raise SpecSearchError(
            "falta --as-of: el modulo no lee el reloj y sin instante no se escribe nada"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SpecSearchError(f"--as-of no es un ISO-8601 valido: {error}") from error
    return _as_utc(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de la busqueda.

    Codigos de salida: ``0`` = informe emitido; ``2`` = no se emite nada (falta o no vale
    ``--as-of``, no hay muestra suficiente, el registro no se puede escribir): el motivo sale por
    ``stderr`` y **no** se escribe un numero inventado.
    """
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=(
            "Busqueda pre-registrada de especificaciones del forecast de volatilidad (#58): "
            "mide EGARCH, apARCH, HARQ, HAR con saltos, HAR con la term structure del VIX, "
            "ridge/lasso y cuantiles en el walk-forward de #7 y las corrige con el DSR y el PBO "
            "de #16"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raiz del almacen")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="directorio de informes (por defecto <data-root>/derived/reports)",
    )
    parser.add_argument(
        "--trials-root",
        type=Path,
        default=None,
        help=(
            "raiz del registro de ensayos (por defecto ./"
            f"{DEFAULT_TRIALS_ROOT}, nunca `runs/`: la cadena de modelos enumera todas sus "
            "entradas)"
        ),
    )
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument(
        "--as-of",
        default=None,
        help="instante declarado ISO-8601, obligatorio para escribir (el modulo no lee el reloj)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="mide y no escribe nada: ni el informe ni la carpeta del ensayo",
    )
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
        settings = load_settings(args.settings)
    except (SpecSearchError, ConfigurationError) as error:
        print(f"no se puede emitir la busqueda: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else data_root / "derived" / "reports"
    )
    trials_root = (
        Path(args.trials_root) if args.trials_root is not None else Path(DEFAULT_TRIALS_ROOT)
    )
    try:
        report = analyse(
            store=Store(data_root),
            reports_dir=reports_dir,
            trials_root=trials_root,
            as_of=moment,
            write=not bool(args.dry_run),
        )
    except (SpecSearchError, ExperimentLogError, ConfigurationError) as error:
        print(f"no se puede emitir la busqueda: {error}", file=sys.stderr)
        return 2

    verdict = cast("Mapping[str, object]", report.payload["verdict"])
    logger.info(
        "especificaciones: {} filas, {} variantes registradas, veredicto {} (improves={}); "
        "report_sha256 = {}",
        len(report.rows),
        report.registry.n_trials,
        verdict["verdict"],
        verdict["improves"],
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
