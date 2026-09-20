"""Volatilidad de la sesión: ATR normalizado, volatilidad realizada, HAR y VIX (tarea #7).

**Este módulo es la definición ÚNICA de la familia de volatilidad del proyecto**
(`tech_stack.md` §4.6): el ATR normalizado y la volatilidad realizada (Parkinson)
no se reimplementan en ningún otro sitio. Las tareas #20 (features técnicas, que
también nombra el `atr_norm`) y #23 (features de régimen y volatilidad) deben
**importar de aquí**, no copiar las fórmulas.

Es un módulo **puro** (`plan.md` §9, `tech_stack.md` §4.6): entra un
``pl.DataFrame`` y sale otro. No importa el almacén, DuckDB, ``httpx`` ni toca el
sistema de ficheros. La lectura de datos y la clave de sesión en
``America/New_York`` son de la capa de análisis.

Objetivos
---------

En unidades **fracción²** en el cálculo (bp = ×10⁴, y su raíz, al informar):

- **Objetivo primario** — varianza realizada de la sesión por el estimador de
  Parkinson::

      rv_t = (ln(H_t / L_t))² / (4 · ln 2)

  Usa **solo** ``high`` y ``low`` de la sesión, así que es **inmune por
  construcción** al artefacto #52 (el ``open`` repetido del cierre anterior): no
  lo lee. Lo mismo vale para el ATR: su True Range usa ``H``, ``L`` y ``C``, y
  nunca ``O``.
- **Objetivo secundario** — cuadrado del retorno de sesión::

      r_t = ln(C_t / O_t)        ret_sq_t = r_t²

  Este **sí** lee el ``open`` y por tanto **no** es inmune a #52.

Sin *look-ahead*
----------------

La regla que manda es que **la feature de la sesión ``t`` usa solo sesiones
``< t``** (`plan.md` §2.3): a ``t0`` = 08:45 ET la sesión ``t`` todavía no ha
abierto. Por eso:

- ``atr_norm`` de la sesión ``t`` es la media del True Range de las ``n`` sesiones
  **anteriores** (``t-n … t-1``), normalizada por el cierre de la última sesión
  usada (``C_{t-1}``).
- Los regresores del ``HAR`` de la sesión ``t`` son retardos de ``rv`` de sesiones
  anteriores (t-1, t-2…t-5, t-6…t-22), sin ventanas parciales.
- La feature de ``VIX`` de la sesión ``t`` usa **solo** el cierre del VIX de la
  sesión ``t-1``, y sus estadísticas de normalización son de **ventana
  expandida** (mínimo 250 sesiones), nunca de la muestra completa (`plan.md` §9:
  normalizar sobre todo el conjunto es *look-ahead*).

Los **objetivos** (`parkinson_rv`, `ret_sq`) usan la sesión ``t`` misma: son lo
que se predice, no features.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import arch
import numpy as np
import polars as pl

__all__ = [
    "ATR_WINDOW",
    "GARCH_MIN_TRAIN",
    "GARCH_REFIT_EVERY",
    "HAR_LAG_DAILY",
    "HAR_LAG_MONTHLY",
    "HAR_LAG_WEEKLY",
    "HAR_MIN_TRAIN",
    "HAR_WARMUP",
    "PARKINSON_DENOMINATOR",
    "VIX_MIN_SESSIONS",
    "GarchFit",
    "HarCoefficients",
    "add_features",
    "fit_garch",
    "fit_log_har",
    "garch_fold_bounds",
    "garch_forecasts",
    "garch_one_step_forecast",
    "har_forecast",
    "har_regressors",
    "normalised_atr",
    "parkinson_variance",
    "session_returns",
    "true_range",
    "vix_features",
]

#: Ventana por defecto del ATR normalizado, en sesiones.
ATR_WINDOW: Final[int] = 14

#: Retardos exactos del HAR, en sesiones.
HAR_LAG_DAILY: Final[int] = 1
"""``t-1``: la última sesión."""

HAR_LAG_WEEKLY: Final[int] = 4
"""``t-2 … t-5``: la semana anterior, sin ventanas parciales."""

HAR_LAG_MONTHLY: Final[int] = 17
"""``t-6 … t-22``: el mes anterior, sin ventanas parciales."""

#: Sesiones de calentamiento del HAR: sin los tres retardos completos no hay regresores.
HAR_WARMUP: Final[int] = HAR_LAG_DAILY + HAR_LAG_WEEKLY + HAR_LAG_MONTHLY
"""22 sesiones: las ``HAR_WARMUP`` primeras de la serie quedan a ``NULL``."""

#: Mínimo de filas de entrenamiento para aceptar un ajuste HAR.
#:
#: Una regresión de 4 parámetros con una o dos filas es un ajuste degenerado
#: (rango insuficiente), y devolver un pronóstico de ahí sería inventar. Se exige
#: un mes de retardos completos de entrenamiento.
HAR_MIN_TRAIN: Final[int] = 22

#: Sesiones mínimas de historia para normalizar el VIX (ventana expandida).
VIX_MIN_SESSIONS: Final[int] = 250

#: Sesiones mínimas de entrenamiento del GARCH(1,1). Es el esquema que #7
#: pre-registró y con el que el GARCH resultó elegido: subirlo o bajarlo cambia
#: la serie de pronósticos, no solo su arranque.
GARCH_MIN_TRAIN: Final[int] = 500

#: Reajuste del GARCH cada N sesiones, el **mismo** para todos los candidatos de #7.
GARCH_REFIT_EVERY: Final[int] = 21

#: Denominador del estimador de Parkinson: ``4 · ln 2``.
PARKINSON_DENOMINATOR: Final[float] = 4.0 * math.log(2.0)

#: Límite superior del logaritmo de la varianza antes de que ``exp`` desborde.
#: ``exp(709.78)`` ya es ``inf`` en coma flotante de doble precisión.
_MAX_LOG_VARIANCE: Final[float] = 700.0

#: Columnas que necesita el módulo y columnas que produce.
_REQUIRED: Final[tuple[str, ...]] = ("session", "open", "high", "low", "close")


# ─────────────────────────────────────────────────────────────────────────────
# Estimadores por sesión
# ─────────────────────────────────────────────────────────────────────────────
def true_range(frame: pl.DataFrame) -> pl.DataFrame:
    """Añade ``true_range``: el recorrido real de cada sesión.

    ``TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)``. En la primera
    sesión no hay ``C_{t-1}``, así que el recorrido real es ``H_t - L_t`` (polars
    ignora los nulos de las dos expresiones absolutas). **No lee el ``open``.**
    """
    return frame.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ).alias("true_range")
    )


def normalised_atr(frame: pl.DataFrame, *, window: int = ATR_WINDOW) -> pl.DataFrame:
    """Añade ``atr_norm``: el ATR normalizado y adimensional de cada sesión.

    Media del True Range de las ``window`` sesiones **anteriores** (``t-window …
    t-1``) dividida por el cierre de la última sesión usada (``C_{t-1}``). La
    sesión ``t`` no entra: a ``t0`` todavía no ha abierto. Adimensional porque el
    precio se quita dividiendo por un precio. Las primeras ``window`` sesiones
    quedan a ``NULL`` (sin ventana parcial). **No lee el ``open``.**

    Requiere que el frame haya pasado por :func:`true_range`.
    """
    if window < 2:
        raise ValueError("la ventana del ATR debe ser al menos 2 sesiones")
    return frame.with_columns(
        (
            pl.col("true_range").shift(1).rolling_mean(window_size=window, min_samples=window)
            / pl.col("close").shift(1)
        ).alias("atr_norm")
    )


def parkinson_variance(frame: pl.DataFrame) -> pl.DataFrame:
    """Añade ``parkinson_rv``: la varianza realizada de Parkinson de cada sesión.

    ``rv_t = (ln(H_t / L_t))² / (4 · ln 2)``, en fracción². Si ``high == low`` vale
    **0** exactamente, no ``NaN``. **No lee el ``open``**: es inmune al artefacto
    #52 por construcción.
    """
    return frame.with_columns(
        ((pl.col("high") / pl.col("low")).log().pow(2) / PARKINSON_DENOMINATOR).alias(
            "parkinson_rv"
        )
    )


def session_returns(frame: pl.DataFrame) -> pl.DataFrame:
    """Añade ``ret_log`` (``r_t = ln(C_t/O_t)``) y ``ret_sq`` (``r_t²``).

    El objetivo secundario. **Sí** lee el ``open``, así que **no** es inmune al
    artefacto #52: sobre muestras anteriores al corte limpio está contaminado.
    """
    return frame.with_columns(
        (pl.col("close") / pl.col("open")).log().alias("ret_log"),
    ).with_columns(pl.col("ret_log").pow(2).alias("ret_sq"))


def har_regressors(frame: pl.DataFrame, *, rv_column: str = "parkinson_rv") -> pl.DataFrame:
    """Añade los tres regresores del HAR, sin ventanas parciales.

    - ``har_lag1`` — ``rv`` de la sesión ``t-1`` (``HAR_LAG_DAILY``).
    - ``har_lag4`` — media de ``rv`` de ``t-2 … t-5`` (``HAR_LAG_WEEKLY``).
    - ``har_lag17`` — media de ``rv`` de ``t-6 … t-22`` (``HAR_LAG_MONTHLY``).

    El regresor del HAR **solo está definido cuando la ventana mensual está
    completa**: los tres retardos se calculan con la misma historia, así que las
    ``HAR_WARMUP`` (22) primeras sesiones quedan a ``NULL`` en los tres —no a 0 ni
    interpoladas— y no hay ninguna fila con ventanas parciales.
    """
    lag1, lag4, lag17 = HAR_LAG_DAILY, HAR_LAG_WEEKLY, HAR_LAG_MONTHLY
    base = pl.col(rv_column)
    lag_daily = base.shift(lag1)
    lag_weekly = base.shift(lag1 + 1).rolling_mean(window_size=lag4, min_samples=lag4)
    lag_monthly = base.shift(lag1 + lag4 + 1).rolling_mean(window_size=lag17, min_samples=lag17)
    complete = lag_monthly.is_not_null()
    return frame.with_columns(
        pl.when(complete).then(lag_daily).otherwise(None).alias("har_lag1"),
        pl.when(complete).then(lag_weekly).otherwise(None).alias("har_lag4"),
        lag_monthly.alias("har_lag17"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# HAR: ajuste y pronóstico
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class HarCoefficients:
    """Coeficientes del ajuste en espacio logarítmico: ``ln(rv) ~ 1 + regresores``.

    Los tres primeros regresores son siempre los del HAR; ``har_vix`` añade las
    features de VIX, así que el número de pendientes no está fijado.
    """

    intercept: float
    slopes: tuple[float, ...]

    def predict(self, *features: float) -> float | None:
        """Varianza pronosticada (fracción²), siempre ``> 0``, o ``None`` si no es finita.

        Se predice ``ln(rv)`` y se deshace el logaritmo: así la varianza es
        **positiva por construcción** (nunca un valor negativo de una regresión
        lineal sobre el nivel). Si el exponente desborda o no es finito, no hay
        pronóstico: ``None``, nunca ``inf`` ni ``NaN``.
        """
        value = self.intercept + sum(
            slope * feature for slope, feature in zip(self.slopes, features, strict=True)
        )
        if not math.isfinite(value) or value > _MAX_LOG_VARIANCE:
            return None
        variance = math.exp(value)
        return variance if math.isfinite(variance) and variance > 0.0 else None


def fit_log_har(
    design: np.ndarray, log_variance: np.ndarray, *, min_train: int = HAR_MIN_TRAIN
) -> HarCoefficients | None:
    """Ajuste por mínimos cuadrados en ``ln(rv)`` con ``numpy``.

    ``design`` trae **una columna por regresor** —los tres retardos del HAR y, para
    el candidato ``har_vix``, las features de VIX— y **sin** columna de unos;
    ``log_variance`` trae el ``ln`` del objetivo. Devuelve ``None`` cuando el
    ajuste **no es posible**: menos de ``min_train`` filas, valores no finitos,
    rango degenerado (una columna constante o redundante) o coeficientes no
    finitos. Nunca lanza y nunca devuelve un ajuste inventado.

    No se usa ``statsmodels`` (no está aprobado): aquí no se pide inferencia, solo
    un pronóstico, y ``numpy.linalg.lstsq`` es exactamente eso.
    """
    if design.ndim != 2 or design.shape[1] < 1:
        return None
    if log_variance.ndim != 1 or design.shape[0] != log_variance.shape[0]:
        return None
    if log_variance.shape[0] < max(min_train, HAR_MIN_TRAIN):
        return None
    if not (np.isfinite(design).all() and np.isfinite(log_variance).all()):
        return None
    matrix = np.column_stack((np.ones(log_variance.shape[0]), design))
    solution, _residuals, rank, _singular = np.linalg.lstsq(matrix, log_variance, rcond=None)
    if int(rank) < matrix.shape[1]:
        return None
    values = [float(value) for value in solution]
    if not all(math.isfinite(value) for value in values):
        return None
    return HarCoefficients(intercept=values[0], slopes=tuple(values[1:]))


def har_forecast(frame: pl.DataFrame, *, min_train: int = HAR_MIN_TRAIN) -> pl.DataFrame:
    """Añade ``har_forecast``: pronóstico a un paso con reajuste en cada sesión.

    Ventana **expansiva**: la sesión ``t`` se pronostica ajustando el HAR con
    todas las sesiones ``< t`` que tengan regresores y objetivo, y aplicando el
    resultado a los regresores de ``t``. Es un pronóstico *point-in-time* puro
    (añadir sesiones posteriores no cambia el valor de ``t``). Queda a ``NULL``
    mientras no haya ``min_train`` filas de entrenamiento o cuando el ajuste no
    sea posible, nunca a 0, ``inf`` ni ``NaN``.

    Requiere que el frame haya pasado por :func:`har_regressors` y
    :func:`parkinson_variance`.
    """
    columns = ("har_lag1", "har_lag4", "har_lag17")
    regressors = np.column_stack(
        [np.asarray(frame.get_column(name).to_numpy(), dtype=float) for name in columns]
    )
    variance = np.asarray(frame.get_column("parkinson_rv").to_numpy(), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        target = np.where(variance > 0.0, np.log(variance), np.nan)

    forecasts = np.full(variance.shape, np.nan)
    usable = np.isfinite(regressors).all(axis=1) & np.isfinite(target)
    for index in range(target.shape[0]):
        if not usable[index]:
            continue
        train = np.flatnonzero(usable[:index])
        if train.size < max(min_train, HAR_MIN_TRAIN):
            continue
        fitted = fit_log_har(regressors[train], target[train])
        if fitted is None:
            continue
        predicted = fitted.predict(*(float(value) for value in regressors[index]))
        if predicted is not None:
            forecasts[index] = predicted
    return frame.with_columns(pl.Series("har_forecast", forecasts, nan_to_null=True))


# ─────────────────────────────────────────────────────────────────────────────
# GARCH(1,1): ajuste y pronóstico a un paso
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class GarchFit:
    """Parámetros ajustados de GARCH(1,1) y el estado de la varianza condicional.

    ``sigma2`` es la varianza condicional de la última sesión de entrenamiento y
    ``epsilon2`` el cuadrado de su residuo (media cero): con ellos, el pronóstico
    a un paso es ``omega + alpha * epsilon2 + beta * sigma2``. Todo en fracción².
    """

    omega: float
    alpha: float
    beta: float
    sigma2: float
    epsilon2: float


def fit_garch(returns: np.ndarray) -> GarchFit:
    """Ajusta GARCH(1,1) con ``arch``: media cero, errores normales, sin exógenas.

    ``rescale=False`` a propósito: los parámetros quedan en las unidades de la
    serie (fracciones), no en una escala interna, para que la recursión de la
    varianza condicional sea exacta. Un ajuste no finito o con varianza no
    admisible es un ``ValueError``: el llamante decide si eso es un fallo o un
    pronóstico que no existe.
    """
    model = arch.arch_model(
        returns, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    )
    result = cast("Any", model.fit(disp="off", show_warning=False))
    params = result.params
    fit = GarchFit(
        omega=float(params["omega"]),
        alpha=float(params["alpha[1]"]),
        beta=float(params["beta[1]"]),
        sigma2=float(result.conditional_volatility[-1]) ** 2,
        epsilon2=float(result.resid[-1]) ** 2,
    )
    if not all(math.isfinite(value) for value in (fit.omega, fit.alpha, fit.beta, fit.sigma2)):
        raise ValueError("el ajuste GARCH no ha dado parámetros finitos")
    if fit.omega <= 0.0 or fit.alpha < 0.0 or fit.beta < 0.0 or fit.sigma2 <= 0.0:
        raise ValueError("el ajuste GARCH no ha dado una varianza admisible")
    return fit


def garch_one_step_forecast(returns: np.ndarray) -> float:
    """Pronóstico de varianza **a un paso** de GARCH(1,1) ajustado con ``returns``.

    Con los parámetros ajustados sobre ``returns``, la varianza pronosticada para
    la sesión siguiente es ``sigma² = omega + alpha·eps²_T + beta·sigma²_T``, donde
    ``T`` es la última sesión de entrenamiento. Los retornos van **en fracciones**
    (``0,01`` = 1 %).

    Se expone aparte de :func:`garch_forecasts` para poder comprobar la
    especificación contra una serie de varianza conocida (#7, A8) sin montar el
    *walk-forward* completo.
    """
    fit = fit_garch(returns)
    return fit.omega + fit.alpha * fit.epsilon2 + fit.beta * fit.sigma2


def garch_forecasts(
    returns: np.ndarray, *, bounds: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, str | None]:
    """Pronósticos de varianza a un paso de GARCH(1,1) con reajuste en cada fold.

    Dentro del fold los parámetros están **congelados** (el reajuste es cada
    ``GARCH_REFIT_EVERY`` sesiones, igual que para los demás candidatos) y la
    varianza condicional se actualiza con la recursión de GARCH(1,1):
    ``sigma²_t = omega + alpha·eps²_{t-1} + beta·sigma²_{t-1}``. El valor que se
    guarda en ``t`` es esa recursión aplicada con la información de ``t-1``:
    exactamente el pronóstico a un paso con lo disponible antes de la sesión ``t``
    (incluida la primera sesión de cada fold, que si no quedaría un paso por
    detrás).

    Devuelve el vector de pronósticos (``NaN`` donde no hay) y, si algún fold no
    se puede estimar, el motivo: ``arch`` falla de muchas formas distintas y
    ninguna debe abortar el estudio ni publicarse como un número.
    """
    forecasts = np.full(returns.shape, np.nan)
    for start, end in bounds:
        try:
            fit = fit_garch(returns[:start])
        except Exception as error:  # `arch` falla de muchas formas distintas
            return forecasts, f"GARCH(1,1) no estimable: {type(error).__name__}: {error}"
        sigma2, epsilon2 = fit.sigma2, fit.epsilon2
        for index in range(start, end):
            sigma2 = fit.omega + fit.alpha * epsilon2 + fit.beta * sigma2
            forecasts[index] = sigma2
            epsilon2 = returns[index] ** 2
    return forecasts, None


def garch_fold_bounds(
    sessions: int, *, min_train: int = GARCH_MIN_TRAIN, refit_every: int = GARCH_REFIT_EVERY
) -> list[tuple[int, int]]:
    """Tramos de reajuste: ventana expansiva, reajuste cada ``refit_every`` sesiones.

    El primer tramo empieza en la posición ``min_train`` (las ``min_train``
    primeras sesiones son entrenamiento puro) y el último se recorta al final de
    la serie. Sin sesiones suficientes devuelve una lista vacía: quien decide si
    eso es un error es el llamante.
    """
    bounds: list[tuple[int, int]] = []
    start = min_train
    while start < sessions:
        end = min(start + refit_every, sessions)
        bounds.append((start, end))
        start = end
    return bounds


# ─────────────────────────────────────────────────────────────────────────────
# VIX como feature de régimen
# ─────────────────────────────────────────────────────────────────────────────
def vix_features(
    frame: pl.DataFrame, *, column: str = "vix_close", min_sessions: int = VIX_MIN_SESSIONS
) -> pl.DataFrame:
    """Añade ``vix_level``, ``vix_zscore`` y ``vix_percentile`` (solo sesiones ``< t``).

    La feature de la sesión ``t`` usa **solo** el cierre del VIX de la sesión
    ``t-1``: a ``t0`` = 08:45 ET la sesión ``t`` no ha abierto y su VIX no existe.
    Se entregan tres lecturas del mismo dato:

    - ``vix_level`` — el cierre del VIX de ``t-1``, en puntos del índice.
    - ``vix_zscore`` — ``(nivel - media) / desviación`` con **ventana expandida** de
      ``min_sessions`` sesiones como mínimo y ``ddof = 1``; las estadísticas se
      calculan sobre las sesiones ``< t`` (nunca sobre la muestra completa:
      `plan.md` §9).
    - ``vix_percentile`` — fracción de los niveles anteriores que no superan al
      actual; misma ventana expandida.

    Las primeras ``min_sessions`` sesiones quedan a ``NULL``. Si el frame no trae
    la columna del VIX no se añade nada.
    """
    if column not in frame.columns:
        return frame
    level = pl.col(column).shift(1)
    history = pl.col(column).shift(1)
    count = history.is_not_null().cum_sum()
    total = history.cum_sum()
    total_sq = history.pow(2).cum_sum()
    mean = total / count
    # Desviación típica muestral expandida: E[x²] - E[x]² con corrección de ddof=1.
    variance = (total_sq - count * mean.pow(2)) / (count - 1)
    std = pl.when(variance > 0.0).then(variance.sqrt()).otherwise(None)
    enough = count.shift(1) >= min_sessions

    frame = frame.with_columns(level.alias("vix_level"))
    frame = frame.with_columns(
        pl.when(enough)
        .then((level - mean.shift(1)) / std.shift(1))
        .otherwise(None)
        .alias("vix_zscore"),
        pl.Series(
            "vix_percentile", _expanding_percentile(frame, column, min_sessions), nan_to_null=True
        ),
    )
    return frame


def _expanding_percentile(frame: pl.DataFrame, column: str, min_sessions: int) -> np.ndarray:
    """Percentil del nivel de ``t-1`` dentro de los niveles anteriores a ``t``.

    Ventana expandida y mínimo de sesiones: mientras no haya historia suficiente el
    valor es ``NaN`` (que polars escribe como ``null``). Se calcula con ``numpy``
    porque es una comparación contra toda la historia anterior, no un cuantil
    rodante de ventana fija.
    """
    values: np.ndarray = np.asarray(frame.get_column(column).to_numpy(), dtype=float)
    shifted: np.ndarray = np.concatenate((np.array([np.nan]), values[:-1]))
    out = np.full(values.shape, np.nan)
    for index in range(shifted.shape[0]):
        current = float(shifted[index])
        if not math.isfinite(current):
            continue
        history = np.asarray(shifted[:index], dtype=float)
        history = history[np.isfinite(history)]
        if history.size < min_sessions:
            continue
        out[index] = float(np.mean(history <= current))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Composición
# ─────────────────────────────────────────────────────────────────────────────
def add_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Calcula todas las features de volatilidad sobre un frame en memoria.

    Entra un ``pl.DataFrame`` con ``session``, ``open``, ``high``, ``low``,
    ``close`` (y opcionalmente ``vix_close``) y sale el mismo frame ordenado por
    sesión con las columnas de :func:`true_range`, :func:`normalised_atr`,
    :func:`parkinson_variance`, :func:`session_returns`,
    :func:`har_regressors`, :func:`har_forecast` y, si hay VIX,
    :func:`vix_features`.
    """
    missing = [name for name in _REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError(f"faltan columnas en el frame: {', '.join(missing)}")
    ordered = frame.sort("session")
    return (
        ordered.pipe(true_range)
        .pipe(normalised_atr)
        .pipe(parkinson_variance)
        .pipe(session_returns)
        .pipe(har_regressors)
        .pipe(har_forecast)
        .pipe(vix_features)
    )
