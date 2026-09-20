"""Features tecnicas de la sesion (`technical_v1`, tarea #20).

**Familia cerrada.** Este modulo entrega el calculo completo de ``technical_v1``:
las diez columnas declaradas en
:data:`cfdtrader.features.store.TECHNICAL_FEATURE_CATALOG` (tres retornos
multi-ventana, el ATR normalizado, la distancia a la media movil, el RSI de
Wilder, la posicion en el rango, la ruptura del recorrido y dos z-scores de
ventana expandida). No entrega ninguna otra: una columna fuera del catalogo no es
de esta familia.

**No reimplementa.** El recorrido real y el ATR normalizado se importan de
:mod:`cfdtrader.features.volatility` (#7), la **definicion unica** del proyecto, y
la normalizacion robusta de ventana expandida se importa de
:mod:`cfdtrader.features.store` (#19). Aqui no hay una segunda formula de ATR,
Parkinson, HAR ni VIX.

Sin *look-ahead*
----------------

Toda columna de la sesion ``t`` usa solo sesiones ``<= t``:

- Los retornos son ``ln(C_t / C_{t-k})`` con ``k >= 1``: miran al pasado.
- ``atr_norm`` de ``t`` es el ATR de ``t-14 ... t-1``; la sesion ``t`` no entra,
  asi que es la unica columna cuyo ``required_as_of`` es el cierre de ``t-1`` (a
  ``t0`` la sesion ``t`` todavia no ha abierto).
- Las ventanas rodantes (media, minimo, maximo, media del recorrido) exigen la
  ventana **completa** (``min_samples = window``): no hay ventanas parciales.
- Los dos ``_z`` son de ventana **expandida** (mediana y MAD de la historia
  ``<= t``), nunca de la muestra completa (`_docs/plan.md` §9).

La columna ``open`` **no se lee**: la familia tecnica se calcula con ``high``,
``low`` y ``close``. Es una propiedad, no un detalle: mutar el ``open`` de una
sesion no puede cambiar ninguna celda de esta matriz.

Lo no computable se publica ``null``
------------------------------------

Un ``NaN`` (un logaritmo de un precio no positivo) o un ``inf`` (un denominador
cero) se convierten en ``null`` antes de salir: un ``null`` es un dato ("aqui no
hay valor"), un ``NaN`` es un veneno que se propaga. La escritura lo verifica otra
vez: :func:`cfdtrader.features.store.daily_records` rechaza un ``NaN`` o un
``inf`` en cualquier columna del catalogo.

El modulo es **puro** (entra un ``pl.DataFrame``, sale otro), **no lee el reloj**
y no toca el sistema de ficheros: la persistencia la hace
:func:`cfdtrader.features.store.save_daily` con la spec de :func:`technical_spec`.
"""

from __future__ import annotations

import math
from typing import Final, cast

import polars as pl

from cfdtrader.features.store import (
    DEFAULT_TECHNICAL_SOURCES,
    DEFAULT_TECHNICAL_WINDOWS,
    RANGE_WINDOW,
    RETURN_LAGS,
    RSI_WINDOW,
    TECHNICAL_FEATURE_COLUMNS,
    TECHNICAL_FEATURE_SET,
    TECHNICAL_MIN_SESSIONS,
    FeatureSpec,
    InvalidFeatureMatrixError,
    normalise_expanding,
)
from cfdtrader.features.volatility import ATR_WINDOW, normalised_atr, true_range

__all__ = [
    "TECHNICAL_INPUT_COLUMNS",
    "technical_matrix",
    "technical_spec",
]

#: Columnas de entrada que necesita el calculo. **Sin** ``open``: la familia
#: tecnica no lo lee (A4). ``session`` es el ancla temporal, no una feature.
TECHNICAL_INPUT_COLUMNS: Final[tuple[str, ...]] = ("session", "high", "low", "close")

#: Nombre de la columna del recorrido real, tal y como la publica #7.
_TRUE_RANGE: Final[str] = "true_range"

#: Nombre de la columna del ATR normalizado y de la distancia a la media. Los dos
#: nombres coinciden con los del catalogo: son las claves de ``CatalogEntry.name``.
_ATR: Final[str] = "atr_norm"
_DIST_SMA: Final[str] = "dist_sma_20"
_RSI: Final[str] = "rsi_14"
_RANGE_POSITION: Final[str] = "range_pos_20"
_VOL_BREAK: Final[str] = "vol_break_20"


# ─────────────────────────────────────────────────────────────────────────────
# Formulas (todas forward: miran sesiones <= t)
# ─────────────────────────────────────────────────────────────────────────────
def _returns(lag: int) -> pl.Expr:
    """``ln(C_t / C_{t-lag})`` (A4). Solo lee ``close``."""
    return (pl.col("close") / pl.col("close").shift(lag)).log()


def _distance_to_sma(window: int) -> pl.Expr:
    """``C_t / media(C_{t-window+1} ... C_t) - 1`` (A6)."""
    average = pl.col("close").rolling_mean(window_size=window, min_samples=window)
    return pl.col("close") / average - 1.0


def _range_position(window: int) -> pl.Expr:
    """``(C_t - min(low)) / (max(high) - min(low))`` sobre la ventana cerrada en ``t`` (A8)."""
    lowest = pl.col("low").rolling_min(window_size=window, min_samples=window)
    highest = pl.col("high").rolling_max(window_size=window, min_samples=window)
    return (pl.col("close") - lowest) / (highest - lowest)


def _volatility_break(window: int) -> pl.Expr:
    """``TR_t / media(TR de t-window ... t-1)`` (A8): la ventana es **anterior** a ``t``."""
    average = pl.col(_TRUE_RANGE).shift(1).rolling_mean(window_size=window, min_samples=window)
    return pl.col(_TRUE_RANGE) / average


def _rsi_from_averages(average_gain: float, average_loss: float) -> float:
    """RSI a partir de las dos medias suavizadas (A7), con los tres bordes fijados.

    ``100 * gain / (gain + loss)`` es la misma cuenta que ``100 - 100 / (1 + RS)``,
    pero no revienta cuando el denominador es cero:

    - sin ganancias ni perdidas (serie plana) ⇒ ``50.0``;
    - sin perdidas ⇒ ``100.0``;
    - sin ganancias ⇒ ``0.0``.
    """
    total = average_gain + average_loss
    if total == 0.0:
        return 50.0
    if average_loss == 0.0:
        return 100.0
    if average_gain == 0.0:
        return 0.0
    return 100.0 * average_gain / total


def _wilder_rsi(values: list[float | None], *, window: int) -> list[float | None]:
    """RSI de Wilder (A7) sobre ``d_t = C_t - C_{t-1}``, en una pasada forward.

    ``alpha = 1 / window`` y semilla = media simple de las ``window`` primeras
    diferencias (las dos medias, de ganancias y de perdidas, se siembran con la
    misma historia). Las ``window`` primeras sesiones quedan a ``None``: sin
    ventana completa no hay semilla, y una semilla inventada seria un dato falso.

    Un hueco en los cierres rompe la racha: el estado se descarta y el RSI vuelve a
    sembrarse con las ``window`` diferencias siguientes. No se interpola ni se
    arrastra un suavizado contaminado.
    """
    result: list[float | None] = [None] * len(values)
    gains: list[float] = []
    losses: list[float] = []
    average_gain: float | None = None
    average_loss: float | None = None
    previous: float | None = None
    for index, value in enumerate(values):
        if value is None or previous is None:
            gains.clear()
            losses.clear()
            average_gain = None
            average_loss = None
            previous = value
            continue
        delta = value - previous
        previous = value
        if average_gain is None or average_loss is None:
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))
            if len(gains) < window:
                continue
            average_gain = math.fsum(gains) / window
            average_loss = math.fsum(losses) / window
        else:
            average_gain += (max(delta, 0.0) - average_gain) / window
            average_loss += (max(-delta, 0.0) - average_loss) / window
        result[index] = _rsi_from_averages(average_gain, average_loss)
    return result


def _numeric_values(frame: pl.DataFrame, *, column: str) -> list[float | None]:
    """Columna del frame como ``float`` con ``None`` en los huecos.

    Un valor no numerico es un error tipado, y se comprueba **antes** de construir
    cualquier expresion: una columna de texto tiene que dar este error, no uno de
    polars. Un ``NaN`` o un ``inf`` es un hueco (``None``), porque "no computable"
    no puede convertirse en una resta.
    """
    values: list[float | None] = []
    for value in cast("list[object]", frame.get_column(column).to_list()):
        if value is None:
            values.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidFeatureMatrixError(
                f"la columna '{column}' no es numerica: {type(value).__name__}"
            )
        number = float(value)
        values.append(number if math.isfinite(number) else None)
    return values


def _finite_or_null(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> pl.DataFrame:
    """Convierte ``NaN``/``inf`` en ``null`` en las columnas que se le pidan.

    ``is_finite`` es propagadora de nulos, asi que un ``null`` de entrada sigue
    siendo ``null``: la conversion no inventa un valor donde no lo habia. Se aplica
    **antes** de normalizar (``normalise_expanding`` rechaza los valores no
    finitos) y **despues** de calcular todo el catalogo.
    """
    return frame.with_columns(
        *(
            pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
            for name in columns
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Spec y matriz
# ─────────────────────────────────────────────────────────────────────────────
def technical_spec() -> FeatureSpec:
    """Spec por defecto de ``technical_v1``: el catalogo entero y una sola fuente.

    Los valores por defecto de :class:`FeatureSpec` son los de ``volatility_v1``
    (familias distintas), asi que la spec tecnica se declara **explicitamente**:
    sus ventanas son las de su catalogo y su fuente es el indice, **sin** VIX.
    """
    return FeatureSpec(
        feature_set=TECHNICAL_FEATURE_SET,
        windows=DEFAULT_TECHNICAL_WINDOWS,
        sources=DEFAULT_TECHNICAL_SOURCES,
    )


def technical_matrix(frame: pl.DataFrame, *, spec: FeatureSpec) -> pl.DataFrame:
    """Matriz persistible de ``technical_v1``: ``session`` + las diez features (A3).

    Parameters
    ----------
    frame:
        Frame con ``session``, ``high``, ``low`` y ``close``. ``open`` **no** hace
        falta: esta familia no lo lee.
    spec:
        Spec de la familia tecnica. Tiene que declarar ``technical_v1``: la matriz
        de volatilidad la construye :func:`cfdtrader.features.store.build_matrix`.

    Returns
    -------
    pl.DataFrame
        ``session`` mas una columna por entrada del catalogo, **en el orden del
        catalogo**, ordenado por sesion y con lo no computable a ``null``.
    """
    if spec.feature_set != TECHNICAL_FEATURE_SET:
        raise InvalidFeatureMatrixError(
            f"'technical_matrix' es la entrada de '{TECHNICAL_FEATURE_SET}': la familia "
            f"'{spec.feature_set}' tiene su propia funcion de calculo"
        )
    missing = [name for name in TECHNICAL_INPUT_COLUMNS if name not in frame.columns]
    if missing:
        raise InvalidFeatureMatrixError(
            f"faltan columnas de entrada para calcular la familia tecnica: {missing}"
        )

    ordered = frame.sort("session")
    # La validacion de tipos va **antes** de cualquier aritmetica: una columna de
    # texto tiene que dar un error tipado, no un fallo de polars a mitad del calculo.
    closes = _numeric_values(ordered, column="close")
    for name in ("high", "low"):
        _numeric_values(ordered, column=name)

    # El recorrido real y el ATR normalizado son de #7: se importan, no se copian.
    computed = normalised_atr(true_range(ordered), window=ATR_WINDOW)
    computed = computed.with_columns(
        *(_returns(lag).alias(name) for name, lag in RETURN_LAGS),
        _distance_to_sma(RANGE_WINDOW).alias(_DIST_SMA),
        _range_position(RANGE_WINDOW).alias(_RANGE_POSITION),
        _volatility_break(RANGE_WINDOW).alias(_VOL_BREAK),
    )
    # El RSI es recursivo: no cabe en una expresion de polars y se calcula en numpy.
    computed = computed.with_columns(
        pl.Series(_RSI, _wilder_rsi(closes, window=RSI_WINDOW), dtype=pl.Float64)
    )
    # Un `NaN`/`inf` se limpia **antes** de normalizar: `normalise_expanding` rechaza
    # los valores no finitos, y un denominador cero no puede tumbar la matriz entera.
    computed = _finite_or_null(computed, columns=(_ATR, _DIST_SMA))
    # Las dos `_z` son de #19: ventana expandida con el minimo declarado, importada.
    for column in (_ATR, _DIST_SMA):
        computed = normalise_expanding(computed, column, min_sessions=TECHNICAL_MIN_SESSIONS)

    computed = _finite_or_null(computed, columns=TECHNICAL_FEATURE_COLUMNS)
    return computed.select(["session", *TECHNICAL_FEATURE_COLUMNS]).sort("session")
