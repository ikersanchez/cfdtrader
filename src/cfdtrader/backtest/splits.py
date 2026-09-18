"""Particiones *walk-forward* con purga y embargo — tarea #12.

Este modulo es el **unico** sitio del proyecto donde se decide que sesiones entran en el
*train* y cuales en el *test*. Lo **consumen** #13 (motor *walk-forward*), #16 (PBO y
Deflated Sharpe), #24 (entrenamiento) y #25 (calibracion); ninguno reimplementa la logica
(``tech_stack.md`` §4.7, fila «Purga y embargo»: «Implementacion propia (~50 lineas)»;
``mlfinlab`` esta prohibido). ``plan.md`` §2 (principio 5) exige que nada llegue a
produccion sin *backtest* *walk-forward* con purga y embargo.

Unidad: **sesiones**, nunca dias de calendario
----------------------------------------------

Los indices que devuelve ``SplitPlan`` son **posiciones dentro de la secuencia de sesiones
que se le pasa**. El modulo **no** hace aritmetica de calendario: no importa el
``MarketCalendar``, no mira si hay un festivo en medio y no usa ``datetime.timedelta``. Un
puente de diez dias de calendario sigue contando como **una** sesion, porque ``plan.md``
§11.1 habla de purga y embargo **en sesiones**.

``t0`` y ``t1``: la etiqueta de una sesion
------------------------------------------

Para la muestra ``i``, ``t0`` es el **open de la subasta** de esa sesion (09:30 ET) y
``t1`` es el **cierre de la misma sesion**, el ``close_utc`` que publica #10
(``cfdtrader.models.labels``). El proyecto es intradia puro ``open`` -> ``close``, sin
*overnight* (``plan.md`` §12, regla 6); por eso ``t1`` cae **en la propia sesion** ``i`` y
el **horizonte de etiqueta** ``h[i]`` —las sesiones que la etiqueta ocupa despues de su
sesion— vale **0** en **todas** las muestras etiquetadas de #10. ``h[i]`` es un parametro
**por muestra**, jamas una constante: el modulo acepta cualquier ``h[i] >= 0`` y no
supone que valga 0.

El horizonte es 0 tambien en las **medias sesiones**: ``t1`` de una media sesion es su
cierre (13:00 ET, 42 barras de 5 minutos), no las 16:00 ET. Una media sesion es **una
posicion mas**: el modulo no asume ninguna duracion de sesion ni un ``t1`` constante (el
artefacto de #10 declara ``labelled_half_sessions = 21`` y sus folds no cambian por ello).

Purga y embargo: que son y por que son no-ops aqui
--------------------------------------------------

- **Purga** (``plan.md`` §11.1): del *train* del fold ``j`` se elimina la muestra ``i``
  cuando su intervalo de etiqueta ``[i, i + h[i]]`` toca el test, es decir cuando
  ``i + h[i] >= test_start_j``. La implementacion es **literal** (A9): no es «quitar los
  ultimos N dias» ni una constante.
- **Embargo**: **tras cada** bloque de test se excluyen las ``embargo_sessions`` sesiones
  siguientes (``test_stop_j ... test_stop_j + E - 1``, recortado a ``[0, n)``),
  literalmente lo que describe ``plan.md`` §11.1 («tras cada bloque de test»), que da como
  guia 1-5 % del tamano de la muestra **sin** fijar ningun valor por defecto. El embargo
  es un parametro obligatorio: no hay embargo inventado.

Con el **train estrictamente anterior** al test (invariante dura de A7) y el **horizonte
de #10** (``h = 0``), las dos exclusiones son **no-ops estructurales**: la purga solo
podria eliminar ``i`` si ``i >= test_start``, imposible en un train anterior, y las
sesiones embargadas caen **despues** del test (en bloques de test posteriores o al final
de la serie), nunca en el train. El modulo **no lo afirma: lo mide** y lo publica
(``purge_total``, ``embargo_total``, ``embargo_in_train_total`` y el flag
``exclusions_are_no_op``). Un esquema donde la purga y el embargo **si** hacen trabajo
real es **CPCV** (**#67**); este walk-forward no lo es.

Esto es una decision del propietario (2026-09-18): ``plan.md`` §11.1 se sigue **literal**
(el embargo va **despues** del test, no es un hueco antes del test) y **no** se anade un
``gap_sessions``. La duda queda declarada aqui y en #12.

Modos de train
--------------

``max_train_size = None`` da un train **expansivo** (todas las posiciones validas
anteriores). ``max_train_size = m`` da un train **rodante** (solo las ``m`` posiciones
validas anteriores mas recientes). Los dos modos estan cubiertos por los tests y su
resultado difiere donde debe.

``plan_sha256``
---------------

``SplitPlan.plan_sha256`` es un sha256 hexadecimal calculado **solo** sobre los parametros
declarados —``n_sessions``, ``n_splits``, ``test_size``, ``embargo_sessions``,
``max_train_size`` y ``label_horizon``— con el formato estable que documenta
``PLAN_HASH_FORMAT``: mismas entradas dan el mismo hash y cambiar cualquier parametro lo
cambia. No entra ninguna entrada dependiente de la hora de ejecucion.

Que **no** hace este modulo (fronteras declaradas)
--------------------------------------------------

- **No** es #13: no recorre sesiones, no pide *snapshots* *point-in-time*, no toca
  ``bid``/``ask``, no modela el *gap* de apertura y no simula precios. Solo entrega el
  plan de particiones.
- **No** calcula metricas netas (**#15**), PBO ni Deflated Sharpe (**#16**).
- **No** entrena ningun modelo (**#24**) ni calibra dentro del esquema de purga (**#25**).
- **No** reserva ni toca el *holdout* final intocable (**#68**): este modulo **no** aparta
  ninguna ventana reservada.
- **No** lee el ``Store``, ni el ``MarketCalendar``, ni ``config/``, ni ``data/``: es puro
  respecto al disco (A22), sin reloj y sin azar.
- **No** usa ``mlfinlab`` (prohibido: ``tech_stack.md`` §4.7 y §6) ni
  ``sklearn.model_selection.KFold`` / ``train_test_split`` aleatorio (``plan.md`` §11.1).

El nucleo solo usa **biblioteca estandar** (``dataclasses``, ``hashlib``, ``collections.abc``);
``numpy`` se acepta como **entrada** (cualquier ``Sequence`` ordenada), nunca como
dependencia del calculo.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

__all__ = [
    "FOLLOW_UPS",
    "PLAN_HASH_FORMAT",
    "SPLITS_DOES_NOT_DO",
    "Fold",
    "InsufficientSessionsError",
    "InvalidSplitParameterError",
    "SplitPlan",
    "SplitsError",
    "walk_forward_splits",
]

#: Formato estable del ``plan_sha256`` (A23). Se hashea un texto UTF-8 con **una linea**
#: por parametro declarado, en este orden, con la forma ``clave=valor``; ``max_train_size``
#: nulo se escribe ``none`` y ``label_horizon`` como enteros separados por comas.
PLAN_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de las lineas, separadas por salto de linea y en este orden: "
    "n_sessions=<int>, n_splits=<int>, test_size=<int>, embargo_sessions=<int>, "
    "max_train_size=<int|none>, label_horizon=<h0,h1,...>"
)

#: Que **no** hace el modulo, legible por maquina (A32). Cada frontera con su issue.
SPLITS_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_es_el_motor",
        "issue": "#13",
        "statement": (
            "no recorre sesiones, no pide snapshots point-in-time, no toca bid/ask, no "
            "modela el gap de apertura y no simula precios: solo entrega el plan"
        ),
    },
    {
        "id": "no_calcula_metricas_ni_pbo",
        "issue": "#16",
        "statement": (
            "no calcula metricas netas (#15), ni PBO, ni Deflated Sharpe: los consume de "
            "otra capa sobre las particiones"
        ),
    },
    {
        "id": "no_entrena_ni_calibra",
        "issue": "#24",
        "statement": (
            "no entrena ningun modelo ni calibra dentro del esquema de purga (#25): solo "
            "decide que posiciones van a cada train y cada test"
        ),
    },
    {
        "id": "no_reserva_holdout",
        "issue": "#68",
        "statement": (
            "no reserva ni toca el holdout final intocable: la politica del plan.md §11.4 "
            "es #68, no este modulo"
        ),
    },
    {
        "id": "no_usa_cpcv",
        "issue": "#67",
        "statement": (
            "no implementa CPCV (el esquema donde purga y embargo hacen trabajo real): es "
            "un walk-forward de bloques de test contiguos"
        ),
    },
    {
        "id": "no_lee_disco",
        "issue": "#12",
        "statement": (
            "no lee el Store, el MarketCalendar, config/ ni data/, y no consulta el reloj "
            "ni el azar: es una funcion pura de la secuencia de sesiones"
        ),
    },
)

#: Seguimientos abiertos que este modulo deja declarados (A31).
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#67",
        "topic": "CPCV (Combinatorial Purged Cross-Validation)",
        "why": "es el esquema alternativo donde la purga y el embargo hacen trabajo real",
    },
    {
        "issue": "#68",
        "topic": "reserva y politica del holdout final intocable",
        "why": "`plan.md` §11.4 y §21 pregunta 10; este modulo no aparta ninguna ventana",
    },
    {
        "issue": "#13",
        "topic": "motor walk-forward",
        "why": "consume el plan de particiones con un coste explicito por operacion",
    },
    {
        "issue": "#16",
        "topic": "PBO y Deflated Sharpe",
        "why": "se calculan sobre las particiones; no se calculan aqui",
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados (A1, A24, A25, A26)
# ─────────────────────────────────────────────────────────────────────────────
class SplitsError(Exception):
    """Raiz de los errores de las particiones *walk-forward*."""


class InvalidSplitParameterError(SplitsError):
    """Un parametro de la llamada no es admisible (A24): error tipado, nunca silencioso."""


class InsufficientSessionsError(SplitsError):
    """La serie no da para los bloques pedidos, o un train se queda vacio (A25, A26)."""


# ─────────────────────────────────────────────────────────────────────────────
# Fold y SplitPlan (A3): tuplas de int estandar, sin fechas ni aritmetica de calendario
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Fold:
    """Un bloque de test contiguo y el train que le corresponde.

    ``test_stop`` es **exclusivo**: el test es ``range(test_start, test_stop)``. ``train``,
    ``purged`` y ``embargoed`` son tuplas de posiciones (A3), disjuntas dos a dos.
    """

    index: int
    test_start: int
    test_stop: int
    train: tuple[int, ...]
    purged: tuple[int, ...]
    embargoed: tuple[int, ...]

    @property
    def test(self) -> tuple[int, ...]:
        """Posiciones del test: ``range(test_start, test_stop)`` (``test_stop`` exclusivo)."""
        return tuple(range(self.test_start, self.test_stop))


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Plan completo de particiones, reproducible byte a byte (A3, A18, A21, A23).

    ``inputs`` es el eco de los parametros declarados (``n_splits``, ``test_size``,
    ``embargo_sessions``, ``max_train_size``, ``label_horizon``, ``n_sessions``); no se
    muta. ``uncovered`` son las posiciones que no pertenecen a ningun test (el
    calentamiento inicial y el remanente de la division), nunca descartadas en silencio.

    ``exclusions_are_no_op`` es un resultado **medido**, no una afirmacion: es ``True``
    solo si ninguna de las dos exclusiones quito **ninguna muestra del train**
    (``purge_total == 0`` y ``embargo_in_train_total == 0``). Con el horizonte de #10
    (``h = 0``) es ``True`` y no debe presentarse como un filtro activo.
    """

    n_sessions: int
    inputs: dict[str, object]
    folds: tuple[Fold, ...]
    uncovered: tuple[int, ...]
    purge_total: int
    embargo_total: int
    embargo_in_train_total: int
    exclusions_are_no_op: bool
    plan_sha256: str


# ─────────────────────────────────────────────────────────────────────────────
# Validacion (A24) e invariantes (A7, A8, A19)
# ─────────────────────────────────────────────────────────────────────────────
def _validate(
    sessions: Sequence[date],
    *,
    label_horizon: Sequence[int],
    n_splits: int,
    test_size: int,
    embargo_sessions: int,
    max_train_size: int | None,
) -> None:
    """Rechaza con error tipado cualquier parametro inadmisible (A24)."""
    if n_splits <= 0:
        raise InvalidSplitParameterError(f"n_splits debe ser >= 1, no {n_splits} (A24)")
    if test_size <= 0:
        raise InvalidSplitParameterError(f"test_size debe ser >= 1, no {test_size} (A24)")
    if embargo_sessions < 0:
        raise InvalidSplitParameterError(
            f"embargo_sessions debe ser >= 0, no {embargo_sessions} (A24)"
        )
    if max_train_size is not None and max_train_size <= 0:
        raise InvalidSplitParameterError(
            f"max_train_size debe ser >= 1 o None, no {max_train_size} (A24)"
        )
    if len(label_horizon) != len(sessions):
        raise InvalidSplitParameterError(
            f"len(label_horizon) ({len(label_horizon)}) != len(sessions) ({len(sessions)}) (A24)"
        )
    negatives = [i for i, horizon in enumerate(label_horizon) if horizon < 0]
    if negatives:
        raise InvalidSplitParameterError(
            f"label_horizon no puede ser negativo; posiciones {negatives} (A24)"
        )
    for i in range(len(sessions) - 1):
        if sessions[i] >= sessions[i + 1]:
            raise InvalidSplitParameterError(
                f"las sesiones deben ser estrictamente crecientes y sin duplicados; "
                f"se rompe en la posicion {i + 1} (A24)"
            )


def _verify_fold(fold: Fold, *, label_horizon: Sequence[int]) -> None:
    """Comprueba las invariantes duras del fold (A7, A8, A19). Se ejecuta **siempre**."""
    if fold.train and max(fold.train) >= fold.test_start:
        raise SplitsError(
            f"invariante rota (A7): el train del fold {fold.index} contiene la posicion "
            f"{max(fold.train)} >= test_start={fold.test_start}"
        )
    train = set(fold.train)
    overlap = sorted(train.intersection(range(fold.test_start, fold.test_stop)))
    if overlap:
        raise SplitsError(
            f"invariante rota (A19): train y test se solapan en el fold {fold.index}: {overlap}"
        )
    embargo_overlap = sorted(train.intersection(fold.embargoed))
    if embargo_overlap:
        raise SplitsError(
            f"invariante rota (A19): el train del fold {fold.index} contiene sesiones "
            f"embargadas: {embargo_overlap}"
        )
    touching = [i for i in fold.train if i + label_horizon[i] >= fold.test_start]
    if touching:
        raise SplitsError(
            f"invariante rota (A19): la purga dejo en el train del fold {fold.index} "
            f"muestras cuyo t1 invade el test: {touching}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Hash reproducible de los parametros declarados (A23)
# ─────────────────────────────────────────────────────────────────────────────
def _plan_sha256(
    *,
    n_sessions: int,
    n_splits: int,
    test_size: int,
    embargo_sessions: int,
    max_train_size: int | None,
    label_horizon: Sequence[int],
) -> str:
    """``sha256`` del formato estable de ``PLAN_HASH_FORMAT``: solo los parametros."""
    horizon = ",".join(str(value) for value in label_horizon)
    max_train = "none" if max_train_size is None else str(max_train_size)
    payload = "\n".join(
        (
            f"n_sessions={n_sessions}",
            f"n_splits={n_splits}",
            f"test_size={test_size}",
            f"embargo_sessions={embargo_sessions}",
            f"max_train_size={max_train}",
            f"label_horizon={horizon}",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Nucleo: folds + purga + embargo (puro y determinista)
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_splits(
    sessions: Sequence[date],
    *,
    label_horizon: Sequence[int],
    n_splits: int,
    test_size: int,
    embargo_sessions: int,
    max_train_size: int | None = None,
) -> SplitPlan:
    """Genera el plan *walk-forward* con purga y embargo (A2).

    ``sessions`` es una secuencia **ordenada** de sesiones (``list``, ``tuple`` o
    ``numpy.ndarray``): solo importan su longitud y su orden, y se valida que sea
    estrictamente creciente y sin duplicados. ``label_horizon[i]`` es el horizonte de
    etiqueta **en sesiones** de la muestra ``i`` (``t1`` cae en ``i + h[i]``). El resto de
    parametros son *keyword-only*: ``n_splits`` bloques de test contiguos de ``test_size``,
    ``embargo_sessions`` sesiones embargadas tras cada bloque y ``max_train_size`` para el
    modo expansivo (``None``) o rodante. Todos son obligatorios salvo ``max_train_size``,
    cuyo valor por defecto ``None`` significa «expansivo»: **no** se inventa ningun tamano.
    """
    n = len(sessions)
    _validate(
        sessions,
        label_horizon=label_horizon,
        n_splits=n_splits,
        test_size=test_size,
        embargo_sessions=embargo_sessions,
        max_train_size=max_train_size,
    )
    if n_splits * test_size >= n:
        raise InsufficientSessionsError(
            f"n_splits * test_size ({n_splits} * {test_size} = {n_splits * test_size}) "
            f">= n_sessions ({n}): no quedan posiciones anteriores para el train (A25)"
        )

    folds: list[Fold] = []
    purge_total = 0
    embargo_total = 0
    embargo_in_train_total = 0

    for index in range(n_splits):
        test_start = n - (n_splits - index) * test_size
        test_stop = test_start + test_size
        candidates = range(test_start)  # train estrictamente anterior (A7)
        purged = tuple(i for i in candidates if i + label_horizon[i] >= test_start)  # A9
        embargoed = tuple(range(test_stop, min(n, test_stop + embargo_sessions)))  # A16
        excluded = set(purged).union(embargoed)
        valid = [i for i in candidates if i not in excluded]
        if max_train_size is not None:
            valid = valid[-max_train_size:]  # modo rodante (A20)
        train = tuple(valid)
        if not train:
            raise InsufficientSessionsError(
                f"el train del fold {index} queda vacio: test_start={test_start}, "
                f"test_size={test_size}, embargo_sessions={embargo_sessions}, "
                f"max_train_size={max_train_size} (A26)"
            )
        fold = Fold(
            index=index,
            test_start=test_start,
            test_stop=test_stop,
            train=train,
            purged=purged,
            embargoed=embargoed,
        )
        _verify_fold(fold, label_horizon=label_horizon)  # A8/A19: siempre, no solo en tests
        folds.append(fold)
        purge_total += len(purged)
        embargo_total += len(embargoed)
        embargo_in_train_total += len(set(embargoed).intersection(train))

    uncovered = tuple(range(n - n_splits * test_size))  # calentamiento inicial (A6)
    inputs: dict[str, object] = {
        "n_splits": n_splits,
        "test_size": test_size,
        "embargo_sessions": embargo_sessions,
        "max_train_size": max_train_size,
        "label_horizon": tuple(label_horizon),
        "n_sessions": n,
    }
    return SplitPlan(
        n_sessions=n,
        inputs=inputs,
        folds=tuple(folds),
        uncovered=uncovered,
        purge_total=purge_total,
        embargo_total=embargo_total,
        embargo_in_train_total=embargo_in_train_total,
        exclusions_are_no_op=purge_total == 0 and embargo_in_train_total == 0,
        plan_sha256=_plan_sha256(
            n_sessions=n,
            n_splits=n_splits,
            test_size=test_size,
            embargo_sessions=embargo_sessions,
            max_train_size=max_train_size,
            label_horizon=label_horizon,
        ),
    )
