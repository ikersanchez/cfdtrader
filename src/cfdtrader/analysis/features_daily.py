"""Productor real de ``derived.features_daily``: arma las cinco familias y las persiste (#73).

Antes de este modulo, **ninguna** familia de features llegaba al almacen del repositorio: los
modulos de calculo son puros y ``features.store.save_daily`` solo se llamaba desde la suite, en
``tmp_path``, asi que el artefacto «features versionadas» de #19-#23 no tenia evidencia en el
almacen. Esto **produce**; verificarlo contra un *golden* es #17.

Lo que este modulo **no** hace, y es la mitad de su contrato:

- **No lee el reloj.** ``--as-of`` y ``--fetched-at`` los pasa el llamante: ``fetched_at`` es una
  decision de auditoria (el instante en que se produjo la fila), no un detalle que el modulo
  pueda rellenar por su cuenta.
- **No publica el futuro.** Solo se escriben sesiones **anteriores** a la sesion de ``--as-of``:
  una sesion que no habia cerrado en ese instante no puede tener filas.
- **No decide ninguna regla de features.** Familias, catalogos, ventanas, alineamiento y
  normalizacion son de #19-#23; aqui se lee, se une el ``as_of`` del ancla y se escribe.
- **No escribe ``raw``.** ``derived.features_daily`` es recalculable, asi que la escritura va por
  ``Store.replace``: una revision nueva sustituye a la vigente y la anterior queda en disco.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Final

import polars as pl
from loguru import logger

from cfdtrader.analysis.feature_frame import (
    ANCHOR_SERIES,
    COLUMNS_BY_FAMILY,
    FAMILY_ORDER,
    FeatureFrameError,
    build_family_frames,
    family_spec,
)
from cfdtrader.data.calendar import MarketCalendar
from cfdtrader.data.store import StorageError, Store, WriteOutcome
from cfdtrader.features import store as feature_store

__all__ = [
    "DEFAULT_DATA_ROOT",
    "FeaturesDailyError",
    "InvalidInstantError",
    "main",
    "persist_family_frames",
    "session_of",
]

#: Raiz del almacen por defecto de la CLI, la misma que la del resto de comandos del runbook.
DEFAULT_DATA_ROOT: Final[Path] = Path("data")


class FeaturesDailyError(Exception):
    """Raiz de los errores del productor de features."""


class InvalidInstantError(FeaturesDailyError):
    """Un instante de la corrida no es admisible (sin zona horaria, o anterior al ``as_of``)."""


def _require_aware(value: datetime, *, field: str) -> datetime:
    """Un instante **con zona horaria**: una hora sin zona no es un instante (convencion)."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidInstantError(
            f"'{field}' tiene que llevar zona horaria: una hora sin zona no es un instante"
        )
    return value


def session_of(as_of: datetime) -> date:
    """La **sesion** de un instante: su dia en ``America/New_York``, nunca la fecha UTC.

    Una sesion a caballo del cambio de hora se duplicaria si se truncara en UTC: el mismo
    criterio que ``analysis.backtest_report`` y ``analysis.feature_frame``.
    """
    return MarketCalendar.to_et(_require_aware(as_of, field="as_of")).date()


def persist_family_frames(
    store: Store,
    *,
    fetched_at: datetime,
    sessions_before: date,
    series_id: str = ANCHOR_SERIES,
) -> dict[str, WriteOutcome]:
    """Escribe las **cinco** familias en ``derived.features_daily`` y dice que paso en cada una.

    ``sessions_before`` es la sesion de referencia: solo se persisten sesiones
    **estrictamente anteriores**, porque una sesion que no habia cerrado no puede tener features.
    ``fetched_at`` lo declara el llamante: el modulo no lee el reloj en ningun caso.
    """
    fetched = _require_aware(fetched_at, field="fetched_at")
    built = build_family_frames(store, series_id=series_id)
    selected = built.instants.filter(pl.col("session") < sessions_before)
    if selected.height == 0:
        raise InvalidInstantError(
            f"ninguna sesion de {series_id!r} es anterior a {sessions_before.isoformat()}: no hay "
            "nada que persistir. La sesion de referencia tiene que ser posterior a la primera "
            "sesion del diario"
        )
    outcomes: dict[str, WriteOutcome] = {}
    for family in FAMILY_ORDER:
        # Solo `session` + las columnas **de su catalogo**: la matriz de una familia trae tambien
        # las barras y las series de entrada (`open`, `vix_close`...), y `daily_records` rechaza
        # cualquier columna ajena al catalogo en vez de ignorarla.
        matrix = (
            built.frames[family]
            .select("session", *COLUMNS_BY_FAMILY[family])
            .join(selected, on="session", how="inner")
        )
        if matrix.height == 0:
            raise FeaturesDailyError(
                f"la familia {family!r} no comparte ninguna sesion con el ancla: persistirla "
                "escribiria cero filas y la familia desapareceria del dataset sin decirlo"
            )
        outcomes[family] = feature_store.save_daily(
            store,
            spec=family_spec(family),
            matrix=matrix,
            series_id=series_id,
            fetched_at=fetched,
        )
    return outcomes


def _aware_argument(value: str) -> datetime:
    """Convierte el argumento y **exige** zona horaria (``argparse`` no lanza el error tipado)."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"'{value}' no es un instante ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError(
            f"'{value}' no lleva zona horaria: una hora sin zona no es un instante"
        )
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfdtrader.analysis.features_daily",
        description=(
            "Arma las cinco familias de features y las persiste en derived.features_daily (#73)"
        ),
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="raiz del almacen"
    )
    parser.add_argument(
        "--as-of",
        type=_aware_argument,
        default=None,
        help="instante de referencia, ISO-8601 **con zona horaria** (obligatorio)",
    )
    parser.add_argument(
        "--fetched-at",
        type=_aware_argument,
        default=None,
        help="instante de la corrida, ISO-8601 con zona horaria (obligatorio: lo pasa el llamante)",
    )
    parser.add_argument("--series-id", default=ANCHOR_SERIES, help="serie ancla del estudio")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: exige ``--as-of`` y ``--fetched-at``, no lee el reloj y escribe el dataset.

    Codigos de salida: ``0`` cuando las cinco familias quedan escritas (o sin cambios) y ``2``
    cuando falta un argumento o la corrida no se puede hacer: en ese caso **no se escribe ningun
    fichero**.
    """
    args = _parser().parse_args(argv)
    if args.as_of is None or args.fetched_at is None:
        logger.error(
            "faltan --as-of y/o --fetched-at: los dos son obligatorios y el modulo no lee el reloj"
        )
        return 2
    store = Store(args.data_root)
    try:
        session = session_of(args.as_of)
        outcomes = persist_family_frames(
            store,
            fetched_at=args.fetched_at,
            sessions_before=session,
            series_id=args.series_id,
        )
    except (FeaturesDailyError, FeatureFrameError, StorageError) as error:
        logger.error(f"no se puede persistir las features: {error}")
        return 2
    for family in FAMILY_ORDER:
        logger.info(f"features: {family} -> {outcomes[family].value} ({session.isoformat()})")
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de la CLI
    sys.exit(main())
