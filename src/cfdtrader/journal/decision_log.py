"""Capa de persistencia del diario de decisiones (tarea #39).

El diario es el **dato irreversible** del sistema (``plan.md`` §19.1): cada dia que pasa, el
sistema que produjo la recomendacion deja de existir (el modelo se reentrena, las features
evolucionan, la cache se purga), asi que sin diario no hay auditoria ni atribucion. Este
modulo es el unico que **escribe** el diario, y lo hace con las cuatro garantias que pide
``plan.md`` §19 y ``tech_stack.md`` §12:

- **Append-only por identidad.** Un documento JSON por identidad bajo ``Journal(root)``:
  ``<root>/<tabla>/<identidad>.json``. Reescribir esa identidad con el **mismo** contenido es
  un no-op (``WriteOutcome.UNCHANGED``); con **otro** contenido lanza ``JournalRewriteError``
  y el fichero queda intacto: nunca una sobrescritura silenciosa.
- **Esquema cerrado.** ``TABLE_COLUMNS`` declara las columnas exactas de las 8 tablas
  (``journal.decisions`` de 24 columnas, en el orden de §12.5, y las otras 7 de §12.5/§12.6).
  Un payload con una clave de mas es error tipado: el diario no inventa columnas.
- **Determinista.** JSON canonico (``ensure_ascii=False``, ``sort_keys=True``, ``indent=2``,
  salto final) y digest autoconsistente ``doc_sha256 = "sha256:" + sha256(canonical_text(...))``
  (el canonico de #13, **importado** de ``backtest.engine``): la misma identidad escrita en dos
  procesos con ``PYTHONHASHSEED`` distinto produce bytes identicos.
- **Sin reloj y sin red.** La fecha entra por parametro (``trade_date``/``as_of``); el modulo
  no consulta el reloj, no importa ``yfinance``/``requests``/``urllib`` y no escribe fuera de
  la raiz configurable. El ``git_commit`` es un parametro explicito: no se lee de git aqui.

**Los cuatro estados de §19.2** (``recommendation``, ``no_recommendation_stale_data``,
``no_recommendation_data_quality``, ``error``) se registran con **el mismo conjunto de
columnas**: una sesion sin recomendacion no se omite ni se colapsa en ``NOTHING``
(``direction = null`` en los estados "no se", ``nothing``/``long``/``short`` en
``recommendation``). El quinto estado del gate (``no_recommendation_undecided``, §11 bis) **no**
es un valor del diario: el llamante lo mapea o no registra.

**Forma del documento.** El fichero es el payload mas una clave ``doc_sha256``; ``read_record``
devuelve el payload sin esa clave y **verifica** la autoconsistencia del digest. El modulo
**reutiliza por import** ``WriteOutcome`` (``data.store``, sin modificarlo), ``canonical_text``
y ``Direction`` (``backtest.engine``) y ``GateOutput``/``GateStatus`` (``decision.gate``); no
redefine ninguno de esos tipos.

**Fuera de alcance** (declarado en #39): cablear el diario en ``delivery/run_daily.py`` (#112),
la observabilidad de ``ops.*`` (#43), la retencion (#44) y los campos de Fase 3
(``prompt_hashes`` poblados, ``agent_signals``, ``llm_overlay`` aplicado).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

from cfdtrader.backtest.engine import Direction, canonical_text
from cfdtrader.data.store import WriteOutcome
from cfdtrader.decision.gate import GateOutput, GateStatus

__all__ = [
    "DECISION_STATUSES",
    "DIGEST_KEY",
    "DIRECTIONS",
    "JOURNAL_TABLES",
    "LLM_OVERLAYS",
    "OPS_TABLES",
    "SHA256_PREFIX",
    "TABLES",
    "TABLE_COLUMNS",
    "ClosedSchemaError",
    "DecisionLogError",
    "InvalidDirectionError",
    "InvalidStatusError",
    "Journal",
    "JournalIntegrityError",
    "JournalRewriteError",
    "MissingIdentityError",
    "RecordNotFoundError",
    "UnknownTableError",
    "build_decision",
    "counts_by_status",
    "identity_column",
    "main",
    "read_decision",
    "read_decisions",
    "read_record",
    "write_record",
]

#: Prefijo obligatorio de los digests del repositorio (detect-secrets: nunca un hex desnudo).
SHA256_PREFIX: Final[str] = "sha256:"

#: Clave del digest autoconsistente dentro del documento (no es una columna de ninguna tabla).
DIGEST_KEY: Final[str] = "doc_sha256"

#: Sangria del JSON canonico (decision 2 del grooming).
JSON_INDENT: Final[int] = 2

#: Los **cuatro** estados de ``plan.md`` §19.2. El 5o del gate no es un valor del diario.
DECISION_STATUSES: Final[tuple[str, ...]] = (
    GateStatus.RECOMMENDATION.value,
    GateStatus.NO_RECOMMENDATION_STALE_DATA.value,
    GateStatus.NO_RECOMMENDATION_DATA_QUALITY.value,
    GateStatus.ERROR.value,
)

#: Direcciones admitidas en ``journal.decisions`` (``null`` en los estados "no se").
DIRECTIONS: Final[tuple[str, ...]] = (
    Direction.LONG.value,
    Direction.SHORT.value,
    Direction.NOTHING.value,
)

#: Estados del overlay LLM de §12.5 (declarados; su aplicacion es Fase 3 #30-#38).
LLM_OVERLAYS: Final[tuple[str, ...]] = (
    "applied",
    "veto",
    "disabled_budget",
    "disabled_error",
    "disabled_timeout",
)

#: Las 5 tablas del diario (§12.5); su identidad es ``trade_date``.
JOURNAL_TABLES: Final[tuple[str, ...]] = (
    "decisions",
    "agent_signals",
    "trades",
    "overrides",
    "attribution",
)

#: Las 3 tablas operacionales (§12.6).
OPS_TABLES: Final[tuple[str, ...]] = ("run_log", "llm_calls", "backtest_runs")

#: Las 8 tablas declaradas por la tarea #39.
TABLES: Final[tuple[str, ...]] = (*JOURNAL_TABLES, *OPS_TABLES)

#: Columnas exactas de cada tabla (esquema **cerrado**). La primer columna es la identidad.
TABLE_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "decisions": (
        "trade_date",
        "as_of",
        "status",
        "features_version",
        "model_version",
        "prompt_hashes",
        "git_commit",
        "prob_up_raw",
        "prob_up_calibrated",
        "expected_move_pct",
        "cost_pct",
        "ev_net_pct",
        "direction",
        "stop_pct",
        "target_pct",
        "size_notional_eur",
        "size_fraction",
        "leverage_implied",
        "tier",
        "blocking_events",
        "bull_case",
        "bear_case",
        "llm_overlay",
        "report_text",
    ),
    "agent_signals": (
        "trade_date",
        "agent",
        "prob_up",
        "confidence",
        "veto",
        "veto_reason",
        "evidence",
    ),
    "trades": (
        "trade_date",
        "entry_price",
        "exit_price",
        "entry_time",
        "exit_time",
        "notional",
        "pnl_pct",
        "costs_pct",
        "exit_reason",
        "closed_by_close",
    ),
    "overrides": (
        "trade_date",
        "model_recommendation",
        "human_action",
        "reason",
        "declared_confidence",
    ),
    "attribution": ("trade_date", "agent", "would_have_won", "evidence"),
    "run_log": ("run_id", "as_of", "stage", "duration_ms", "ok", "error"),
    "llm_calls": (
        "call_id",
        "as_of",
        "provider",
        "model",
        "system_fingerprint",
        "purpose",
        "tokens_in",
        "tokens_out",
        "cache_hit",
        "cost_estimate",
        "latency_ms",
        "ok",
    ),
    "backtest_runs": ("run_sha256", "config", "metrics"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados
# ─────────────────────────────────────────────────────────────────────────────
class DecisionLogError(Exception):
    """Raiz de los errores del diario de decisiones."""


class UnknownTableError(DecisionLogError):
    """La tabla pedida no es una de las 8 declaradas por ``TABLE_COLUMNS``."""


class ClosedSchemaError(DecisionLogError):
    """El payload trae una clave que no es columna de la tabla: esquema cerrado, error tipado."""


class MissingIdentityError(DecisionLogError):
    """Falta la columna de identidad (o no es un texto utilizable): sin identidad no hay fila."""


class InvalidStatusError(DecisionLogError):
    """``status`` fuera de los cuatro de §19.2 (incluye el ``no_recommendation_undecided``)."""


class InvalidDirectionError(DecisionLogError):
    """Una direccion fuera de ``long``/``short``/``nothing``, o incoherente con el estado."""


class JournalRewriteError(DecisionLogError):
    """Reescritura de una identidad ya registrada con **otro** contenido: nunca se sobrescribe."""


class RecordNotFoundError(DecisionLogError):
    """Lectura de una identidad que no existe: no se inventa un registro vacio."""


class JournalIntegrityError(DecisionLogError):
    """El documento del diario no es un JSON valido o su ``doc_sha256`` no es autoconsistente."""


# ─────────────────────────────────────────────────────────────────────────────
# Normalizacion a tipos JSON puros
# ─────────────────────────────────────────────────────────────────────────────
def _plain_value(value: object) -> object:
    """Traduce un valor a tipos JSON puros (``Decimal`` exacto, ``date``/``datetime`` ISO).

    Es el mismo criterio que el canonico de #13 (``backtest.engine._plain``): el digest y el
    fichero se calculan sobre la **misma** forma, asi que no puede haber divergencia.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return {str(key): _plain_value(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast("list[object] | tuple[object, ...]", value)
        return [_plain_value(item) for item in sequence]
    raise DecisionLogError(
        "el diario solo admite tipos JSON (str/int/float/bool/None), Decimal, date/datetime y "
        f"secuencias; llego {type(value).__name__}"
    )


def _plain_record(payload: Mapping[str, object]) -> dict[str, object]:
    """El payload en tipos JSON puros, con las claves ya validadas como texto."""
    return {str(key): _plain_value(value) for key, value in payload.items()}


def _require_table(table: str) -> str:
    """Valida que la tabla exista y devuelve su nombre (para encadenar)."""
    if table not in TABLE_COLUMNS:
        raise UnknownTableError(
            f"tabla desconocida: {table!r}; se esperaba una de {list(TABLE_COLUMNS)}"
        )
    return table


def identity_column(table: str) -> str:
    """La columna de identidad de la tabla: por convencion, la primera del esquema."""
    return TABLE_COLUMNS[_require_table(table)][0]


def _identity_text(value: object, column: str) -> str:
    """La identidad como texto seguro para un nombre de fichero.

    Acepta ``str`` no vacio, ``date`` y ``datetime`` (ISO). Rechaza separadores de ruta y
    ``.``/``..``: la identidad es parte de la ruta, no un valor libre.
    """
    if isinstance(value, (datetime, date)):
        text = value.isoformat()
    elif isinstance(value, str) and value:
        text = value
    else:
        raise MissingIdentityError(
            f"la columna de identidad {column!r} debe ser un texto no vacio, date o datetime; "
            f"llego {value!r}"
        )
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise MissingIdentityError(
            f"la identidad {text!r} no es un nombre de fichero seguro (columna {column!r})"
        )
    return text


def _document_digest(record: Mapping[str, object]) -> str:
    """``sha256:`` + sha256 del canonico de #13 del payload **sin** el digest."""
    return SHA256_PREFIX + hashlib.sha256(canonical_text(record).encode("utf-8")).hexdigest()


def _render_document(document: Mapping[str, object]) -> str:
    """JSON canonico (decision 2): ``ensure_ascii=False``, ``sort_keys=True``, sangria 2 y salto."""
    return json.dumps(dict(document), ensure_ascii=False, sort_keys=True, indent=JSON_INDENT) + "\n"


def _write_immutable(path: Path, text: str) -> WriteOutcome:
    """Escribe si no existe; contenido identico es no-op; otro contenido, error (patron #16)."""
    encoded = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() == encoded:
            return WriteOutcome.UNCHANGED
        raise JournalRewriteError(
            f"{path} ya existe con **otro** contenido: la identidad del diario es su contenido, "
            "asi que reescribirla con otro payload es error tipado, nunca una sobrescritura "
            "silenciosa"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return WriteOutcome.CREATED


# ─────────────────────────────────────────────────────────────────────────────
# Validacion de valores por tabla (dominios de §19.2)
# ─────────────────────────────────────────────────────────────────────────────
def _validate_record(table: str, record: Mapping[str, object]) -> None:
    """Valida los dominios declarados de ``decisions`` (el resto: solo esquema cerrado)."""
    if table != "decisions":
        return
    if "status" not in record:
        raise InvalidStatusError(
            "journal.decisions exige `status`: es el campo que distingue 'hoy no hay "
            "oportunidad' de 'no se' (§8.4)"
        )
    status = record["status"]
    if not isinstance(status, str) or status not in DECISION_STATUSES:
        raise InvalidStatusError(
            f"status invalido: {status!r}; se esperaba uno de {list(DECISION_STATUSES)} "
            "(el 5o estado del gate, `no_recommendation_undecided`, no es un valor del diario)"
        )
    direction = record.get("direction")
    if direction is not None and direction not in DIRECTIONS:
        raise InvalidDirectionError(
            f"direction invalida: {direction!r}; se esperaba None o una de {list(DIRECTIONS)}"
        )
    if status == GateStatus.RECOMMENDATION.value:
        if direction is None:
            raise InvalidDirectionError(
                "el estado `recommendation` exige `direction` (nothing/long/short): 'hoy no "
                "veo oportunidad' es una recomendacion, no un 'no se'"
            )
    elif direction is not None:
        raise InvalidDirectionError(
            f"el estado {status!r} es un 'no se': exige `direction = null`, llego {direction!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Raiz configurable del diario
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Journal:
    """Raiz configurable del diario: ``<root>/<tabla>/<identidad>.json``.

    La raiz es un parametro explicito (en pruebas, siempre ``tmp_path``): el modulo nunca
    escribe bajo el ``data/`` ni el ``runs/`` del repositorio.
    """

    root: Path = field()

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(str(self.root)))

    def directory(self, table: str) -> Path:
        """El directorio de la tabla (sin crearlo: escribir lo crea, leer solo lo consulta)."""
        return self.root / _require_table(table)

    def path(self, table: str, identity: str) -> Path:
        """La ruta del documento de esa identidad."""
        return self.directory(table) / f"{_identity_text(identity, identity_column(table))}.json"

    def write(self, table: str, payload: Mapping[str, object]) -> WriteOutcome:
        """Escribe (o confirma sin cambios) una fila; ver ``write_record``."""
        return write_record(self, table, payload)

    def read(self, table: str, identity: object) -> dict[str, object]:
        """Lee una fila por identidad; ver ``read_record``."""
        return read_record(self, table, identity)

    def read_decisions(self) -> list[dict[str, object]]:
        """Todas las decisiones, ordenadas por ``trade_date``."""
        return read_decisions(self)

    def read_decision(self, trade_date: date | str) -> dict[str, object]:
        """Una decision por su ``trade_date``."""
        return read_decision(self, trade_date)

    def counts_by_status(self) -> dict[str, int]:
        """Recuento de decisiones por estado (los cuatro, con ceros si no hay filas)."""
        return counts_by_status(self)


def _coerce_journal(journal: Journal | Path | str) -> Journal:
    """Acepta un ``Journal``, un ``Path`` o una ruta en texto (comodidad del llamante)."""
    return journal if isinstance(journal, Journal) else Journal(Path(journal))


# ─────────────────────────────────────────────────────────────────────────────
# Escritura y lectura de una fila
# ─────────────────────────────────────────────────────────────────────────────
def write_record(journal: Journal, table: str, payload: Mapping[str, object]) -> WriteOutcome:
    """Escribe una fila con esquema cerrado y digest autoconsistente.

    Pasos: la tabla debe existir; ninguna clave puede quedar fuera de ``TABLE_COLUMNS``; la
    columna de identidad debe venir; el payload se normaliza a JSON puro; se calcula el
    ``doc_sha256``; y se escribe de forma inmutable (``UNCHANGED`` si el contenido ya estaba).
    """
    columns = TABLE_COLUMNS[_require_table(table)]
    allowed = set(columns)
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise ClosedSchemaError(
            f"{table}: claves que no son columnas ({unknown}); el esquema es cerrado y se esperaba "
            f"un subconjunto de {list(columns)}"
        )
    identity_name = columns[0]
    if identity_name not in payload:
        raise MissingIdentityError(
            f"{table}: falta la columna de identidad {identity_name!r}; sin identidad no hay fila"
        )
    record = _plain_record(payload)
    _validate_record(table, record)
    identity = _identity_text(record[identity_name], identity_name)
    digest = _document_digest(record)
    document: dict[str, object] = {**record, DIGEST_KEY: digest}
    return _write_immutable(journal.path(table, identity), _render_document(document))


def _payload_from_document(document: object, path: Path) -> dict[str, object]:
    """Extrae el payload de un documento y **verifica** su ``doc_sha256``."""
    if not isinstance(document, dict):
        raise JournalIntegrityError(f"{path}: el documento no es un objeto JSON")
    raw = cast("dict[object, object]", document)
    record: dict[str, object] = {str(key): value for key, value in raw.items()}
    digest = record.pop(DIGEST_KEY, None)
    if not isinstance(digest, str):
        raise JournalIntegrityError(f"{path}: falta el digest {DIGEST_KEY!r}")
    expected = _document_digest(record)
    if digest != expected:
        raise JournalIntegrityError(
            f"{path}: el digest {digest!r} no es autoconsistente (esperado {expected!r})"
        )
    return record


def read_record(journal: Journal, table: str, identity: object) -> dict[str, object]:
    """Lee una fila (payload **sin** el digest) y verifica su autoconsistencia."""
    _require_table(table)
    path = journal.path(table, _identity_text(identity, identity_column(table)))
    if not path.exists():
        raise RecordNotFoundError(f"{table}: no existe la identidad {path.stem!r} ({path})")
    text = path.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise JournalIntegrityError(f"{path}: JSON invalido ({error})") from error
    return _payload_from_document(document, path)


def read_decisions(journal: Journal | Path | str) -> list[dict[str, object]]:
    """Todas las decisiones del diario, **ordenadas por ``trade_date``**."""
    root = _coerce_journal(journal)
    directory = root.directory("decisions")
    if not directory.exists():
        return []
    records = [
        read_record(root, "decisions", path.stem) for path in sorted(directory.glob("*.json"))
    ]
    records.sort(key=lambda record: str(record.get("trade_date", "")))
    return records


def read_decision(journal: Journal | Path | str, trade_date: date | str) -> dict[str, object]:
    """Una decision por ``trade_date`` (``RecordNotFoundError`` si no existe)."""
    return read_record(
        _coerce_journal(journal), "decisions", _identity_text(trade_date, "trade_date")
    )


def counts_by_status(journal: Journal | Path | str) -> dict[str, int]:
    """Recuento de decisiones por estado, con los cuatro estados siempre presentes."""
    records = read_decisions(journal)
    return {
        status: sum(1 for record in records if record.get("status") == status)
        for status in DECISION_STATUSES
    }


# ─────────────────────────────────────────────────────────────────────────────
# Constructor del payload de `journal.decisions`
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_status(status: GateStatus | str | None, output: GateOutput | None) -> str:
    """Normaliza y valida el ``status``; sin ``status``, lo toma del ``GateOutput``."""
    if status is None:
        if output is None:
            raise InvalidStatusError(
                "se requiere `status` cuando no hay `GateOutput` del que derivarlo"
            )
        raw = output.status.value
    else:
        raw = status.value if isinstance(status, GateStatus) else status
    if raw not in DECISION_STATUSES:
        raise InvalidStatusError(
            f"status invalido: {raw!r}; se esperaba uno de {list(DECISION_STATUSES)} "
            "(el 5o estado del gate, `no_recommendation_undecided`, no es un valor del diario)"
        )
    return raw


def _required_text(value: object, field_name: str) -> str:
    """Un parametro obligatorio, texto no vacio: nunca se inventa un valor."""
    if not isinstance(value, str) or not value:
        raise DecisionLogError(f"{field_name}: se esperaba un texto no vacio, llego {value!r}")
    return value


def _as_of_text(value: datetime | date | str) -> str:
    """``as_of`` como texto ISO: un ``datetime`` se serializa; un texto se persiste tal cual."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return _required_text(value, "as_of")


def _trade_date_text(value: date | datetime | str) -> str:
    """``trade_date`` normalizado a ``YYYY-MM-DD`` (identidad de la fila de diario)."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _required_text(value, "trade_date")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as error:
        raise DecisionLogError(f"trade_date no es una fecha ISO valida: {text!r}") from error


def _optional_float(value: Decimal | float | None) -> float | None:
    """Un numero opcional como ``float``; ``None`` sigue siendo ``None`` (nunca un 0)."""
    return None if value is None else float(value)


def build_decision(
    *,
    trade_date: date | str,
    as_of: datetime | date | str,
    features_version: str,
    model_version: str,
    git_commit: str,
    report_text: str,
    status: GateStatus | str | None = None,
    output: GateOutput | None = None,
    prob_up_raw: float | None = None,
    blocking_events: Sequence[str] | None = None,
    bull_case: Sequence[object] | None = None,
    bear_case: Sequence[object] | None = None,
    prompt_hashes: Mapping[str, object] | None = None,
    llm_overlay: str | None = None,
) -> dict[str, object]:
    """Construye un payload completo de ``journal.decisions`` con las 24 columnas.

    Deriva de ``output`` (el ``GateOutput`` de ``decision.gate``) lo que el gate ya calcula:
    direccion, probabilidad calibrada, movimiento esperado, coste, EV neto, stop, objetivo,
    apalancamiento implicito, tier y ``blocking_events`` (los ``blockers[].code``). Con
    ``output=None`` (estados "no se" y ``error``) esos campos salen ``null`` y
    ``blocking_events`` es la lista que aporte el llamante (vacia si no aporta ninguna).

    **Campos declarados ausentes, nunca inventados** (decision 3 del grooming): ``prompt_hashes``
    es ``{}`` hoy (los prompts son Fase 3), ``bull_case``/``bear_case`` son ``[]``,
    ``llm_overlay`` es ``None`` y ``size_notional_eur``/``size_fraction`` son ``None`` (no hay
    conversion FX offline: capital y nocional son USD). ``ev_net_pct`` es ``null`` (nunca ``0``)
    cuando el coste total es ``null``. ``report_text`` se persiste **verbatim**.
    """
    resolved = _resolve_status(status, output)
    resolved_features = _required_text(features_version, "features_version")
    resolved_model = _required_text(model_version, "model_version")
    resolved_commit = _required_text(git_commit, "git_commit")
    resolved_report = _required_text(report_text, "report_text")

    direction: str | None = None
    prob_up_calibrated: float | None = None
    expected_move_pct: float | None = None
    cost_pct: float | None = None
    ev_net_pct: float | None = None
    stop_pct: float | None = None
    target_pct: float | None = None
    leverage_implied: float | None = None
    tier: str | None = None
    derived_blockers: list[str] = []

    if output is not None:
        direction = None if output.direction is None else output.direction.value
        prob_up_calibrated = output.prob_up_calibrated
        expected_move_pct = float(output.expected_move_pct)
        cost_pct = float(output.cost_pct)
        # `ev_net_pct` es null (nunca 0) mientras el coste total no sea medible.
        ev_net_pct = None if output.cost_total_pct is None else _optional_float(output.ev_net_pct)
        stop_pct = _optional_float(output.stop_pct)
        target_pct = _optional_float(output.target_pct)
        leverage_implied = _optional_float(output.leverage_implied)
        tier = output.tier
        derived_blockers = [str(entry["code"]) for entry in output.blockers]
    elif blocking_events is not None:
        derived_blockers = [str(code) for code in blocking_events]

    payload: dict[str, object] = {
        "trade_date": _trade_date_text(trade_date),
        "as_of": _as_of_text(as_of),
        "status": resolved,
        "features_version": resolved_features,
        "model_version": resolved_model,
        "prompt_hashes": dict(prompt_hashes) if prompt_hashes is not None else {},
        "git_commit": resolved_commit,
        "prob_up_raw": _optional_float(prob_up_raw),
        "prob_up_calibrated": prob_up_calibrated,
        "expected_move_pct": expected_move_pct,
        "cost_pct": cost_pct,
        "ev_net_pct": ev_net_pct,
        "direction": direction,
        "stop_pct": stop_pct,
        "target_pct": target_pct,
        "size_notional_eur": None,
        "size_fraction": None,
        "leverage_implied": leverage_implied,
        "tier": tier,
        "blocking_events": derived_blockers,
        "bull_case": list(bull_case) if bull_case is not None else [],
        "bear_case": list(bear_case) if bear_case is not None else [],
        "llm_overlay": llm_overlay,
        "report_text": resolved_report,
    }
    _validate_record("decisions", payload)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# CLI de solo lectura: recuento por estado
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Sequence[str] | None = None) -> int:
    """CLI de **solo lectura**: imprime el recuento por ``status`` sin tocar el diario."""
    parser = argparse.ArgumentParser(
        prog="cfdtrader.journal.decision_log",
        description=(
            "Recuento por estado del diario de decisiones (solo lectura; nunca escribe ni purga)"
        ),
    )
    parser.add_argument("--root", required=True, help="raiz configurable del diario")
    args = parser.parse_args(argv)

    counts = counts_by_status(Journal(Path(args.root)))
    for status in DECISION_STATUSES:
        print(f"{status}: {counts[status]}")
    print(f"total: {sum(counts.values())}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
