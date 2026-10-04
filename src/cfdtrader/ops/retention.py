"""Retención e higiene de disco (`tech_stack.md` §6.3, §12.6, §12.7) — tarea #44.

La pregunta que responde este módulo: **¿cómo evitamos que el disco crezca sin que nadie lo note?**

Dos comandos, **manuales y a demanda** (no hay scheduler ni servicio que los dispare, §4.11):

- ``sizes`` (**mensual**): mide el tamaño recursivo de cada componente físico frente a su
  **presupuesto declarado** (§12.7) y **avisa** de los que se pasan. Escribe un informe JSON + MD.
- ``retention`` (**trimestral**): aplica la política de retención —``ops.run_log`` 90 días,
  ``ops.llm_cache`` 18 meses y binarios de modelos un trimestre— y **simula por defecto**:
  sin ``--apply`` no se borra nada.

Dos reglas que no se negocian:

- **Nada se borra sin ``--apply``.** La simulación es el modo por defecto; el borrado es explícito.
- **Un modelo de producción no se purga jamás.** Un ``runs/<run_sha256>/`` es producción si su
  ``run_sha256`` aparece como ``model_version`` en ``journal.decisions``; los demás binarios de
  modelos se purgan al trimestre.

Este módulo lee y borra ficheros, pero **no lee el reloj**: el instante entra por ``--as-of``
(declarado), de modo que el mismo ``as_of`` da el mismo plan y el mismo informe.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from cfdtrader.journal.decision_log import read_decisions

__all__ = [
    "BUDGETS",
    "COMPONENTS",
    "RETENTION_POLICIES",
    "ApplyOutcome",
    "Component",
    "ComponentSize",
    "ConfigurationError",
    "PlannedDeletion",
    "RetentionPlan",
    "RetentionPolicy",
    "Roots",
    "SizesReport",
    "apply_retention",
    "check_budgets",
    "component_sizes",
    "main",
    "plan_retention",
    "render_markdown",
]

#: Unidad declarada: 1 MB = 1024 · 1024 bytes (los presupuestos de §12.7 son órdenes de magnitud).
MB: Final[int] = 1024 * 1024


class ConfigurationError(Exception):
    """La configuración del comando no es válida (falta una raíz, un ``as-of`` sin zona...)."""


@dataclass(frozen=True, slots=True)
class Component:
    """Un componente físico del disco: dónde vive y cuánto puede ocupar (§12.7)."""

    name: str
    base: Literal["data", "journal", "runs"]
    relative: str
    budget_bytes: int
    excluded: tuple[str, ...] = ()


#: Los componentes que se miden. El presupuesto sale de los "tamaños estabilizados" de §12.7; la
#: caché HTTP se incluye aunque sea regenerable (se mide, pero no se le fija edad de purga).
COMPONENTS: Final[tuple[Component, ...]] = (
    Component("raw", "data", "raw", 256 * MB),
    Component("derived", "data", "derived", 64 * MB),
    Component("http_cache", "data", "cache", 64 * MB),
    Component("journal", "journal", ".", 32 * MB, excluded=("ops",)),
    Component("ops_run_log", "journal", "ops", 50 * MB, excluded=("llm_cache",)),
    Component("ops_llm_cache", "journal", "ops/llm_cache", 250 * MB),
    Component("runs", "runs", ".", 512 * MB),
)

#: Presupuesto declarado por nombre de componente, en bytes.
BUDGETS: Final[dict[str, int]] = {
    component.name: component.budget_bytes for component in COMPONENTS
}


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Una regla de retención: qué se purga y a partir de qué edad (``max_age_days``)."""

    name: str
    target: Literal["run_log", "llm_cache", "model_binaries"]
    max_age_days: int


#: Las tres retenciones de §12.6/§12.7. El `run_log` es de 90 días; la caché del LLM, de 18 meses
#: (~540 días, el extremo conservador del rango 12–18); los binarios de modelos, de un trimestre.
RETENTION_POLICIES: Final[tuple[RetentionPolicy, ...]] = (
    RetentionPolicy("run_log", "run_log", 90),
    RetentionPolicy("llm_cache", "llm_cache", 540),
    RetentionPolicy("model_binaries", "model_binaries", 90),
)


@dataclass(frozen=True, slots=True)
class Roots:
    """Las tres raíces sobre las que trabaja el comando (el almacén, el diario y los runs)."""

    data: Path
    journal: Path
    runs: Path

    def base(self, name: str) -> Path:
        """La ruta de una raíz por su nombre (``data``/``journal``/``runs``)."""
        mapping = {"data": self.data, "journal": self.journal, "runs": self.runs}
        if name not in mapping:
            raise ConfigurationError(f"raiz desconocida: {name!r}")
        return mapping[name]

    def path_of(self, component: Component) -> Path:
        """La ruta absoluta de un componente."""
        return self.base(component.base) / component.relative


class ComponentSize(BaseModel):
    """El tamaño de un componente y su presupuesto, con la marca de exceso."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component: str = Field(min_length=1)
    path: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    budget_bytes: int = Field(ge=0)
    over_budget: bool


class SizesReport(BaseModel):
    """El informe del comando ``sizes``: cada componente frente a su presupuesto."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["sizes"] = "sizes"
    as_of: str
    components: tuple[ComponentSize, ...]
    total_bytes: int = Field(ge=0)
    total_budget_bytes: int = Field(ge=0)
    over_budget: tuple[str, ...] = ()


class PlannedDeletion(BaseModel):
    """Un borrado que la política marca: qué, por qué política y con qué edad."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: str = Field(min_length=1)
    path: str = Field(min_length=1)
    age_days: int = Field(ge=0)
    reason: str = Field(min_length=1)


class RetentionPlan(BaseModel):
    """El plan del comando ``retention``: lo que **se borraría** (o se borra, con ``--apply``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["retention"] = "retention"
    as_of: str
    apply: bool
    deletions: tuple[PlannedDeletion, ...] = ()
    protected_runs: tuple[str, ...] = ()


class ApplyOutcome(BaseModel):
    """El resultado de aplicar un plan: qué se borró y qué falló."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deleted_paths: tuple[str, ...] = ()
    deleted_bytes: int = Field(default=0, ge=0)
    failures: tuple[str, ...] = ()


def _aware(value: datetime) -> datetime:
    """El instante declarado en UTC; sin zona es error tipado (nada de reloj implícito)."""
    if value.utcoffset() is None:
        raise ConfigurationError("as_of: se espera un datetime con zona (TZ-aware)")
    return value.astimezone(UTC)


def _dir_size(path: Path, excluded: frozenset[str] = frozenset()) -> int:
    """Tamaño recursivo de ``path`` en bytes, excluyendo los subdirectorios nombrados."""
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if not child.is_file() or not excluded.isdisjoint(child.relative_to(path).parts):
            continue
        total += child.stat().st_size
    return total


def component_sizes(roots: Roots) -> tuple[ComponentSize, ...]:
    """Mide cada componente del disco frente a su presupuesto declarado."""
    rows: list[ComponentSize] = []
    for component in COMPONENTS:
        path = roots.path_of(component)
        size = _dir_size(path, frozenset(component.excluded))
        rows.append(
            ComponentSize(
                component=component.name,
                path=str(path),
                bytes=size,
                budget_bytes=component.budget_bytes,
                over_budget=size > component.budget_bytes,
            )
        )
    return tuple(rows)


def check_budgets(roots: Roots, *, as_of: datetime) -> SizesReport:
    """El informe mensual: tamaños reales frente a los presupuestados, con los que se pasan."""
    moment = _aware(as_of)
    rows = component_sizes(roots)
    return SizesReport(
        as_of=moment.isoformat(),
        components=rows,
        total_bytes=sum(row.bytes for row in rows),
        total_budget_bytes=sum(row.budget_bytes for row in rows),
        over_budget=tuple(row.component for row in rows if row.over_budget),
    )


def _human_bytes(size: int) -> str:
    """Un tamaño legible: ``14.0 MB``."""
    amount = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if amount < 1024:
            return f"{int(amount)} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"


def _render_sizes(report: SizesReport) -> str:
    over = (
        ", ".join(f"`{name}`" for name in report.over_budget) if report.over_budget else "ninguno"
    )
    lines = [
        "# Higiene de disco: tamanos frente a presupuesto (tarea #44)",
        "",
        "- **Modo:** `sizes` (job mensual, manual)",
        f"- **as_of:** `{report.as_of}`",
        f"- **Componentes por encima de presupuesto:** {over}",
        "",
        "| Componente | Ruta | Tamano | Presupuesto | Estado |",
        "|---|---|---:|---:|---|",
    ]
    for row in report.components:
        state = "AVISO: por encima" if row.over_budget else "ok"
        lines.append(
            f"| `{row.component}` | `{row.path}` | {_human_bytes(row.bytes)} | "
            f"{_human_bytes(row.budget_bytes)} | {state} |"
        )
    lines.append(
        f"| **TOTAL** | | **{_human_bytes(report.total_bytes)}** | "
        f"**{_human_bytes(report.total_budget_bytes)}** | |"
    )
    lines.append("")
    lines.append(
        "> Comando **manual**: no hay scheduler ni servicio que lo dispare (`tech_stack.md` §4.11)."
    )
    return "\n".join(lines) + "\n"


def _render_retention(plan: RetentionPlan) -> str:
    state = "aplicado" if plan.apply else "simulacion (nada se borra sin `--apply`)"
    lines = [
        "# Retencion e higiene de disco (tarea #44)",
        "",
        "- **Modo:** `retention` (job trimestral, manual)",
        f"- **as_of:** `{plan.as_of}`",
        f"- **Estado:** {state}",
        f"- **Borrados planificados:** {len(plan.deletions)}",
        f"- **Runs de produccion protegidos:** {len(plan.protected_runs)}",
        "",
        "| Politica | Ruta | Edad (dias) | Motivo |",
        "|---|---|---:|---|",
    ]
    for deletion in plan.deletions:
        lines.append(
            f"| `{deletion.policy}` | `{deletion.path}` | {deletion.age_days} | {deletion.reason} |"
        )
    lines.append("")
    lines.append(
        "> Comando **manual**: no hay scheduler ni servicio que lo dispare (`tech_stack.md` §4.11)."
    )
    return "\n".join(lines) + "\n"


def render_markdown(report: SizesReport | RetentionPlan) -> str:
    """El informe en Markdown, sea de tamaños o de retención."""
    return _render_sizes(report) if isinstance(report, SizesReport) else _render_retention(report)


def _age_days(path: Path, moment: datetime) -> int:
    """Días transcurridos entre la última modificación de ``path`` y el instante declarado."""
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return max(0, (moment - modified).days)


def _production_runs(roots: Roots) -> frozenset[str]:
    """Los ``run_sha256`` que son **producción**: aparecen como ``model_version`` en el diario.

    Un modelo que generó decisiones en vivo no se purga jamás; el resto de binarios de ``runs/``
    son de experimentos y sí se acotan a un trimestre (§12.6).
    """
    production: set[str] = set()
    for decision in read_decisions(roots.journal):
        value = decision.get("model_version")
        if isinstance(value, str) and value.strip():
            production.add(value)
    return frozenset(production)


def _deletions_for(
    policy: RetentionPolicy, roots: Roots, moment: datetime, production: frozenset[str]
) -> list[PlannedDeletion]:
    """Las rutas que esa política marca para borrar (todas con su edad y su motivo)."""
    out: list[PlannedDeletion] = []
    if policy.target == "run_log":
        ops = roots.journal / "ops"
        candidates = (
            [
                session
                for session in sorted(ops.iterdir())
                if session.is_dir() and session.name != "llm_cache"
            ]
            if ops.is_dir()
            else []
        )
        for session in candidates:
            log = session / "run_log.jsonl"
            if not log.is_file():
                continue
            age = _age_days(log, moment)
            if age > policy.max_age_days:
                out.append(
                    PlannedDeletion(
                        policy=policy.name,
                        path=str(session),
                        age_days=age,
                        reason=f"run_log de {age} dias (> {policy.max_age_days})",
                    )
                )
    elif policy.target == "llm_cache":
        cache = roots.journal / "ops" / "llm_cache"
        if cache.is_dir():
            for entry in sorted(cache.rglob("*")):
                if not entry.is_file():
                    continue
                age = _age_days(entry, moment)
                if age > policy.max_age_days:
                    out.append(
                        PlannedDeletion(
                            policy=policy.name,
                            path=str(entry),
                            age_days=age,
                            reason=f"entrada de cache de {age} dias (> {policy.max_age_days})",
                        )
                    )
    elif roots.runs.is_dir():
        for run in sorted(roots.runs.iterdir()):
            model = run / "model.json"
            if not run.is_dir() or not model.is_file() or run.name in production:
                continue
            age = _age_days(model, moment)
            if age > policy.max_age_days:
                out.append(
                    PlannedDeletion(
                        policy=policy.name,
                        path=str(model),
                        age_days=age,
                        reason=f"binario no productivo de {age} dias (> {policy.max_age_days})",
                    )
                )
    return out


def plan_retention(roots: Roots, *, as_of: datetime, apply: bool = False) -> RetentionPlan:
    """El plan de retención: qué se borraría (o se borra, si ``apply``). **No** borra nada."""
    moment = _aware(as_of)
    production = _production_runs(roots)
    deletions: list[PlannedDeletion] = []
    for policy in RETENTION_POLICIES:
        deletions.extend(_deletions_for(policy, roots, moment, production))
    return RetentionPlan(
        as_of=moment.isoformat(),
        apply=apply,
        deletions=tuple(deletions),
        protected_runs=tuple(sorted(production)),
    )


def apply_retention(plan: RetentionPlan) -> ApplyOutcome:
    """Borra de verdad lo que el plan marca. Solo se llama con ``--apply``."""
    deleted: list[str] = []
    failures: list[str] = []
    total = 0
    for deletion in plan.deletions:
        path = Path(deletion.path)
        try:
            if path.is_dir():
                size = _dir_size(path)
                shutil.rmtree(path)
            elif path.is_file():
                size = path.stat().st_size
                path.unlink()
            else:
                continue
        except OSError as error:
            failures.append(f"{deletion.path}: {error}")
            continue
        deleted.append(deletion.path)
        total += size
    return ApplyOutcome(deleted_paths=tuple(deleted), deleted_bytes=total, failures=tuple(failures))


def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado, ISO-8601 y con zona; sin reloj interno."""
    if value is None or not value.strip():
        raise ConfigurationError("falta --as-of: el instante declarado (ISO-8601 con zona)")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as error:
        raise ConfigurationError(f"--as-of no es ISO-8601: {error}") from error
    return _aware(moment)


def _write_report(
    reports_dir: Path, stem: str, report: SizesReport | RetentionPlan, moment: datetime
) -> tuple[Path, Path]:
    """Escribe el informe JSON + MD en ``reports_dir``; devuelve sus rutas."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{moment.date().isoformat()}"
    json_path = reports_dir / f"{name}.json"
    md_path = reports_dir / f"{name}.md"
    payload = json.dumps(report.model_dump(), ensure_ascii=False, sort_keys=True, indent=2)
    json_path.write_text(payload + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfdtrader.ops.retention",
        description=(
            "Higiene de disco y retencion (tarea #44). Comandos manuales y a demanda: "
            "no hay scheduler ni servicio que los dispare (tech_stack.md §4.11)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sizes = sub.add_parser("sizes", help="mide cada componente frente a su presupuesto (mensual)")
    retention = sub.add_parser("retention", help="aplica la politica de retencion (trimestral)")
    for command in (sizes, retention):
        command.add_argument(
            "--as-of", default=None, help="instante declarado ISO-8601 con zona (obligatorio)"
        )
        command.add_argument(
            "--data-root", type=Path, default=Path("data"), help="raiz del almacen"
        )
        command.add_argument(
            "--journal-root", type=Path, default=None, help="raiz del diario (obligatorio)"
        )
        command.add_argument(
            "--runs-root", type=Path, default=Path("runs"), help="raiz del registro de experimentos"
        )
        command.add_argument(
            "--reports-dir", type=Path, default=None, help="donde se escribe el informe"
        )
    retention.add_argument(
        "--apply", action="store_true", help="borra de verdad; sin esto, simula (nada se borra)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entrada de los dos comandos. Códigos: ``0`` hecho; ``2`` configuración inválida."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    journal_root_arg = cast("Path | None", args.journal_root)
    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
        if journal_root_arg is None:
            raise ConfigurationError(
                "falta --journal-root: sin diario no hay donde mirar run_log ni la cache"
            )
    except ConfigurationError as error:
        print(f"no se puede ejecutar la higiene de disco: {error}", file=sys.stderr)
        return 2

    data_root = cast(Path, args.data_root)
    runs_root = cast(Path, args.runs_root)
    reports_dir_arg = cast("Path | None", args.reports_dir)
    reports_dir = (
        reports_dir_arg if reports_dir_arg is not None else data_root / "derived" / "reports"
    )
    roots = Roots(data=data_root, journal=journal_root_arg, runs=runs_root)

    if args.command == "sizes":
        report = check_budgets(roots, as_of=moment)
        _write_report(reports_dir, "disk_sizes", report, moment)
        print(render_markdown(report))
        for name in report.over_budget:
            print(f"AVISO: el componente `{name}` supera su presupuesto", file=sys.stderr)
        return 0

    plan = plan_retention(roots, as_of=moment, apply=bool(cast(bool, args.apply)))
    _write_report(reports_dir, "disk_retention", plan, moment)
    print(render_markdown(plan))
    if plan.apply:
        outcome = apply_retention(plan)
        print(f"borrados: {len(outcome.deleted_paths)} rutas ({outcome.deleted_bytes} bytes)")
        for failure in outcome.failures:
            print(f"FALLO: {failure}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
