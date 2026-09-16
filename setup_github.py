#!/usr/bin/env python3
"""
Bootstrap de GitHub para el proyecto cfdtrader.

Crea el repositorio remoto, los labels, los milestones por fase y **una issue
por cada tarea de `tasks.md`**. Es idempotente: puede ejecutarse varias veces
sin duplicar repositorio, labels, milestones ni issues.

Requisito previo
----------------
    gh auth login          # GitHub CLI autenticado

Uso
---
    python3 setup_github.py                 # hace todo
    python3 setup_github.py --dry-run       # muestra lo que haría, sin tocar nada
    python3 setup_github.py --repo-only     # solo git init + repo remoto + push
    python3 setup_github.py --issues-only   # solo labels, milestones e issues

Fuente de verdad
----------------
El script **lee `tasks.md`** en cada ejecución, así que las issues siempre
reflejan el estado actual del backlog. `tasks.md` sigue siendo la fuente de
verdad: si se edita una tarea aquí, hay que volver a ejecutar el script (las
issues ya creadas no se actualizan automáticamente; se avisa de los cambios).

Formato que espera de `tasks.md`
--------------------------------
    # Fase 0 — Medir la realidad
    ## 1. Título de la tarea
    Goal: una línea
    Description: una o más líneas
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

REPO_NAME = "cfdtrader"
VISIBILITY = "--public"
REPO_DESCRIPTION = (
    "Sistema multiagente de decisión intradía sobre CFD del S&P 500: "
    "medición, backtest con purga y embargo, gate determinista. Diseño, sin código todavía."
)

TASKS_FILE = Path("tasks.md")
IGNORED_DIRS = {".git"}

# Marcas transversales que no se deducen del parseo (cuidado si se renumera)
CRITICAL_TASKS = {6, 8, 17, 39}  # las cuatro de mayor peso sobre el resultado
GATE_TASKS = {9, 18, 29, 45}  # tareas que actúan de puerta de salida de fase
MEASUREMENT_TASKS = {6, 7, 8}  # mediciones de la Fase 0

# ─────────────────────────────────────────────────────────────────────────────
# Parseo de tasks.md
# ─────────────────────────────────────────────────────────────────────────────

TASK_RE = re.compile(r"^##\s+(\d+)\.\s+(.*?)\s*$")
PHASE_RE = re.compile(r"^#\s+Fase\s+(\d+)\s+—\s*(.*?)\s*$")
GOAL_RE = re.compile(r"(?:\*\*)?Goal(?:\*\*)?:\s*(.+?)(?:\n\s*\n|\Z)", re.S)
DESC_RE = re.compile(r"(?:\*\*)?Description(?:\*\*)?:\s*(.+)\Z", re.S)
LEADING_MARKS_RE = re.compile(r"^[\s★⭐⚠️❌🎯]+")
MD_EMPHASIS_RE = re.compile(r"[*_`]")


def clean_title(raw: str) -> str:
    """Quita marcas decorativas y de markdown: los títulos de issue no se renderizan."""
    title = LEADING_MARKS_RE.sub("", raw).strip()
    title = MD_EMPHASIS_RE.sub("", title)
    return re.sub(r"\s{2,}", " ", title).strip()


@dataclass
class Task:
    number: int
    title: str
    goal: str
    description: str
    phase_number: int | None
    phase_title: str
    raw_block: str = ""
    labels: list[str] = field(default_factory=list)


def parse_tasks(path: Path) -> list[Task]:
    """Extrae las tareas y su fase desde `tasks.md`."""
    if not path.exists():
        sys.exit(f"ERROR: no encuentro {path.resolve()}")

    lines = path.read_text(encoding="utf-8").splitlines()
    tasks: list[Task] = []
    phase_number: int | None = None
    phase_title = "(sin fase)"

    i = 0
    while i < len(lines):
        line = lines[i]

        m_phase = PHASE_RE.match(line)
        if m_phase:
            phase_number = int(m_phase.group(1))
            phase_title = m_phase.group(2).strip()
            i += 1
            continue

        m_task = TASK_RE.match(line)
        if m_task:
            number = int(m_task.group(1))
            title = clean_title(m_task.group(2))

            # El bloque termina en el siguiente heading de cualquier nivel
            block: list[str] = []
            j = i + 1
            while j < len(lines) and not lines[j].startswith("#"):
                block.append(lines[j])
                j += 1

            raw_block = "\n".join(block).strip()
            m_goal = GOAL_RE.search(raw_block)
            m_desc = DESC_RE.search(raw_block)

            tasks.append(
                Task(
                    number=number,
                    title=title,
                    goal=m_goal.group(1).strip().replace("\n", " ") if m_goal else "",
                    description=m_desc.group(1).strip() if m_desc else "",
                    phase_number=phase_number,
                    phase_title=phase_title,
                    raw_block=raw_block,
                )
            )
            i = j
            continue

        i += 1

    if not tasks:
        sys.exit(f"ERROR: no he encontrado ninguna tarea en {path}")

    # Validación: numeración correlativa desde 1 sin huecos
    numbers = [t.number for t in tasks]
    expected = list(range(1, len(tasks) + 1))
    if numbers != expected:
        print(f"AVISO: la numeración no es correlativa. Encontradas: {numbers[:12]}…")
        missing = sorted(set(expected) - set(numbers))
        if missing:
            print(f"AVISO: faltan los números {missing}")

    # Etiquetas transversales
    for t in tasks:
        if t.phase_number is not None:
            t.labels.append(f"fase-{t.phase_number}")
        if t.number in CRITICAL_TASKS:
            t.labels.append("critica")
        if t.number in GATE_TASKS:
            t.labels.append("puerta-de-salida")
        if t.number in MEASUREMENT_TASKS:
            t.labels.append("medicion")

    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de shell
# ─────────────────────────────────────────────────────────────────────────────


def run(
    cmd: list[str],
    *,
    dry: bool = False,
    check: bool = True,
    input_text: str | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Ejecuta un comando. Con `dry=True` solo lo imprime."""
    printable = " ".join(cmd) if input_text is None else " ".join(cmd) + " <body>"
    if dry:
        print(f"    · {printable}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=capture,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Fallo ejecutando: {printable}\n{stderr}")
    return result


def require_gh() -> str:
    """Verifica que `gh` existe y está autenticado. Devuelve `owner/repo`."""
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit(
            "ERROR: `gh` (GitHub CLI) no está disponible.\n"
            "  Instálalo con:  sudo apt install gh   |   sudo snap install gh\n"
            "  Y autentícate:  gh auth login"
        )

    status = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if status.returncode != 0:
        sys.exit(
            "ERROR: `gh` no está autenticado.\n"
            "  Ejecuta:  gh auth login\n\n"
            f"{status.stderr or status.stdout}"
        )

    who = subprocess.run(
        ["gh", "api", "user", "--jq", ".login"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(f"  ✓ gh autenticado como: {who}")
    return f"{who}/{REPO_NAME}"


# ─────────────────────────────────────────────────────────────────────────────
# Paso 1 — Repositorio
# ─────────────────────────────────────────────────────────────────────────────


def bootstrap_repo(slug: str, dry: bool) -> None:
    print("\n[1/4] Repositorio")

    if not Path(".git").exists():
        print("  · inicializando repositorio local")
        run(["git", "init", "-b", "main"], dry=dry)

    # ¿Hay algo que commitear?
    pending = run(["git", "status", "--porcelain"], dry=dry)
    if dry or (pending.stdout or "").strip():
        run(["git", "add", "-A"], dry=dry)
        run(
            ["git", "commit", "-m", "docs: plan.md, tech_stack.md y tasks.md"],
            dry=dry,
        )
        print("  · commit inicial creado")
    else:
        print("  · sin cambios pendientes de commit")

    # ¿Existe ya el remoto?
    remotes = run(["git", "remote"], dry=dry)
    has_origin = not dry and "origin" in (remotes.stdout or "").split()

    if dry:
        run(
            ["gh", "repo", "create", REPO_NAME, VISIBILITY, "--source=.", "--push",
             "--remote=origin", "--description", REPO_DESCRIPTION],
            dry=True,
        )
        return

    if has_origin:
        print("  · el remoto `origin` ya existe; empujando cambios")
        run(["git", "push", "-u", "origin", "main"], check=False)
        return

    result = run(
        [
            "gh", "repo", "create", REPO_NAME, VISIBILITY,
            "--source=.", "--remote=origin", "--push",
            "--description", REPO_DESCRIPTION,
        ],
        check=False,
    )
    if result.returncode == 0:
        print(f"  ✓ repositorio creado: https://github.com/{slug}")
        return

    stderr = (result.stderr or "") + (result.stdout or "")
    if "already exists" in stderr.lower():
        print(f"  · el repositorio ya existe en GitHub; enlazando y empujando")
        run(["git", "remote", "add", "origin", f"https://github.com/{slug}.git"],
            check=False)
        run(["git", "push", "-u", "origin", "main"], check=False)
        print(f"  ✓ empujado a https://github.com/{slug}")
    else:
        raise RuntimeError(f"No he podido crear el repositorio:\n{stderr}")


# ─────────────────────────────────────────────────────────────────────────────
# Paso 2 — Labels
# ─────────────────────────────────────────────────────────────────────────────

LABELS: list[tuple[str, str, str]] = [
    ("fase-0", "1D76DB", "Fase 0 — Medir la realidad"),
    ("fase-1", "0E8A16", "Fase 1 — Arnés de backtest"),
    ("fase-2", "FBCA04", "Fase 2 — Núcleo cuantitativo"),
    ("fase-3", "D93F0B", "Fase 3 — Capa LLM"),
    ("fase-4", "5319E7", "Fase 4 — Operación"),
    ("fase-5", "006B75", "Fase 5 — Producción"),
    ("critica", "B60205", "De las tareas de mayor peso sobre el resultado"),
    ("puerta-de-salida", "E99695", "Actúa de puerta de salida de su fase"),
    ("medicion", "C2E0C6", "Medición de realidad, no construcción"),
]


def ensure_labels(dry: bool) -> None:
    print("\n[2/4] Labels")
    for name, color, description in LABELS:
        run(
            ["gh", "label", "create", name, "--color", color,
             "--description", description, "--force"],
            dry=dry,
        )
    print(f"  ✓ {len(LABELS)} labels asegurados")


# ─────────────────────────────────────────────────────────────────────────────
# Paso 3 — Milestones (una por fase)
# ─────────────────────────────────────────────────────────────────────────────


def ensure_milestones(slug: str, tasks: list[Task], dry: bool) -> dict[str, int]:
    """Crea un milestone por fase y devuelve {título: número}."""
    print("\n[3/4] Milestones")

    titles: list[str] = []
    for t in tasks:
        if t.phase_number is None:
            continue
        title = f"Fase {t.phase_number} — {t.phase_title}"
        if title not in titles:
            titles.append(title)

    mapping: dict[str, int] = {}

    existing_raw = run(
        ["gh", "api", f"repos/{slug}/milestones?state=all&per_page=100",
         "--jq", '.[] | "\\(.number)\\t\\(.title)"'],
        dry=dry,
    )
    if not dry:
        for line in (existing_raw.stdout or "").splitlines():
            if "\t" in line:
                num, title = line.split("\t", 1)
                mapping[title.strip()] = int(num)

    for title in titles:
        if title in mapping:
            print(f"  · ya existe: {title}")
            continue
        result = run(
            ["gh", "api", f"repos/{slug}/milestones",
             "-f", f"title={title}", "-f", "state=open",
             "--jq", ".number"],
            dry=dry,
        )
        if not dry:
            mapping[title] = int((result.stdout or "0").strip() or 0)
        else:
            mapping[title] = 0
        print(f"  ✓ creado: {title}")

    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Paso 4 — Issues
# ─────────────────────────────────────────────────────────────────────────────


def build_body(task: Task, slug: str) -> str:
    """Cuerpo de la issue: la tarea tal cual más metadatos de trazabilidad."""
    parts = [
        f"**Goal:** {task.goal}" if task.goal else "",
        "",
        f"**Description:** {task.description}" if task.description else "",
        "",
        "---",
        "",
        "| | |",
        "|---|---|",
        f"| **Fase** | Fase {task.phase_number} — {task.phase_title} |",
        f"| **Origen** | `tasks.md` · tarea `#{task.number}` |",
        f"| **Documentos** | [`plan.md`](https://github.com/{slug}/blob/main/plan.md) · "
        f"[`tech_stack.md`](https://github.com/{slug}/blob/main/tech_stack.md) |",
    ]
    if task.number in GATE_TASKS:
        parts.append(
            f"| **Puerta de salida** | Sí. Ver la nota de fase en `tasks.md`. |"
        )
    parts += [
        "",
        "> Tarea de una sola sesión. **No se cierra sin un artefacto verificable** "
        "y sin respetar su regla de aceptación.",
        "",
        "<sub>Generada desde `tasks.md`, que sigue siendo la fuente de verdad.</sub>",
    ]
    return "\n".join(p for p in parts if p is not None)


def create_issues(slug: str, tasks: list[Task], milestones: dict[str, int], dry: bool) -> None:
    print("\n[4/4] Issues")

    existing_raw = run(
        ["gh", "issue", "list", "--state", "all", "--limit", "1000",
         "--json", "title", "--jq", '.[].title'],
        dry=dry,
    )
    existing = set((existing_raw.stdout or "").splitlines()) if not dry else set()

    created = skipped = 0

    for t in tasks:
        title = f"T{t.number} — {t.title}"

        if title in existing:
            print(f"  · ya existe: {title}")
            skipped += 1
            continue

        milestone = f"Fase {t.phase_number} — {t.phase_title}" if t.phase_number is not None else None

        cmd = ["gh", "issue", "create", "--title", title, "--body-file", "-"]
        for label in t.labels:
            cmd += ["--label", label]
        if milestone and milestone in milestones:
            cmd += ["--milestone", milestone]

        try:
            run(cmd, dry=dry, input_text=build_body(t, slug))
            created += 1
            if not dry:
                print(f"  ✓ {title}")
        except RuntimeError as exc:
            print(f"  ✗ FALLO en {title}\n    {exc}")

    print(f"\n  Resumen: {created} creadas, {skipped} ya existían")
    if dry:
        print("  (modo --dry-run: no se ha creado nada)")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crea el repositorio de GitHub y una issue por tarea de tasks.md"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="muestra lo que haría sin tocar nada")
    parser.add_argument("--repo-only", action="store_true",
                        help="solo git init + repositorio remoto + push")
    parser.add_argument("--issues-only", action="store_true",
                        help="solo labels, milestones e issues")
    parser.add_argument("--show-body", type=int, metavar="N",
                        help="imprime el cuerpo que se generaría para la tarea N y sale")
    args = parser.parse_args()

    if args.show_body is not None:
        tasks = parse_tasks(TASKS_FILE)
        match = next((t for t in tasks if t.number == args.show_body), None)
        if match is None:
            sys.exit(f"ERROR: no existe la tarea {args.show_body}")
        print(f"TÍTULO: T{match.number} — {match.title}\n")
        print(f"LABELS: {', '.join(match.labels)}\n")
        print("-" * 72)
        print(build_body(match, "<usuario>/cfdtrader"))
        print("-" * 72)
        return 0

    print("=" * 72)
    print(f"Bootstrap de GitHub — {REPO_NAME} (público)")
    print("=" * 72)

    tasks = parse_tasks(TASKS_FILE)
    phases = sorted({t.phase_number for t in tasks if t.phase_number is not None})
    print(f"\ntasks.md: {len(tasks)} tareas en {len(phases)} fases "
          f"(tareas {tasks[0].number}–{tasks[-1].number})")

    if args.dry_run:
        print("\n*** MODO --dry-run: no se modificará nada ***")
        slug = f"<usuario>/{REPO_NAME}"
    else:
        slug = require_gh()

    if not args.issues_only:
        bootstrap_repo(slug, args.dry_run)

    if not args.repo_only:
        ensure_labels(args.dry_run)
        milestones = ensure_milestones(slug, tasks, args.dry_run)
        create_issues(slug, tasks, milestones, args.dry_run)

    print("\n" + "=" * 72)
    if not args.dry_run:
        print(f"Listo: https://github.com/{slug}/issues")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("\nInterrumpido por el usuario.")
