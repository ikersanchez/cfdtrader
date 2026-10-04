"""Informe mensual de gasto del LLM (`tech_stack.md` §6.3.5) — tarea #117.

Qué responde
------------
«¿Cuánto costó el LLM el mes pasado, a qué ritmo, y ha cambiado algo en el proveedor que haya que
mirar **antes** de asumirlo?» (§6.3.5). Agrega `ops.llm_calls` **por mes** y publica un informe
JSON + MD. Es un comando **manual** y a demanda, como los de #44: no hay *scheduler* (§4.11).

Las cuatro cifras de §6.3.5, y de dónde sale cada una
---------------------------------------------------
1. **Gasto acumulado** — suma de `cost_estimate` de las filas del mes. Con las llamadas que **no**
   declaran tarifa se dice cuántas son y el total se marca **incompleto**: no se rellena con 0.
2. **Coste por titular procesado** — `gasto / titulares enviados`. El denominador **no** está en la
   tabla de llamadas: lo dejó **#129** en el `manifest` de cada sesión (`counters.headlines_sent`).
3. **Ratio de aciertos de caché** — filas con `cache_hit` / filas totales. Se cumple el contrato de
   **#119**: una fila de acierto declara `tokens` y coste en `null`, así que **no** suma al gasto
   ni a los tokens.
4. **Gasto que la deduplicación evitó** — `titulares duplicados × coste medio por titular enviado`,
   publicado como **estimación** (`is_estimate: true`): es un **contrafactual** (por eso existe la
   palanca 3), no una medición, y no se presenta como si lo fuera.

El límite que el informe declara en vez de disimular
----------------------------------------------------
El **numerador** (`ops.llm_calls`) se conserva **18 meses** (§12.6); el **denominador** (el
`manifest`) vive en el directorio de sesión, que la retención de `ops.run_log` purga a los **90
días** (#44). Así que C2 y C4 solo alcanzan al último trimestre. El informe lo dice con su nombre
(`window_caveat`) y cuenta las sesiones que **no** traen conteo en lugar de inventarles titulares.

Solo lectura
------------
No escribe en el diario ni toca `ops.llm_calls`: lee con la API de #39 (`Journal.directory` para
enumerar y `read_record` para interpretar) y con `json.loads` el `manifest` (#43), que es un
artefacto de datos. **No lee el reloj**: el mes entra por `--month`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, cast

from cfdtrader.journal.decision_log import Journal, read_record
from cfdtrader.llm.budget import (
    MONTHLY_WARNING,
    PRICES_EUR_PER_MTOKENS,
    PRICES_VERIFIED_ON,
    BudgetCaps,
)
from cfdtrader.orchestration.observability import MANIFEST_FILENAME

__all__ = [
    "COUNTERS_READ",
    "MONTH_PATTERN",
    "STALE_DENOMINATOR_DAYS",
    "ConfigurationError",
    "build_report",
    "main",
    "parse_month",
    "read_calls",
    "read_counters",
    "render_markdown",
]

#: Los conteos que el informe **lee** del *manifest* (los escribe #129). Se declaran aqui una vez y
#: una prueba comprueba que son exactamente los que escribe el camino diario.
COUNTERS_READ: Final[tuple[str, ...]] = (
    "headlines_sent",
    "headlines_duplicates",
    "headlines_read",
)

#: Formato del mes declarado: ``YYYY-MM``.
MONTH_PATTERN: Final[str] = "%Y-%m"

#: Retencion del *manifest* que sostiene el denominador de §6.3.5: la de ``ops.run_log`` (#44).
STALE_DENOMINATOR_DAYS: Final[int] = 90


class ConfigurationError(Exception):
    """La configuracion del comando no es valida (mes mal formado, falta el diario)."""


def parse_month(value: str) -> tuple[int, int]:
    """El mes declarado como ``(año, mes)``. Un texto que no sea ``YYYY-MM`` es error tipado."""
    try:
        moment = datetime.strptime(value, MONTH_PATTERN)
    except ValueError as error:
        raise ConfigurationError(f"--month: se espera YYYY-MM, no {value!r}") from error
    return moment.year, moment.month


def _month_of(as_of: object) -> tuple[int, int] | None:
    """El ``(año, mes)`` de una fila, desde su ``as_of`` ISO. ``None`` si no es interpretable."""
    if not isinstance(as_of, str):
        return None
    try:
        moment = datetime.fromisoformat(as_of)
    except ValueError:
        return None
    return moment.year, moment.month


def read_calls(journal_root: Path) -> tuple[dict[str, object], ...]:
    """Todas las filas de ``ops.llm_calls``, **por la API de #39** (no abriendo los JSON a mano).

    ``Journal.directory`` enumera y ``read_record`` interpreta: si el esquema cerrado de #39
    cambiase, se nota aqui y no en el informe.
    """
    journal = Journal(journal_root / "ops")
    directory = journal.directory("llm_calls")
    if not directory.is_dir():
        return ()
    return tuple(
        read_record(journal, "llm_calls", path.stem) for path in sorted(directory.glob("*.json"))
    )


def read_counters(journal_root: Path) -> dict[str, dict[str, int]]:
    """Los conteos del lote por **sesion**, desde el *manifest* de cada una (#43, #129).

    Una sesion sin `manifest` —o sin la seccion `counters`— **no** aparece en el mapa: es distinto
    de una sesion con conteo cero, y el informe lo distingue.
    """
    ops_root = journal_root / "ops"
    found: dict[str, dict[str, int]] = {}
    if not ops_root.is_dir():
        return found
    for session_dir in sorted(path for path in ops_root.iterdir() if path.is_dir()):
        manifest = session_dir / MANIFEST_FILENAME
        if not manifest.is_file():
            continue
        try:
            raw: object = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        counters = cast("dict[str, Any]", raw).get("counters")
        if not isinstance(counters, dict) or not counters:
            continue
        parsed = {
            key: value
            for key, value in cast("dict[str, Any]", counters).items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        if parsed:
            found[session_dir.name] = parsed
    return found


def _opt_decimal(value: object) -> Decimal | None:
    """El coste declarado de una fila, como ``Decimal`` exacto. ``None`` si no es un numero."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, str)):
        try:
            return Decimal(str(value))
        except ArithmeticError:  # pragma: no cover - `Decimal(str(x))` solo falla con basura
            return None
    return None


def _text(value: Decimal | None, digits: int = 6) -> str | None:
    """``Decimal`` -> cadena exacta, **sin ceros de relleno** y sin notacion cientifica.

    ``None`` viaja como ``None``. Se cortan solo los ceros de la parte decimal: ``20.000000`` es
    ``20`` y ``0.000160`` es ``0.00016``, pero nunca se toca la parte entera.
    """
    if value is None:
        return None
    text = format(round(value, digits), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _opt_int(value: object) -> int | None:
    """Un entero declarado, o ``None``. Un ``bool`` no cuenta como entero."""
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def session_names(journal_root: Path) -> tuple[str, ...]:
    """Los nombres de las sesiones que hay bajo ``ops/``, **tengan o no** *manifest*."""
    ops_root = journal_root / "ops"
    if not ops_root.is_dir():
        return ()
    return tuple(sorted(path.name for path in ops_root.iterdir() if path.is_dir()))


def build_report(journal_root: Path, month: str) -> dict[str, Any]:
    """El informe del mes declarado. Funcion **pura** sobre el diario: no lee el reloj."""
    year, month_number = parse_month(month)
    target = (year, month_number)
    prefix = f"{year:04d}-{month_number:02d}-"

    calls = [row for row in read_calls(journal_root) if _month_of(row.get("as_of")) == target]
    counters = read_counters(journal_root)
    in_month = {name: value for name, value in counters.items() if name.startswith(prefix)}
    at_month = [name for name in session_names(journal_root) if name.startswith(prefix)]
    without_counters = [name for name in at_month if name not in in_month]

    hits = sum(1 for row in calls if row.get("cache_hit") is True)
    costs = [_opt_decimal(row.get("cost_estimate")) for row in calls]
    known = [cost for cost in costs if cost is not None]
    spend: Decimal | None = sum(known, Decimal(0)) if calls else None
    tokens_in = sum(value for value in (_opt_int(row.get("tokens_in")) for row in calls) if value)
    tokens_out = sum(value for value in (_opt_int(row.get("tokens_out")) for row in calls) if value)
    without_tokens = sum(
        1
        for row in calls
        if _opt_int(row.get("tokens_in")) is None and _opt_int(row.get("tokens_out")) is None
    )

    sent = sum(value.get("headlines_sent", 0) for value in in_month.values())
    duplicates = sum(value.get("headlines_duplicates", 0) for value in in_month.values())
    read = sum(value.get("headlines_read", 0) for value in in_month.values())
    ratio = Decimal(hits) / Decimal(len(calls)) if calls else None
    per_headline = spend / Decimal(sent) if spend is not None and sent > 0 else None

    cap = BudgetCaps().caps().monthly_budget_eur
    breached = cap is not None and spend is not None and spend > Decimal(str(cap))

    return {
        "task": "#117",
        "mode": "llm_cost_report",
        "month": month,
        "sources": {
            "calls": (journal_root / "ops" / "llm_calls").as_posix(),
            "counters": (journal_root / "ops" / "<sesion>" / MANIFEST_FILENAME).as_posix(),
        },
        "prices": {
            "table_eur_per_mtoken": {
                model: [_text(Decimal(str(entry))) for entry in tariff]
                for model, tariff in PRICES_EUR_PER_MTOKENS.items()
            },
            "verified_on": PRICES_VERIFIED_ON,
        },
        "calls": {
            "total": len(calls),
            "real": len(calls) - hits,
            "cache_hits": hits,
            "cache_hit_ratio": _text(ratio),
            "without_declared_cost": len(calls) - len(known),
        },
        "tokens": {
            "in": tokens_in,
            "out": tokens_out,
            "calls_without_tokens": without_tokens,
        },
        "spend_eur": {
            "total": _text(spend),
            "complete": len(known) == len(calls),
            "state": "measured" if calls else "not_measurable",
            "reason": None
            if len(known) == len(calls)
            else "hay llamadas sin tarifa declarada: el total es un minimo conocido, no el gasto",
        },
        "per_headline_eur": {
            "value": _text(per_headline),
            "headlines_sent": sent,
            "headlines_read": read,
            "sessions_with_counters": len(in_month),
            "sessions_without_counters": len(without_counters),
            "state": "measured" if per_headline is not None else "not_measurable",
            "reason": None
            if per_headline is not None
            else (
                "sin gasto del mes o sin ningun conteo de titulares: el denominador de §6.3.5 no "
                "existe. El conteo vive en el manifest de la sesion (#129), que la retencion de "
                "ops.run_log purga a los 90 dias (#44)"
            ),
        },
        "avoided_spend_eur": {
            "is_estimate": True,
            "duplicates": duplicates,
            "per_headline_eur": _text(per_headline),
            "value": _text(duplicates * per_headline if per_headline is not None else None),
            "formula": "titulares_duplicados x coste medio por titular enviado",
            "reason": (
                "es un contrafactual, no una medicion: la palanca 3 existe justamente para no "
                "pagar esos titulares, asi que el gasto evitado no aparece en ninguna fila"
            ),
        },
        "monthly_cap_eur": {
            "declared": None if cap is None else _text(Decimal(str(cap))),
            "breached": breached,
            "warning": MONTHLY_WARNING if breached else None,
        },
        "window_caveat": (
            f"el numerador (ops.llm_calls) se conserva 18 meses; el denominador (el manifest) solo "
            f"{STALE_DENOMINATOR_DAYS} dias (#44): las cifras por titular alcanzan al ultimo "
            "trimestre y se declaran como tales"
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """El informe en prosa legible, con lo medido y lo estimado separados a la vista."""
    calls = report["calls"]
    spend = report["spend_eur"]
    per_headline = report["per_headline_eur"]
    avoided = report["avoided_spend_eur"]
    cap = report["monthly_cap_eur"]
    tokens = report["tokens"]
    prices = report["prices"]

    ratio = calls["cache_hit_ratio"]
    lines = [
        f"# Coste del LLM: {report['month']} (tarea #117)",
        "",
        f"- **Llamadas:** {calls['total']} ({calls['real']} reales, "
        f"{calls['cache_hits']} de cache)",
        f"- **Aciertos de cache:** {'sin llamadas' if ratio is None else f'{ratio}'}",
        f"- **Tokens:** {tokens['in']} de entrada, {tokens['out']} de salida "
        f"({tokens['calls_without_tokens']} filas sin tokens: aciertos, #119)",
        f"- **Tarifa verificada el:** {prices['verified_on']}",
        "",
        "| Cifra | Valor | Estado |",
        "|---|---:|---|",
        f"| Gasto del mes | {spend['total']} | {spend['state']} |",
        f"| Coste por titular procesado | {per_headline['value']} | {per_headline['state']} |",
        f"| Gasto que la deduplicacion evito | {avoided['value']} | **estimacion** |",
        "",
    ]
    if spend["reason"] is not None:
        lines.append(f"> **Gasto incompleto:** {spend['reason']}")
    if per_headline["reason"] is not None:
        lines.append(f"> **Sin coste por titular:** {per_headline['reason']}")
    lines += [
        f"> **{avoided['formula']}** sobre {avoided['duplicates']} titulares duplicados. "
        f"{avoided['reason']}",
        "",
        f"- Titulares leidos: {per_headline['headlines_read']} | enviados: "
        f"{per_headline['headlines_sent']} | sesiones con conteo: "
        f"{per_headline['sessions_with_counters']} | sin conteo: "
        f"{per_headline['sessions_without_counters']}",
        f"- Tope mensual declarado: {cap['declared']} | superado: {cap['breached']}",
        f"- Fuente de llamadas: `{report['sources']['calls']}`",
        f"- Fuente del conteo: `{report['sources']['counters']}`",
        "",
        f"> ⚠️ {report['window_caveat']}",
        "",
        "> Comando **manual**: no hay *scheduler* ni servicio que lo dispare "
        "(`tech_stack.md` §4.11).",
        "",
    ]
    if cap["warning"] is not None:
        lines.insert(len(lines) - 3, f"> ⚠️ **{cap['warning']}**")
    return "\n".join(lines)


def _write_report(reports_dir: Path, report: Mapping[str, Any], month: str) -> tuple[Path, Path]:
    """Escribe el JSON y el MD del mes. Determinista: las claves van ordenadas."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"llm_cost_{month}"
    json_path = reports_dir / f"{stem}.json"
    md_path = reports_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfdtrader.ops.cost_report",
        description=(
            "Informe mensual de gasto del LLM (tarea #117). Comando manual y a demanda: no hay "
            "scheduler ni servicio que lo dispare (tech_stack.md §4.11)."
        ),
    )
    parser.add_argument("--month", default=None, help="mes declarado YYYY-MM (obligatorio)")
    parser.add_argument("--journal-root", default=None, help="raiz del diario (obligatorio)")
    parser.add_argument(
        "--reports-dir", type=Path, default=None, help="donde se escribe el informe"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entrada del comando. Codigos: ``0`` hecho; ``2`` configuracion invalida."""
    args = _build_parser().parse_args(argv)
    month_arg = args.month
    journal_root_arg = args.journal_root
    try:
        if not isinstance(month_arg, str) or not month_arg.strip():
            raise ConfigurationError("falta --month: sin mes declarado no hay informe")
        parse_month(month_arg)
        if not isinstance(journal_root_arg, str) or not journal_root_arg.strip():
            raise ConfigurationError("falta --journal-root: sin diario no hay llamadas que sumar")
    except ConfigurationError as error:
        print(f"no se puede emitir el informe de coste: {error}", file=sys.stderr)
        return 2

    journal_root = Path(journal_root_arg)
    reports_dir = args.reports_dir if isinstance(args.reports_dir, Path) else journal_root / "ops"
    report = build_report(journal_root, month_arg)
    _write_report(reports_dir, report, month_arg)
    print(render_markdown(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
