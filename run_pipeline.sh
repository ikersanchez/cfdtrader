#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh — pipeline completo de cfdtrader + REPORTE FINAL de decisión.
#
# Un solo comando para: ingerir, regenerar los artefactos, capturar el
# pre-mercado del ES, emitir la pista del día, evaluar la valla de cartera,
# publicar la tarjeta de operación, aplicar la puerta del paper y, al final,
# imprimir el REPORTE FINAL con la decisión: (1) la publicada SIN overlay
# (LONG/SHORT, probabilidad, corte del tier A, EV, stop/objetivo, coste, tamaño)
# y (2) la variante CON overlay (que puede convertirla en NOTHING por veto).
#
# Uso:
#   bash run_pipeline.sh                 # pipeline completo (1ª vez / tras ingesta)
#   bash run_pipeline.sh --daily-only    # solo el día (sin la regeneración pesada)
#   bash run_pipeline.sh --help
#
# La fuente de verdad del procedimiento es _docs/runbook.md. Este script lo
# ejecuta tal cual y no inventa pasos.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

cd "$(dirname "$0")" || exit 1

DAILY_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --daily-only) DAILY_ONLY=1 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "argumento no reconocido: $arg" >&2; exit 2 ;;
    esac
done

export SESSION="$(date -u +%Y-%m-%d)"
TS="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
R="data/derived/reports"
OPS="journal/ops/$SESSION"
mkdir -p "$OPS" "$R"

step() {
    local name="$1"; shift
    echo
    echo "────────────────────────────────────────────────────────────────"
    echo "▶ $name"
    "$@"
    local rc=$?
    [ "$rc" -ne 0 ] && echo "  ⚠  '$name' terminó con rc=$rc (se continúa)"
    return 0
}

echo "════════════════════════════════════════════════════════════════════"
echo "  cfdtrader · pipeline $([ "$DAILY_ONLY" -eq 1 ] && echo '(solo día)' || echo '(completo)') · sesión $SESSION"
echo "════════════════════════════════════════════════════════════════════"

# ── 1. Ingesta — refresca el almacén hasta el cierre de ayer ─────────────────
step "1/8 ingesta · market"   uv run python -m cfdtrader.data.market   --data-root data --now "$TS"
step "1/8 ingesta · macro"    uv run python -m cfdtrader.data.macro    --data-root data --now "$TS"
step "1/8 ingesta · earnings" uv run python -m cfdtrader.data.earnings --data-root data --now "$TS"
step "1/8 ingesta · noticias" uv run python -m cfdtrader.data.news --data-root data --as-of "$TS" \
    --feed 'cnbc-top=https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114' \
    --feed 'cnbc-energy=https://www.cnbc.com/id/19836768/device/rss/rss.html' \
    --feed 'oilprice=https://oilprice.com/rss/main' \
    --feed 'google-news=https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en'

# ── 2. Regeneración en orden (_docs/process.md regla 2) ──────────────────────
if [ "$DAILY_ONLY" -eq 0 ]; then
    step "2/8 regeneración · labels"              uv run python -m cfdtrader.models.labels                --data-root data --reports-dir "$R" --now 2026-09-18T00:00:00+00:00
    step "2/8 regeneración · volatility"          uv run python -m cfdtrader.analysis.volatility_forecast --data-root data --now 2026-09-18T00:00:00+00:00
    step "2/8 regeneración · backtest_report"     uv run python -m cfdtrader.analysis.backtest_report     --data-root data --reports-dir "$R" --as-of 2026-09-19T00:00:00+00:00
    step "2/8 regeneración · phase1_report"       uv run python -m cfdtrader.analysis.phase1_report       --data-root data --reports-dir "$R" --as-of 2026-09-19T22:00:00+00:00
    step "2/8 regeneración · baseline_report"     uv run python -m cfdtrader.analysis.baseline_report     --data-root data --reports-dir "$R" --runs-root runs --as-of 2026-09-22T22:00:00+00:00 --previous-artifact "$R/baseline_2026-09-22.json"
    step "2/8 regeneración · baseline_report raw" uv run python -m cfdtrader.analysis.baseline_report     --data-root data --reports-dir "$R" --runs-root runs --as-of 2026-09-22T22:00:00+00:00 --raw --previous-artifact "$R/baseline_2026-09-22.json"

    # La cadena de modelos exige apartar los ensayos del barrido (§19.17, #82):
    # si no, model_comparison queda con matriz incompleta y phase2 no copia el pbo.
    STASH=".scratch/runs_search"
    mkdir -p "$STASH"
    moved=0
    for d in runs/*/; do
        [ -d "$d" ] || continue
        variant="$(python3 -c "import json;print(json.load(open('$d/config.json'))['config']['variant_id'])" 2>/dev/null || true)"
        case "$variant" in lightgbm_search_v1#*) mv "$d" "$STASH/"; moved=$((moved + 1)) ;; esac
    done
    echo "  (apartados $moved ensayos del barrido; runs/ se restaura al final)"
    restore_runs() { for d in "$STASH"/*/; do [ -d "$d" ] && mv "$d" runs/; done; }
    trap restore_runs EXIT

    step "2/8 regeneración · model_comparison"  uv run python -m cfdtrader.analysis.model_comparison --data-root data --reports-dir "$R" --runs-root runs --as-of 2026-09-22T22:00:00+00:00 --previous-artifact "$R/model_comparison_2026-09-22.json"
    step "2/8 regeneración · pipeline_report"   uv run python -m cfdtrader.analysis.pipeline_report  --data-root data --reports-dir "$R" --as-of 2026-09-23T22:00:00+00:00 --previous-artifact "$R/pipeline_backtest_2026-09-23.json"
    step "2/8 regeneración · gate_sweep"        uv run python -m cfdtrader.analysis.gate_sweep       --data-root data --reports-dir "$R" --as-of 2026-09-23T22:00:00+00:00
    step "2/8 regeneración · phase2_dominance"  uv run python -m cfdtrader.analysis.phase2_dominance --data-root data --reports-dir "$R" --as-of 2026-09-24T00:00:00+00:00 --previous-artifact "$R/phase2_dominance_2026-09-24.json"

    restore_runs
    trap - EXIT
else
    echo
    echo "▶ 2/8 regeneración — omitida (--daily-only); se usa lo ya publicado en $R"
fi


# ── 3. Pre-mercado del ES (#141) — dato declarado de la decisión ─────────────
step "3/8 pre-mercado ES" uv run python -m cfdtrader.analysis.premarket_gap \
    --as-of "$(date -u +%Y-%m-%d)T13:00:00+00:00" \
    --reports-dir "$R" \
    --capture-bars "$OPS/premarket_bars.json"

# ── 4. Camino diario: la pista del día (§19.7) ───────────────────────────────
echo
echo "────────────────────────────────────────────────────────────────"
echo "▶ 4/8 camino diario (la pista del día)"
uv run python -m cfdtrader.delivery.run_daily \
    --as-of "$(date -u +%Y-%m-%d)T13:00:00+00:00" \
    --variant-id lightgbm_gbdt_v1 \
    --journal-root journal \
    --runs-root runs \
    --git-commit "$(git rev-parse HEAD)" \
    --premarket-bars "$OPS/premarket_bars.json" \
    > "$OPS/daily_report.txt" 2> "$OPS/daily_report.err"
RD_RC=$?
cat "$OPS/daily_report.txt"
if [ "$RD_RC" -ne 0 ]; then
    echo "  ⚠  el camino diario terminó con rc=$RD_RC:"
    cat "$OPS/daily_report.err"
fi

# ── 5. Valla de cartera (§12 reglas 3, 4 y 5 — #83) ──────────────────────────
step "5/8 valla de cartera" uv run python -m cfdtrader.analysis.portfolio_rules \
    --journal-root journal \
    --session "$SESSION" \
    --as-of "$(date -u +%Y-%m-%d)T12:00:00+00:00" \
    --reports-dir "$R"

# ── 6. Tarjeta de operación con tamaño mínimo (#47) ──────────────────────────
step "6/8 tarjeta de operación" uv run python -m cfdtrader.delivery.production \
    --journal-root journal --session "$SESSION"

# ── 7. Puerta de la Fase 4 — el veredicto del paper (§16, #45) ───────────────
step "7/8 puerta del paper" uv run python -m cfdtrader.analysis.paper_trading \
    --data-root data --journal-root journal \
    --reference-artifact "$R/pipeline_backtest_2026-09-23.json" \
    --reports-dir "$R" \
    --as-of "$(date -u +%Y-%m-%d)T22:00:00+00:00"

# ── 8. REPORTE FINAL DE LA DECISIÓN ──────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════════"
echo "  REPORTE FINAL DE LA DECISIÓN — sesión $SESSION"
echo "════════════════════════════════════════════════════════════════════"
uv run python - <<'PY'
import json
import os
import pathlib
import sys

session = os.environ.get("SESSION", "")

# Constantes declaradas del escenario S1 (no inventadas: están en el código).
TIER_A_CUT = 0.58        # params.tier_a_min_probability (escenario S1, #60)
DIRECTION_CUT = 0.50     # DECISION_THRESHOLD (decision/gate.py)
TIER_A_COST_MULT = 3     # params.tier_a_cost_multiple (regla 10)

path = pathlib.Path("journal/decisions") / f"{session}.json"
if not path.exists():
    print(f"\nNo hay fila en journal/decisions/{session}.json.")
    print("El pipeline no emitió decisión para esta sesión (revisa el paso 4).")
    sys.exit(0)

d = json.loads(path.read_text(encoding="utf-8"))

# Algunas claves (p. ej. `ev_declarado_pct`) no viven en la fila del diario pero sí en el
# informe; se leen de sus líneas `clave: valor` como respaldo.
text = d.get("report_text") or ""
from_text = {}
for ln in text.splitlines():
    if ":" in ln and not ln.startswith(" "):
        key, _, value = ln.partition(":")
        from_text.setdefault(key.strip(), value.strip())

def get(key):
    value = d.get(key)
    return value if value is not None else from_text.get(key)

def show_pct(value):
    if value is None:
        return "null"
    try:
        return f"{float(value):.4f} %"
    except (TypeError, ValueError):
        return str(value)

def show_raw(value):
    return "null" if value is None else value

p = d.get("prob_up_calibrated")
status = d.get("status")
direction = d.get("direction")
tier = d.get("tier")

# Probabilidad a favor del lado que el decididor opera: p si p >= 0.50, si no 1-p.
fav = None if p is None else (p if p >= DIRECTION_CUT else 1.0 - p)

if status == "recommendation" and direction:
    decision = direction.upper()
elif status == "recommendation":
    decision = "NOTHING (no se opera)"
else:
    decision = f"SIN RECOMENDACIÓN ({status})"

side = "null"
if p is not None:
    side = "LONG" if p >= DIRECTION_CUT else "SHORT"

print()
print(f"  Estado del informe ......... {status}")
print(f"  DECISIÓN FINAL (sin overlay)  {decision}")
print()
print(f"  p(up) calibrada ............ {'null' if p is None else f'{p:.4f}'}")
print(f"  p a favor del lado ......... {'null' if fav is None else f'{fav:.4f}'}   (p si p>=0.50, 1-p si no)")
print(f"  Corte dirección (p) ........ {DIRECTION_CUT:.4f}   -> {side}")
if fav is None:
    corte = "null"
else:
    corte = "PASA (> 0.58)" if fav > TIER_A_CUT else "NO PASA (<= 0.58)"
print(f"  Corte tier A (p a favor) ... {TIER_A_CUT:.4f}   -> {corte}")
print(f"  Tier ....................... {tier}")
print()

# ── Variante «con overlay» (§19.19 / #149): vive aparte en journal/agent_signals ──
# La recomendación PUBLICADA se emite SIN overlay; aquí se muestra qué haría el
# overlay (p. ej. un veto que la convierte en NOTHING).
overlay_state = d.get("llm_overlay")
overlay_decision = "= sin overlay (no hay señal registrada)"
overlay_prob = None
overlay_reason = None
overlay_file = pathlib.Path("journal/agent_signals") / f"{session}__news.json"
if overlay_file.exists():
    o = json.loads(overlay_file.read_text(encoding="utf-8"))
    ev = o.get("evidence") or {}
    overlay_state = ev.get("state") or overlay_state
    overlay_prob = o.get("prob_up")
    overlay_reason = o.get("veto_reason")
    odir = ev.get("direction")
    if odir == "nothing":
        overlay_decision = "NOTHING (veto del overlay)" if o.get("veto") else "NOTHING"
    elif odir in ("long", "short"):
        overlay_decision = odir.upper()
    else:
        overlay_decision = "NOTHING"
print("  ── Overlay de noticias (§19.19) ──")
print(f"  Estado del overlay ......... {show_raw(overlay_state)}")
print(f"  Decisión CON overlay ....... {overlay_decision}")
print(f"  p(up) con overlay .......... {show_raw(overlay_prob)}")
if overlay_reason:
    print(f"  Motivo del veto ............ {overlay_reason}")
print("  (la recomendación PUBLICADA se emite sin overlay; esta línea es informativa)")
print()
print(f"  EV declarado (%) ........... {show_pct(get('ev_declarado_pct'))}   (regla 9: > {TIER_A_COST_MULT}x coste)")
print(f"  EV neto · sensibilidad (%) . {show_pct(get('ev_net_pct'))}")
print(f"  Coste declarado (%) ........ {show_pct(get('cost_pct'))}")
print(f"  Movimiento esperado (%) .... {show_pct(get('expected_move_pct'))}")
print(f"  Stop (%) ................... {show_pct(get('stop_pct'))}")
print(f"  Objetivo (%) ............... {show_pct(get('target_pct'))}")
print()
print(f"  Tamaño (fracción capital) .. {show_raw(d.get('size_fraction'))}")
print(f"  Nocional (EUR) ............. {show_raw(d.get('size_notional_eur'))}")
print(f"  Apalancamiento implícito ... {show_raw(d.get('leverage_implied'))}")
print(f"  Overlay de noticias ........ {show_raw(d.get('llm_overlay'))}")

lines = [ln for ln in text.splitlines()
         if ln.startswith(("redaccion:", "redaccion_modelo:", "motivo:", "bloqueo:", "contra_argumento:"))]
if lines:
    print()
    print("  ── Redacción y contra-argumentos ──")
    for line in lines:
        print(f"  {line}")

print()
print("  ── Valla de honestidad ──")
print("  no hay edge demostrado · ejecución manual · apoyo a la decisión, no una estrategia validada")
print()
print("  Artefactos:")
print(f"    informe completo ..... journal/ops/{session}/daily_report.txt")
print(f"    fila del diario ...... journal/decisions/{session}.json")
print(f"    traza/manifiesto ..... journal/ops/{session}/manifest.json")
PY

echo
echo "PIPELINE COMPLETO — revisa el REPORTE FINAL de arriba."
