"""El bloque final de `run_pipeline.sh`: los dos ejes de la decisión (#161).

El pipeline **no** se ejecuta: se extrae el heredoc `uv run python - <<'PY' … PY` del script y se
lanza como un proceso aparte, con un diario sintético en `tmp_path`. El diario real de la sesión es
inmutable (`JournalRewriteError`) y esta prueba no lo toca.

Lo que se fija aquí son los dos ejes que el informe tiene que mostrar **por separado**:

- eje 1 — estado y veto del overlay (la decisión: `NOTHING` por veto, o el lado del gate);
- eje 2 — el sesgo del lote (`tilt`), que es **descripción del lote en una escala declarada**, no
  una previsión ni una señal de edge, y que no tiene autoridad sobre la decisión.

Y la valla de honestidad: con el sesgo ausente el informe dice `sin tilt` y **no** imprime un
`net_pct` que nadie midió.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PIPELINE: Final[Path] = REPO_ROOT / "run_pipeline.sh"

#: Cabecera y cierre del heredoc que imprime el REPORTE FINAL DE LA DECISIÓN.
HEREDOC_OPEN: Final[str] = "uv run python - <<'PY'"
HEREDOC_CLOSE: Final[str] = "\nPY\n"

SESSION: Final[str] = "2026-11-02"

#: Las once claves que publica el sesgo del lote (#161).
TILT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "net_pct",
        "clamped",
        "ratio",
        "side",
        "weighted",
        "n_events",
        "n_neutral",
        "consensus",
        "horizon",
        "dominant",
        "capped_events",
    }
)

#: Un sesgo alcista y recortado: dos `high` alcistas suman 20 pp y el ajuste se acota a 10.
TILT: Final[dict[str, object]] = {
    "net_pct": 20.0,
    "clamped": True,
    "ratio": 1.0,
    "side": "long",
    "weighted": {"bullish": 20.0, "bearish": 0.0, "neutral": 0.0},
    "n_events": 2,
    "n_neutral": 0,
    "consensus": 1.0,
    "horizon": {"intraday": 1, "days": 1},
    "dominant": {
        "event_type": "monetary_policy",
        "sentiment": "bullish",
        "magnitude": "high",
        "rationale": "recorte de tipos declarado",
    },
    "capped_events": 2,
}


def _report_script() -> str:
    """El cuerpo del heredoc final, verbatim (sin la cabecera `uv run` ni el `PY` de cierre)."""
    text = PIPELINE.read_text(encoding="utf-8")
    assert text.count(HEREDOC_OPEN) == 1, "el bloque final del pipeline debe ser único"
    start = text.index(HEREDOC_OPEN) + len(HEREDOC_OPEN)
    end = text.index(HEREDOC_CLOSE, start)
    return text[start:end]


def _decision(*, direction: str = "long", report_text: str = "") -> dict[str, object]:
    """Una fila mínima de `journal.decisions`: solo las claves que el informe lee de verdad."""
    return {
        "status": "recommendation",
        "direction": direction,
        "prob_up_calibrated": 0.5607,
        "tier": "B",
        "llm_overlay": "applied",
        "report_text": report_text,
    }


def _signal(
    *,
    tilt: object,
    state: str = "applied",
    direction: str = "nothing",
    veto: bool = False,
    veto_reason: str | None = None,
    include_tilt: bool = True,
) -> dict[str, object]:
    evidence: dict[str, object] = {
        "state": state,
        "direction": direction,
        "reasons": [] if veto_reason is None else [veto_reason],
        "gate_direction_with_overlay": direction,
    }
    if include_tilt:
        evidence["tilt"] = tilt
    return {
        "trade_date": SESSION,
        "agent": "news",
        "prob_up": 0.4607,
        "confidence": None,
        "veto": veto,
        "veto_reason": veto_reason,
        "evidence": evidence,
    }


def _run(
    tmp_path: Path, *, decision: Mapping[str, object], signal: Mapping[str, object] | None
) -> str:
    """Lanza el heredoc con un diario sintético colgando de `tmp_path` (nunca el real)."""
    decisions = tmp_path / "journal" / "decisions"
    decisions.mkdir(parents=True)
    (decisions / f"{SESSION}.json").write_text(json.dumps(decision), encoding="utf-8")
    if signal is not None:
        signals = tmp_path / "journal" / "agent_signals"
        signals.mkdir(parents=True, exist_ok=True)
        (signals / f"{SESSION}__news.json").write_text(json.dumps(signal), encoding="utf-8")

    environment = {**os.environ, "SESSION": SESSION}
    completed = subprocess.run(
        [sys.executable, "-"],
        input=_report_script(),
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _eje1(output: str) -> str:
    """El bloque del estado y el veto: entre su cabecera y la del sesgo del lote."""
    start = output.index("Las dos decisiones (eje 1")
    end = output.index("Las dos decisiones (eje 2")
    return output[start:end]


# ─────────────────────────────────────────────────────────────────────────────
# A11 · El sesgo del lote se muestra con composición, horizonte y dominante
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_tilt_is_printed_with_composition_horizon_and_dominant(tmp_path: Path) -> None:
    output = _run(tmp_path, decision=_decision(), signal=_signal(tilt=TILT))

    for token in (
        "estado del overlay",
        "side=",
        "net_pct=",
        "bullish",
        "bearish",
        "neutral",
        "intraday",
        "days",
        "dominant",
        "capped_events",
    ):
        assert token in output, f"falta {token!r} en el informe del sesgo"
    assert "recorte de tipos declarado" in output, "el rationale del dominante se imprime entero"
    assert "recortado" in output, "clamped=True se marca"
    # La valla de honestidad, en prosa y en la salida.
    assert "descripción del lote" in output
    assert "no una previsión" in output


def test_a11_a_tilt_that_is_not_clamped_does_not_say_recortado(tmp_path: Path) -> None:
    sin_recorte = {
        **TILT,
        "net_pct": 2.5,
        "clamped": False,
        "ratio": 0.25,
        "weighted": {"bullish": 5.0, "bearish": 2.5, "neutral": 2.5},
        "n_neutral": 1,
        "consensus": 0.5,
    }
    output = _run(tmp_path, decision=_decision(), signal=_signal(tilt=sin_recorte))

    assert "net_pct=2.5" in output
    assert "recortado" not in output, "sin recorte no se declara un recorte que no hubo"


# ─────────────────────────────────────────────────────────────────────────────
# A12 · Sin sesgo el informe lo declara: `sin tilt`, nunca un `0` inventado
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_sin_tilt_when_the_signal_declares_it_null(tmp_path: Path) -> None:
    output = _run(tmp_path, decision=_decision(), signal=_signal(tilt=None))

    assert "sin tilt" in output
    assert "net_pct" not in output, "sin tilt no se imprime un sesgo que nadie midió"


def test_a12_sin_tilt_when_the_key_is_absent(tmp_path: Path) -> None:
    output = _run(tmp_path, decision=_decision(), signal=_signal(tilt=None, include_tilt=False))

    assert "sin tilt" in output
    assert "net_pct" not in output


def test_a12_sin_tilt_when_the_overlay_is_disabled(tmp_path: Path) -> None:
    output = _run(
        tmp_path,
        decision=_decision(),
        signal=_signal(tilt=None, state="disabled_error", direction=""),
    )

    assert "sin tilt" in output
    assert "net_pct" not in output


# ─────────────────────────────────────────────────────────────────────────────
# A13 · El bloqueo explica la decisión, y los dos ejes no se confunden
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_bloqueo_line_lives_inside_the_two_decisions_block(tmp_path: Path) -> None:
    output = _run(
        tmp_path,
        decision=_decision(report_text="bloqueo: 10:tier_no_autorizado\n"),
        signal=_signal(tilt=TILT),
    )

    assert "bloqueo: 10:tier_no_autorizado" in _eje1(output), (
        "el bloqueo que explica la decisión va dentro del bloque de las dos decisiones"
    )


def test_a13_los_ejes_are_distinct_with_origin_long_and_overlay_nothing_by_veto(
    tmp_path: Path,
) -> None:
    reason = "monetary_policy/high: una frase"
    output = _run(
        tmp_path,
        decision=_decision(direction="long"),
        signal=_signal(tilt=TILT, direction="nothing", veto=True, veto_reason=reason),
    )
    eje1 = _eje1(output)

    assert "Recomendación de origen (sin overlay)" in eje1
    assert "Decisión final (con overlay)" in eje1
    assert "LONG" in eje1, "el origen conserva su lado"
    assert "NOTHING (veto del overlay)" in eje1, "el overlay vetado no invierte: anula"
    assert reason in eje1, "el motivo del veto se declara"
    eje2 = output[output.index("Las dos decisiones (eje 2") :]
    assert "side=long" in eje2, "el sesgo se publica en su propio eje, pese al veto"
