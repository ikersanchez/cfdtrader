- Tasks are github issues, one at a time
- Read the acceptance criteria before starting and before closing
- Commit regurarly


Roles

- PM - grooms a task before anyone implements it, follows _docs/team/pm.md
- Engineer - implements one groomed task, follows _docs/team/software-engineer.md
- QA - checks the result against the acceptance criteria, follows _docs/team/qa-engineer.md


Orchestrator

The main session is the orchestrator. It launches the PM, the engineer
and QA as subagents. It does not groom, implement or test itself.


QA profiles

QA has two profiles. Pick one from the issue labels; do not invent a third.

| Profile | When | Budget |
| --- | --- | --- |
| `full` | Issues in `CRITICAL_TASKS` / `GATE_TASKS` (`setup_github.py`) | Whole suite once, independent recomputation of the numbers that decide the verdict, up to ~10 probes |
| `light` | Everything else | Module tests + whole suite once, ~5 probes on the criteria the engineer claims as measured |

The verdict is PASS or FAIL in both. What changes is how much evidence
gets written down, never how honest the verdict is.


Lifecycle

1. Pick the next open issue from the backlog
2. PM grooms it
3. Engineer implements it, in one run and one stage
4. QA verifies it, with the profile its labels select
5. On FAIL, back to step 3 with the QA comment as input (light round)
6. On PASS, close the issue
7. Repeat until the backlog is empty

Step 2 of the next issue may run while step 4 of the current one is still
running: grooming edits the issue body and QA is read-only, so they touch
nothing in common. What waits is the groomed issue, not the orchestrator.

Rules

- Do not skip step 2
- The engineer does not close the issue
- QA does not fix the code, only outputs PASS or FAIL
- The orchestrator closes the issue only after QA outputs PASS
- The engineer has no stages: module and tests land in one run. Stages are
  a recovery mechanism, not a plan.
- The comment is not optional and it is not last. If the engineer is running
  out of context, commit and comment first: a missing comment costs a whole
  extra round (measured: ~10 min of a ~49 min cycle).
- The whole suite is run once per phase; per-file runs while iterating.
  `unshare -rn` only when the criteria ask for the no-network gate.
- Coverage floors are 90 % statements / 85 % branches for the new module, or
  a declared justification. Never open a round just to cross the line.
- A criterion that pins the digest of a regenerable artifact is a grooming FAIL,
  not a style detail. Three instances so far (#89, #95, #96), and all three
  broke `uv run pytest -q` in a later task.


Budget

Measured on the #69 cycle (2026-09-19), before these rules: PM ~7 min,
engineer ~29 min across two rounds (60 % of the cycle), QA ~12 min, whole
cycle ~49 min. Target with these rules: ~25 min per issue.

When a phase goes over, the fix is fewer criteria (PM), one round
(engineer) or a shorter verdict (QA) - never a weaker check.


Regeneration after an ingest (#135)
-----------------------------------

The store grows with every ingest (`cfdtrader.data.market`, `.macro`,
`.news`, `.earnings`). The Fase 0/1/2 artifacts are *derived* from the store,
so an ingest that changes the measured window invalidates them, and the tests
must not pin **absolute** store counts. Two rules:

1. **Relative, never absolute.** Assertions about the live store use the
   **identities** the report already publishes (e.g. `clean = stale_open +
   no_forecast + labelled`) and **deltas** against the previous artifact, not
   totals like `raw_market_daily_rows == 5460`. A literal is only allowed for a
   **stable** value: the document, a table's rows, the feature specs, a
   pre-registered plan parameter.
2. **Regenerate in order** after an ingest (`labels`/`volatility_forecast`
   first, then `backtest_report` -> `phase1_report` -> `baseline_report` ->
   `model_comparison` -> `pipeline_report` -> `gate_sweep`/`phase2_*`), passing
   `--previous-artifact` where the CLI accepts it. Regenerating the published
   artifacts in `data/` does **not** change the git tree (`data/` is ignored).

The observation's daily start is therefore: **ingest, then regenerate.**
