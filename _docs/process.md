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


Budget

Measured on the #69 cycle (2026-09-19), before these rules: PM ~7 min,
engineer ~29 min across two rounds (60 % of the cycle), QA ~12 min, whole
cycle ~49 min. Target with these rules: ~25 min per issue.

When a phase goes over, the fix is fewer criteria (PM), one round
(engineer) or a shorter verdict (QA) - never a weaker check.
