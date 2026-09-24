## Goal

One or two sentences on what should be true when this is done.

## Acceptance criteria

- [ ] 12-15 criteria for a normal task; up to 25 only if the task is in
      `CRITICAL_TASKS` / `GATE_TASKS`. One line per criterion, including the
      awkward ones.
- [ ] A statement you can check by looking at the result: a command
      (`uv run pytest ...`) or a file inspection, never prose.

## Out of scope

- Something that does not belong in this task, moved to #TASK-NUMBER

## Constraints

- Files this should stay inside
- Libraries to use
- Guidelines to follow
- Coverage floor: 90 % statements / 85 % branches of the new module, or a
  declared justification (95/90 only for CRITICAL_TASKS / GATE_TASKS)
- Goldens: no criterion pins the digest of a regenerable artifact. A digest
  golden checks format, self-consistency and determinism instead (#89, #95,
  #96). The only literals allowed are stable values: the document, the rows of
  a table, the feature specs.

Keep the body under ~12 KB. The engineer and the QA each read it twice, so
every extra kilobyte is paid twice per round.
