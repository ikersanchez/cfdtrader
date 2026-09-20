Commands

- 'uv sync' - install dependencies
- 'uv run pytest' - the whole suite
- 'uv run pytest tests/test_home.py' - one test file
- 'unshare -rn uv run pytest' - the whole suite without network access,
  only for the criteria that ask for the no-network gate

Rules

- Dependencies are added in 'pyproject.toml'. Do not add one without asking


Documents

- '_docs/process.md' - how work is organized
