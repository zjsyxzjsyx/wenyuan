from pathlib import Path

# pyproject sets --basetemp=.cache/pytest (the system temp dir is not writable on the dev machine);
# .cache is gitignored, so a fresh checkout (CI) lacks the parent pytest expects to exist.
Path(".cache").mkdir(exist_ok=True)
