"""Where runs live on disk.

Run files are the memory that makes a diff possible, so they are plain JSON in
a directory you can read, grep and delete — not a database and not a cloud
service. The filename is a human-readable label; everything a tool needs is
inside the file.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

from evalix.scoring import Run

MARKERS = (".git", "pyproject.toml")


def project_root(start: Path | None = None) -> Path:
    """Walk up for a project marker, so the run key does not depend on cwd.

    Running an exercise from its own subdirectory has to produce the same key
    as running it from the repo root, or the diff silently compares nothing.
    """
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if any((directory / m).exists() for m in MARKERS):
            return directory
    return here


def runs_dir(explicit: str | Path | None = None, root: Path | None = None) -> Path:
    """--runs-dir, then $EVALIX_RUNS, then <project root>/runs."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("EVALIX_RUNS")
    if env:
        return Path(env).expanduser().resolve()
    return (root or project_root()) / "runs"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", text)


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(self, run: Run) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        meta = run.meta
        stamp = meta.get("timestamp") or dt.datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
        name = "__".join(
            _slug(str(part))
            for part in (meta.get("case_key", "cases"), meta.get("label", "run"), meta.get("model", "model"), stamp)
        )
        body = json.dumps(run.to_dict(), indent=2, ensure_ascii=False)
        # Exclusive create: two runs in the same second get `-2`, `-3` rather
        # than one silently replacing the other.
        path, n = self.root / f"{name}.json", 1
        while True:
            try:
                with path.open("x", encoding="utf-8") as f:
                    f.write(body)
                return path
            except FileExistsError:
                n += 1
                path = self.root / f"{name}-{n}.json"

    def load(self, path: str | Path) -> Run:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("meta", {}), dict):
            raise ValueError(f"{Path(path).name} is not an evalix run file")  # noqa: TRY004 - bad file
        return Run.from_dict(data)

    def all_runs(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime)

    def previous(self, case_key: str, exclude: Path | None = None) -> Run | None:
        """The most recent earlier run of the same case file.

        Keyed on what is *inside* the file, not on the filename — the filename
        is a label and nothing should parse it.
        """
        for path in reversed(self.all_runs()):
            if exclude and path == exclude:
                continue
            try:
                run = self.load(path)
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                # Anything else that happens to be JSON in runs/ is not ours to
                # crash on. ValueError covers bad JSON and bad UTF-8 too.
                continue
            if run.meta.get("case_key") == case_key:
                run.meta.setdefault("path", str(path))
                return run
        return None

    def resolve(self, needle: str | Path) -> tuple[Path, Run]:
        """A path, or any substring of a run filename.

        Repeats of the same experiment (same case file, label and model) resolve
        to the newest. Substrings that hit different experiments — `v1` matching
        both `v1` and `v10` — are an error rather than a guess, because a diff
        against the wrong baseline looks exactly like a real one.
        """
        path = Path(needle)
        if path.is_file():
            return path, self.load(path)
        matches: list[tuple[Path, Run]] = []
        for candidate in self.all_runs():
            if str(needle) not in candidate.name:
                continue
            try:
                matches.append((candidate, self.load(candidate)))
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        if not matches:
            raise FileNotFoundError(f"no run file matching {needle!r} in {self.root}")

        experiments = {
            (run.meta.get("case_key"), run.meta.get("label"), run.meta.get("model"))
            for _, run in matches
        }
        if len(experiments) > 1:
            names = "\n  ".join(p.name for p, _ in matches[-10:])
            raise ValueError(
                f"{needle!r} matches {len(experiments)} different runs — "
                f"use a longer substring or a path:\n  {names}"
            )
        return matches[-1]
