"""Cases: the fixed input set a prompt is measured against.

JSONL, one object per line. `expected` means whatever the chosen scorer says
it means — that looseness is deliberate and is what lets one harness score
classification, extraction and injection resistance without knowing anything
about them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESERVED = ("id", "input", "expected", "tag")


@dataclass(frozen=True)
class Case:
    """One case.

    Unknown keys land in `extra` and stay reachable through `get()` and `[]`.
    That is the cheapest extension point in the package: a custom scorer can
    read whatever fields its task needs without the harness knowing they exist.
    """

    id: str
    input: str = ""
    expected: Any = None
    tag: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any], *, line: int | None = None) -> Case:
        where = f"case at line {line}" if line is not None else "case"
        if not isinstance(row, dict):
            raise ValueError(f"{where} is a {type(row).__name__}, not a JSON object")  # noqa: TRY004
        if "id" not in row:
            raise ValueError(
                f"{where} has no 'id'. Ids are the diff key — without "
                "them every case collides and fixed/broke is meaningless."
            )
        return cls(
            id=str(row["id"]),
            input=row.get("input", ""),
            expected=row.get("expected"),
            tag=row.get("tag"),
            extra={k: v for k, v in row.items() if k not in RESERVED},
        )

    def get(self, key: str, default: Any = None) -> Any:
        if key in RESERVED:
            return getattr(self, key)
        return self.extra.get(key, default)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        return self.get(key, _MISSING) is not _MISSING

    def replace(self, **changes: Any) -> Case:
        from dataclasses import replace

        return replace(self, **changes)


_MISSING = object()


def load_cases(
    path: str | Path,
    *,
    only_tag: str | None = None,
    limit: int | None = None,
    repeat: int = 1,
) -> list[Case]:
    """Read a case file and apply the run-shaping filters.

    Order matters: repeat expands, then limit truncates. `--limit 2 --repeat 3`
    is two calls, not six — the limit is a spending cap, so it has to be last.
    """
    path = Path(path)
    cases: list[Case] = []
    # Line numbers count every physical line, comments and blanks included, so
    # they match what an editor shows.
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON at line {number}: {exc.msg}") from None
        try:
            cases.append(Case.from_dict(row, line=number))
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from None

    if only_tag:
        cases = [c for c in cases if c.tag == only_tag]
    if repeat > 1:
        cases = [c.replace(id=f"{c.id}#{r + 1}") for c in cases for r in range(repeat)]
    if limit:
        cases = cases[:limit]

    duplicates = {c.id for c in cases if sum(1 for o in cases if o.id == c.id) > 1}
    if duplicates:
        raise ValueError(
            f"duplicate case ids: {', '.join(sorted(duplicates))}. "
            "Ids must be unique or the diff cannot line runs up."
        )
    return cases


def case_key(path: str | Path, root: Path) -> str:
    """A stable key identifying *which case file* a run was against.

    Runs are diffed against the previous run of the same key, so this must not
    depend on the directory you happened to run from. It is the path relative
    to the project root, which is why the root is discovered rather than
    assumed.
    """
    import re

    resolved = Path(path).resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        rel = Path(resolved.name)
    return re.sub(r"[^A-Za-z0-9._-]", "-", str(rel.with_suffix("")))
