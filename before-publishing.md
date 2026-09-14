# Before publishing to PyPI

Review done on 2026-09-14 against commit `90a804a`.

**Verdict:** close. Fix the blockers below and it's a solid 0.1.0. PyPI versions can't be
overwritten, so it's worth doing first.

## Already in good shape

- **Tests:** all 103 pass, offline, on Python 3.14 and 3.11 (the minimum).
- **Build:** `uv build` produces a clean sdist and wheel, `twine check` passes, and the README renders.
- **Name:** `evalix` is free on PyPI. `capix>=0.1.2` is published and has everything the runner calls.
- **Design:** one clear place to plug in other providers, output text kept separate from the
  logic, matching API and CLI, and docs that explain why things work the way they do.

## Blockers

### Bugs (all reproduced)

- [ ] **"A scorer that raises stops the run" isn't true.** `api.py:215`: leaving the
  `ThreadPoolExecutor` block waits for every queued case to finish. With 40 cases and a
  scorer that fails on the first one, all 40 API calls were still made before `ScorerError`
  came up. Ctrl+C behaves the same way, so interrupting a big run keeps spending money.
  Fix: on an error, cancel the pending cases (`pool.shutdown(cancel_futures=True)`) and re-raise.
- [ ] **The judge score isn't range-checked.** `scorers/judge.py:74`: a reply of
  `"score": 10` becomes 2.25 and `0` becomes -0.25, which quietly skews the mean. Clamp it,
  or treat anything outside 1-5 as unparseable.
- [ ] **One non-run JSON file in `runs/` crashes every run.** `store.py:84`: a file
  containing `[1,2]` raises `AttributeError`, which isn't in the list of caught errors.
- [ ] **Two runs in the same second overwrite each other.** `store.py:56`: the run filename
  only has one-second precision (2 runs produced 1 file). Add microseconds or a short
  random suffix.
- [ ] **Some CLI errors print a traceback.** `cli.py:151`: a scorer file with no `score()`
  raises `AttributeError`, a non-`.py` file raises `ImportError`, and neither is caught.
  Syntax errors in the user's file aren't caught either.
- [ ] **Error line numbers are wrong.** `cases.py:91` counts only non-empty, non-comment
  lines, so a file starting with a comment reports "line 2" for what is really line 4. Bad
  JSON also shows up without the file name or line number.

### Packaging

- [ ] **Add a `LICENSE` file.** The MIT licence requires shipping its text. Also set
  `license-files = ["LICENSE"]` in `pyproject.toml`.
- [ ] **Keep `atm.json` and `uv.lock` out of the sdist:**
  ```toml
  [tool.hatch.build.targets.sdist]
  include = ["src", "tests", "README.md", "LICENSE"]
  ```
- [ ] **Keep the version number in one place.** It's in both `pyproject.toml` and
  `__init__.py`. Use `importlib.metadata.version("evalix")` or hatch's dynamic version.
- [ ] **Complete the classifiers:** Python versions (3.11-3.14), the licence,
  `Operating System`, and `Typing :: Typed`.

## Worth fixing (less urgent)

- [ ] **Cost is under-reported.** Judge calls add tokens and cost that never reach
  `tokens`/`cost_usd`. The default judge is `claude-opus-5` with 4000 max tokens per case,
  which could easily cost more than the run itself.
- [ ] **Judge failures stop the whole run.** A temporary API error during a judge call
  becomes `ScorerError`. Consider recording it like a runner error.
- [ ] **`extract_json` misses some valid JSON.** If there's text around it and a `}` inside a
  string (e.g. `Sure: {"a": "}"}`), it fails, because bracket matching doesn't account for
  strings.
- [ ] **`compare` matches by substring and quietly takes the newest file,** so `v1` also
  matches `v10`. It should error when more than one file matches.
- [ ] **The judge never sees `expected`,** so you can't grade against a reference answer.
- [ ] **Windows:** `read_text()` has no `encoding="utf-8"`, and the output uses `✓ → ≈ ⚠`.
  Non-ASCII case files and old consoles will break.
- [ ] **README fixes:**
  - The `ctx` example doesn't import `Request`/`Message`.
  - The custom-scorer example divides by zero when `must_contain` is missing.
  - `report.diff.broke` fails on the first run because `diff` is `None` then.
- [ ] **Public API:** `compare()` returns a 5-tuple, which is awkward to change later. A
  small dataclass would be better, and now is the cheapest time to switch.
- [ ] **Lint:** ruff reports 22 issues, 18 auto-fixable (unused imports, import order).
  `pyproject.toml` sets ruff's line length, but ruff isn't in the dev dependencies.
- [ ] **No CI.** Add a GitHub Actions matrix for 3.11-3.14 running pytest and ruff, plus
  trusted publishing to PyPI.

## Release steps

1. Fix the blockers above.
2. Publish to TestPyPI: `uv build && uv publish --publish-url https://test.pypi.org/legacy/`
3. Do a real install in a clean environment and try both `evalix run --dry-run` and a real run.
4. Tag the release and publish to PyPI.
