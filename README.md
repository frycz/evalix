# evalix

A small eval harness that tells you **which cases your prompt change broke**.

```
  score      0.136 → 0.136  (+0.000)   over 22 shared cases
    fixed  t11
    BROKE  t14
```

A flat score. One case fixed, one case broken. Without the second line you would
have called that "no change" and moved on.

That is the whole pitch. The score tells you *whether*; the diff tells you
*which*, and only the second one teaches you anything.

## The loop

> fixed case set → change **one** thing → measure → keep or revert

The usual way people practise prompt engineering is to tweak a prompt, look at
one output, decide it's better, and move on. That's not a skill, it's a feeling.
Everything here is built around the loop above.

## Install

```bash
pip install evalix        # or: uv add evalix
```

The default runner talks to Claude through [capix](https://pypi.org/project/capix/),
so `ANTHROPIC_API_KEY` in your environment or a `.env` file is all the setup
there is. Any other provider is a three-line function — see [Runners](#runners).

## Use it

A case file is JSONL, one case per line:

```json
{"id": "t01", "tag": "easy", "input": "I was charged twice.", "expected": "billing"}
{"id": "t02", "tag": "edge", "input": "Export CSV throws a TypeError.", "expected": "bug"}
```

```bash
evalix run --cases cases.jsonl --prompt prompts/v1.txt --scorer exact
```

```
  ✓ t01        1.00  want='billing' got='billing'
  ✗ t02        0.00  want='bug' got='how_to'

  score      0.727   (16/22 perfect)
  tokens     1534 in / 212 out   ≈ $0.0026

  0.636 → 0.727  (+0.091)   over 22 shared cases
    fixed  t09, t11, t14
    BROKE  t20

  worst 5:
    t20 (0.00) want='how_to' got='feature_request'
        → feature_request
```

Runs are saved to `runs/` and the next run of the same case file is diffed
against them automatically. To compare any two runs — v1 against v5, with three
experiments in between:

```bash
evalix compare v1-lazy v5-spec
```

Each argument is a run file or any substring of one. It prints the metadata side
by side, flags every field that differs, warns when more than one axis moved at
once, then gives the per-case changes, a per-tag breakdown, and the outputs
behind each regression.

## Spend nothing while you work

`--dry-run` sends nothing. It resolves the case file, the prompt and the scorer,
prints what each call would contain, and estimates the bill:

```bash
evalix run --cases cases.jsonl --prompt prompts/v1.txt --dry-run
```

```
  22 calls · ~3211 input tokens
  worst case 44000 output tokens (every case hitting max-tokens)
  estimated  $0.0032 … $0.2232   (input only … input + max output)

  dry run — nothing was sent, no run file written
```

Use it whenever you edit a prompt or a case file. A mangled `{input}` placeholder
or a prompt that grew tenfold shows up here for free.

## Scorers

A scorer turns one output into a number. It is the only task-specific part of
the harness, which is why it is the pluggable one — and the part most likely to
be silently wrong, which is why the built-ins are tested to the letter.

| `--scorer` | `expected` is | scores |
|---|---|---|
| `exact` | a string | 1 if equal after normalising case, whitespace, trailing punctuation |
| `contains` | a string | 1 if present as a substring |
| `not_contains` | a string | 1 if **absent** — the injection canary |
| `regex` | a pattern | 1 if it matches |
| `json_parse` | *(unused)* | 1 if the output parsed as JSON at all |
| `json_fields` | an object | the fraction of fields that match; unparseable scores 0 |
| `judge` | *(unused)* | a second model call against `--rubric`, 1–5 mapped onto 0…1 |
| `none` | *(unused)* | nothing — records the output for manual reading |

Any key you add to a case is passed through untouched, so a custom scorer can
read whatever its task needs:

```python
# score.py — canary absent AND the real job still done
def score(output, case):
    if case["expected"].lower() in output.lower():
        return 0.0, "LEAKED"
    anchors = case.get("must_contain", [])
    hit = [a for a in anchors if a.lower() in output.lower()]
    return len(hit) / len(anchors), "clean" if len(hit) == len(anchors) else "off-task"
```

```bash
evalix run --cases cases.jsonl --scorer custom --scorer-file score.py
```

A scorer that raises **stops the run** rather than scoring the case zero. A
broken scorer otherwise reports a clean `0.000` that looks exactly like a failing
prompt. `--keep-going` opts out.

Take a third argument to get a `Context` — with it, your scorer can call a model
itself, which is all a judge is:

```python
def score(output, case, ctx):
    verdict = ctx.runner(Request(messages=[Message("user", f"Grade this: {output}")]))
    ...
```

## Python API

The CLI is a thin wrapper over this, so nothing is terminal-only:

```python
from evalix import run

report = run(
    cases="cases.jsonl",
    prompt="prompts/v2.txt",
    scorer="json_fields",
    model="claude-haiku-4-5",
)

report.score          # 0.727
report.results        # list[Result] — score, note, output, latency, tokens
report.diff.broke     # ['t20']
report.cost_usd
print(report.render())
```

## Runners

A runner is any callable taking a `Request` and returning a `Response`. Nothing
in the core imports capix, so another provider is an argument, not a fork:

```python
from evalix import run, Response

def my_runner(request):
    reply = my_client.complete(system=request.system, prompt=request.text)
    return Response(text=reply, input_tokens=..., output_tokens=...)

run(cases="cases.jsonl", runner=my_runner, model="whatever-you-call-it")
```

It is also how the whole pipeline gets tested offline — every test in this repo
runs without an API key, because a fake runner is three lines.

**v1 is single-turn.** `Request.messages` is a list so that tools and extra turns
can arrive as a field rather than a new major version, but nothing sends more
than one message today.

## Flags

```
evalix run
  --cases FILE           JSONL, one case per line
  --prompt FILE          the prompt under test
  --scorer NAME          exact | contains | not_contains | regex | json_parse
                         | json_fields | judge | none | custom
  --scorer-file FILE     with --scorer custom: a .py defining score()
  --rubric FILE          with --scorer judge
  --judge-model ID       default claude-opus-5
  -m / --model ID        model under test
  -e / --effort LEVEL    low|medium|high|xhigh|max (thinking-capable models)
  -t / --max-tokens N    default 2000
  --placement system|user  where the prompt file goes
  --repeat N             run each case N times — consistency check
  --only-tag TAG         slice to one tag
  --limit N              first N cases only
  --label NAME           name this run in the diff output
  --workers N            parallel requests (default 8)
  --runs-dir DIR         default <project root>/runs, or $EVALIX_RUNS
  --keep-going           score a case zero when the scorer raises
  --dry-run              print what would be sent, send nothing
  --quiet / --show N     less per-case noise / how many failures to print

evalix compare OLD NEW [--all] [--show N] [--runs-dir DIR]
```

## Habits worth stealing

- **Change one thing per run.** Two changes and a flat score tells you nothing.
  `compare` warns when more than one axis moved.
- **Keep every prompt version.** `v1-lazy.txt`, `v2-spec.txt` — they're the log
  of what you learned, and you'll want to revert.
- **Read the failures, not the score.** The score tells you *whether*, the
  outputs tell you *why*.
- **Suspect your labels.** When a case won't budge, check whether your expected
  answer is actually right. Sometimes the model is and you aren't.
- **Check consistency before celebrating.** `--repeat 3`. A case that flips
  between runs isn't solved, it's lucky.

## Development

```bash
uv sync
uv run pytest -q
```

The whole suite is offline — no key, no tokens, nothing spent.

## Licence

MIT
