"""`evalix run` and `evalix compare`.

A thin wrapper over evalix.api — every flag here maps to a keyword argument
there, so nothing the terminal can do is unavailable from Python.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evalix import api, report
from evalix.prompt import PLACEMENTS, PromptError
from evalix.runners import MissingCredentials
from evalix.scorers import NAMES, ScorerFileError, load_custom


def _load_dotenv() -> None:
    """Applications may read .env; libraries may not. This is the application."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(find_dotenv(usecwd=True))


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cases", required=True, help="JSONL case file, one case per line.")
    p.add_argument("--prompt", help="File holding the prompt under test.")
    p.add_argument("--scorer", default="exact", choices=[*NAMES, "custom"], help="How to score.")
    p.add_argument("--scorer-file", help="With --scorer custom: a .py defining score().")
    p.add_argument("--rubric", help="With --scorer judge: the rubric file.")
    p.add_argument("--judge-model", default="claude-opus-5", help="Model that grades.")
    p.add_argument("-m", "--model", help="Model under test.")
    p.add_argument("-e", "--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("-t", "--max-tokens", type=int, default=2000)
    p.add_argument("--placement", choices=PLACEMENTS, default="system")
    p.add_argument("--repeat", type=int, default=1, help="Run each case N times.")
    p.add_argument("--limit", type=int, help="Only the first N cases.")
    p.add_argument("--only-tag", help="Only cases with this tag.")
    p.add_argument("--workers", type=int, default=8, help="Parallel in-flight requests.")
    p.add_argument("--label", help="Name this run in the diff output.")
    p.add_argument("--show", type=int, default=5, help="How many failing outputs to print.")
    p.add_argument("--quiet", action="store_true", help="Summary only.")
    p.add_argument("--runs-dir", help="Where run files live (default: <project root>/runs).")
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="Score a case zero when the scorer raises, instead of stopping.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve everything, print what would be sent, and stop. No API calls.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalix",
        description="Score a prompt against a case file, and see which cases you broke.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a case file and score it.")
    _add_run_args(run_parser)

    cmp_parser = sub.add_parser("compare", help="Diff any two saved runs.")
    cmp_parser.add_argument("old_run", help="Baseline run file, or a substring of its name.")
    cmp_parser.add_argument("new_run", help="Run to compare against it.")
    cmp_parser.add_argument("--all", action="store_true", help="List unchanged cases too.")
    cmp_parser.add_argument("--show", type=int, default=5, help="Regressions to print in full.")
    cmp_parser.add_argument("--runs-dir", help="Where run files live.")

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    if args.scorer == "custom" and not args.scorer_file:
        print("error: --scorer custom requires --scorer-file", file=sys.stderr)
        return 2
    if args.scorer == "judge" and not args.rubric:
        print("error: --scorer judge requires --rubric", file=sys.stderr)
        return 2
    scorer = load_custom(args.scorer_file) if args.scorer == "custom" else args.scorer

    label = args.label or (Path(args.prompt).stem if args.prompt else "no-prompt")
    model = args.model or "(default)"
    print(f"model={model} · prompt={label} · scorer={args.scorer}\n")

    if args.dry_run:
        est = api.estimate(
            args.cases,
            args.prompt,
            scorer=args.scorer,
            model=args.model,
            placement=args.placement,
            max_tokens=args.max_tokens,
            only_tag=args.only_tag,
            limit=args.limit,
            repeat=args.repeat,
        )
        print(est.render(quiet=args.quiet))
        return 0

    report_ = api.run(
        args.cases,
        args.prompt,
        scorer=scorer,
        model=args.model,
        placement=args.placement,
        max_tokens=args.max_tokens,
        effort=args.effort,
        only_tag=args.only_tag,
        limit=args.limit,
        repeat=args.repeat,
        workers=args.workers,
        label=args.label,
        scorer_config={"rubric": args.rubric, "judge_model": args.judge_model},
        keep_going=args.keep_going,
        runs=args.runs_dir,
        on_result=None if args.quiet else lambda r: print(report.render_case_line(r)),
    )
    print(report_.render(show=args.show))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    comparison = api.compare(args.old_run, args.new_run, runs=args.runs_dir)
    print(comparison.render(show=args.show, show_all=args.all))
    return 0


def _tolerate_narrow_consoles() -> None:
    """Print `?` instead of crashing where the console can't encode ✓ → ≈ ⚠.

    A Windows console piped to a file falls back to a legacy code page; one
    unencodable glyph otherwise kills the run after the money is spent.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def main(argv: list[str] | None = None) -> int:
    _tolerate_narrow_consoles()
    _load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        return _cmd_compare(args)
    except MissingCredentials as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (PromptError, ScorerFileError, api.ScorerError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
