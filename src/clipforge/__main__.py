"""ClipForge command-line entry point.

    python -m clipforge run <url_or_path> [--no-llm]
    python -m clipforge resume <job_id> [--no-llm]

``run`` starts a fresh job (download -> transcribe -> analyze -> cut) and writes
``data/outputs/{job_id}/results.json``. ``resume`` continues an existing job from
its first incomplete stage using the checkpoint in ``state.json``.

``--no-llm`` skips the LLM analysis and instead emits a single fake candidate
covering seconds 0-3, so the whole pipeline can be exercised offline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from .config import Config
from .pipeline import PipelineError, resume_job, run_new


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipforge", description="Local-first AI video clipper.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Process a new URL or local video file end-to-end.")
    p_run.add_argument("source", help="YouTube URL or path to a local video file.")
    p_run.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM analysis; emit one fake 0-3s candidate (offline testing).",
    )

    p_resume = sub.add_parser("resume", help="Continue a job from its first incomplete stage.")
    p_resume.add_argument("job_id", help="The id of the job to resume.")
    p_resume.add_argument(
        "--no-llm",
        action="store_true",
        help="Force offline analysis when resuming into the analyze stage.",
    )

    return parser


def _print_summary(results: dict[str, Any]) -> None:
    """Print a short human-readable summary and the results.json location."""
    clips = results.get("clips", [])
    print(f"\nJob {results['job_id']} complete — {len(clips)} clip(s):")
    for clip in clips:
        print(
            f"  [{clip.get('score')}] {clip.get('start')}-{clip.get('end')}s"
            f"  {clip.get('title') or '(untitled)'}"
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    cfg = Config.load()

    try:
        if args.command == "run":
            results = run_new(args.source, cfg, no_llm=args.no_llm)
        elif args.command == "resume":
            # Only override the saved setting when the flag was actually passed.
            override = True if args.no_llm else None
            results = resume_job(args.job_id, cfg, no_llm=override)
        else:  # pragma: no cover - argparse enforces a valid subcommand
            parser.error(f"unknown command {args.command!r}")
            return 2
    except PipelineError as exc:
        print(
            f"\nStage {exc.stage!r} failed: {exc.message}\nResume with: {exc.resume_command}",
            file=sys.stderr,
        )
        return 1

    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
