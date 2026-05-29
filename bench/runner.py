"""
Benchmark runner (Track D phase 3) — generalizes the Track C runner into the
calibration-data harness.

Runs the agent over `bench/problems.py` for N reps each, against local Ollama,
persisting memory across the *entire* run (predictions must surface and replay
across reps and related problems to accumulate verdicts). For every task run it
appends one JSONL record carrying the replay verdicts — the raw material the
phase-4 calibration analysis (`bench/analyze.py`) consumes.

Usage:

    python -m bench.runner --smoke                 # 3 problems x 2 reps shake-out
    python -m bench.runner --reps 5                 # full run
    python -m bench.runner --reps 5 --limit 10 --docker-image crucible-runtime

Pre-reqs (see docs/NEXT_STEPS.md):
  - 3.12 venv with sentence-transformers installed (otherwise memory recall and
    the "memory helped" analysis are no-ops).
  - Prebaked `crucible-runtime` Docker image (./docker/build.sh) so container
    setup doesn't dominate wall-clock.
  - Ollama running locally with the chosen model pulled.

Memory is NEVER reset between problems or reps — that is the whole point.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Module-level imports so import errors surface immediately, not mid-run.
from agent.core import SelfImprovingAgent
from agent.llm_clients import OllamaClient, OpenAIClient, AnthropicClient
from agent.models import AgentConfig, LoopConfig, IterationState

from bench.problems import PROBLEMS, ProblemSpec


SMOKE_PROBLEMS = 3
SMOKE_REPS = 2


# ---------------------------------------------------------------------------
# LLM factory (Ollama-first, mirroring the Track C runner).
# ---------------------------------------------------------------------------

def make_llm(provider: str, model: Optional[str]):
    if provider == "ollama":
        import os
        return OllamaClient(
            model=model or os.getenv("AGENT_OLLAMA_MODEL", "qwen2.5-coder:7b")
        )
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini")
    if provider == "anthropic":
        return AnthropicClient(model=model or "claude-3-haiku-20240307")
    raise ValueError(f"Unknown LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Per task-run execution.
# ---------------------------------------------------------------------------

async def run_one(
    agent: SelfImprovingAgent,
    problem: ProblemSpec,
    *,
    rep: int,
    run_index: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one (problem, rep) and return a structured JSONL record.

    Retrieval is snapshotted *before* solve() — the agent re-runs it internally
    during planning but doesn't expose the result; re-querying here is cheap
    (embeddings hit the same in-memory cache) and gives us a record of what the
    persisted memory surfaced for this run. `state.replay_report` is the headline
    payload: the confirmed/falsified/off-topic verdicts from success-time replay.
    """
    similar_solutions = await agent.long_term_memory.find_similar_solutions(
        problem.goal, k=5
    )
    relevant_learnings = await agent.long_term_memory.find_relevant_learnings(
        problem.goal, k=5
    )
    relevant_predictions: List[Dict[str, Any]] = []
    pred_mem = getattr(agent, "prediction_memory", None)
    if pred_mem is not None:
        relevant_predictions = await pred_mem.find_relevant(problem.goal, k=5)

    if verbose:
        print(f"\n{'#' * 60}")
        print(f"# {problem.id} rep {rep} (run {run_index}, {problem.category})")
        print(f"# patterns={len(similar_solutions)} "
              f"learnings={len(relevant_learnings)} "
              f"predictions={len(relevant_predictions)}")
        print(f"{'#' * 60}")

    start = time.perf_counter()
    exc_text: Optional[str] = None
    state: Optional[IterationState] = None
    try:
        state = await agent.solve(goal=problem.goal, task_id=f"{problem.id}-r{rep}")
    except Exception:
        exc_text = traceback.format_exc()
    elapsed = time.perf_counter() - start

    return {
        "problem_id": problem.id,
        "rep": rep,
        "run_index": run_index,
        "category": problem.category,
        "function_name": problem.function_name,
        "goal": problem.goal,
        "elapsed_seconds": round(elapsed, 2),
        "status": state.status.value if state else "exception",
        "iterations": state.iteration if state else None,
        "exception": exc_text,
        "retrieved_patterns": [
            {
                "id": p.get("id"),
                "similarity": p.get("similarity"),
                "score_breakdown": p.get("score_breakdown"),
                "project_type": p.get("project_type"),
                "source_goal": p.get("goal") or p.get("source_goal"),
            }
            for p in (similar_solutions or [])
        ],
        "retrieved_learnings": [
            {
                "lesson": (l.get("lesson") or "")[:200],
                "source_task_id": l.get("source_task_id"),
                "similarity": l.get("similarity"),
            }
            for l in (relevant_learnings or [])
        ],
        "retrieved_predictions": [
            {
                "id": pr.get("id"),
                "trigger_input": pr.get("trigger_input"),
                "predicted_error_type": pr.get("predicted_error_type"),
                "confidence": pr.get("confidence"),
                "similarity": pr.get("similarity"),
                "times_tested": pr.get("times_tested"),
                "times_confirmed": pr.get("times_confirmed"),
            }
            for pr in (relevant_predictions or [])
        ],
        # The phase-2 ReplayReport.to_dict() (confirmed/falsified/off-topic).
        "replay_report": state.replay_report if state else None,
    }


# ---------------------------------------------------------------------------
# Batch loop — takes a *pre-built* agent so it stays unit-testable without a
# real LLM / Docker. Memory is never reset between iterations.
# ---------------------------------------------------------------------------

async def run_batch(
    agent: SelfImprovingAgent,
    problems: List[ProblemSpec],
    *,
    reps: int,
    out_path: Path,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run every (problem, rep), appending one JSONL line per run.

    The same `agent` (and therefore the same persisted memory) is reused for
    every run — predictions and patterns accumulate across the whole batch.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    run_index = 0
    with open(out_path, "w") as fp:
        for problem in problems:
            for rep in range(1, reps + 1):
                record = await run_one(
                    agent, problem, rep=rep, run_index=run_index, verbose=verbose
                )
                run_index += 1
                results.append(record)
                fp.write(json.dumps(record) + "\n")
                fp.flush()
    return results


# ---------------------------------------------------------------------------
# Lightweight run summary. The real calibration analysis lives in analyze.py.
# ---------------------------------------------------------------------------

def write_markdown_summary(results: List[Dict[str, Any]], path: Path) -> None:
    total = len(results)
    succeeded = sum(1 for r in results if r["status"] == "success")
    avg_iters = (
        sum(r["iterations"] or 0 for r in results if r["status"] == "success")
        / succeeded if succeeded else 0
    )
    confirmed = falsified = off_topic = 0
    for r in results:
        rep = r.get("replay_report") or {}
        confirmed += rep.get("confirmed", 0)
        falsified += rep.get("falsified", 0)
        off_topic += rep.get("off_topic", 0)

    lines = [
        "# Benchmark Run Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        f"- Task runs: {total}",
        f"- Succeeded: {succeeded}/{total}",
        f"- Avg iterations on success: {avg_iters:.2f}",
        f"- Replay verdicts: {confirmed} confirmed, {falsified} falsified, "
        f"{off_topic} off-topic",
        "",
        "Run `python -m bench.analyze <results.jsonl>` for the calibration report.",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

async def amain(args: argparse.Namespace) -> int:
    if args.smoke:
        problems = PROBLEMS[:SMOKE_PROBLEMS]
        reps = SMOKE_REPS
    else:
        problems = PROBLEMS[: args.limit] if args.limit else PROBLEMS
        reps = args.reps

    llm = make_llm(args.llm, args.model)
    config = AgentConfig(
        loop=LoopConfig(max_iterations=args.max_iterations),
        workspace_path=Path(args.workspace),
        use_docker=True,
        docker_persistent=True,
        docker_image=args.docker_image,
    )
    agent = SelfImprovingAgent(llm_client=llm, config=config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("bench/results")
    jsonl_path = Path(args.out) if args.out else out_dir / f"bench_{ts}.jsonl"
    md_path = jsonl_path.with_suffix(".md")

    print(f"▶ {len(problems)} problem(s) × {reps} rep(s) "
          f"= {len(problems) * reps} task run(s)")

    results = await run_batch(
        agent, problems, reps=reps, out_path=jsonl_path, verbose=not args.quiet
    )
    write_markdown_summary(results, md_path)

    succeeded = sum(1 for r in results if r["status"] == "success")
    print(f"\n✅ Run complete: {succeeded}/{len(results)} succeeded.")
    print(f"   JSONL: {jsonl_path}")
    print(f"   Summary: {md_path}")
    print(f"   Next: python -m bench.analyze {jsonl_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="bench.runner")
    p.add_argument("--reps", type=int, default=5,
                   help="Repetitions per problem (default 5)")
    p.add_argument("--limit", type=int, default=0,
                   help="Run only the first N problems (0 = all)")
    p.add_argument("--smoke", action="store_true",
                   help=f"Shake-out run: first {SMOKE_PROBLEMS} problems × "
                        f"{SMOKE_REPS} reps")
    p.add_argument("--llm", choices=["ollama", "openai", "anthropic"],
                   default="ollama")
    p.add_argument("--model", default=None,
                   help="Model name override (default: provider's default)")
    p.add_argument("--docker-image", default="crucible-runtime",
                   help="Prebaked runtime image (default: crucible-runtime)")
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--workspace", default="./workspace")
    p.add_argument("--out", default=None,
                   help="Output JSONL path (default: bench/results/bench_<ts>.jsonl)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(parse_args())))
