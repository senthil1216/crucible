"""
Track C runner — sequentially runs the agent against a coherent batch of
related tasks against local Ollama, logging per-task metrics and memory
retrievals to a JSONL file.

The point is to validate that the Phase A/B/C/D memory infrastructure
(Patterns, Learnings, env context) actually helps later tasks in a batch.
A single run produces:

  - bench/results/track_c_<timestamp>.jsonl   per-task structured log
  - bench/results/track_c_<timestamp>.md      1-page human summary

Usage:

    python -m bench.track_c_runner [--limit N] [--llm ollama] [--model qwen2.5-coder:7b]

Pre-reqs:
  - Ollama running locally with the chosen model pulled
  - Docker available (the runner forces --docker-persistent so env capture
    fires; without it, Phase D bonuses never kick in)

Status: scaffold. The score-breakdown logging is a TODO — it needs a hook
into Planner / LongTermMemory.find_similar_solutions to capture the
component scores (semantic / project_type / deps / installed packages).
For now the runner logs the *post-retrieval* artifacts (which patterns
and learnings came back) but not the *why*.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local imports — keep at module level so import errors surface immediately
# rather than mid-run.
from agent.core import SelfImprovingAgent
from agent.llm_clients import OllamaClient, OpenAIClient, AnthropicClient
from agent.models import AgentConfig, LoopConfig, IterationState, Status

from bench.track_c_tasks import TASKS, TaskSpec


# ---------------------------------------------------------------------------
# LLM factory (kept minimal — Track C is Ollama-first per NEXT_STEPS).
# ---------------------------------------------------------------------------

def make_llm(provider: str, model: Optional[str]):
    if provider == "ollama":
        return OllamaClient(
            model=model or os.getenv("AGENT_OLLAMA_MODEL", "qwen2.5-coder:7b")
        )
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini")
    if provider == "anthropic":
        return AnthropicClient(model=model or "claude-3-haiku-20240307")
    raise ValueError(f"Unknown LLM provider for Track C: {provider}")


# ---------------------------------------------------------------------------
# Per-task execution.
# ---------------------------------------------------------------------------

async def run_one(
    agent: SelfImprovingAgent,
    task: TaskSpec,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single task. Returns a structured result dict.

    The result captures: iteration count, status, elapsed time, retrieved
    patterns/learnings (id, similarity, source), and any exception text.
    """
    # Snapshot retrieval *before* solve(). The agent will run its own
    # retrieval internally during planning, but solve() doesn't expose the
    # result. Re-querying here is cheap (the embeddings cache hits the
    # same in-memory cache) and gives us a structured record.
    similar_solutions = await agent.long_term_memory.find_similar_solutions(
        task.goal, k=5
    )
    relevant_learnings = await agent.long_term_memory.find_relevant_learnings(
        task.goal, k=5
    )

    if verbose:
        print(f"\n{'#' * 60}")
        print(f"# Task {task.id} ({task.expected_difficulty})")
        print(f"# Patterns retrieved: {len(similar_solutions)}, "
              f"Learnings retrieved: {len(relevant_learnings)}")
        print(f"{'#' * 60}")

    start = time.perf_counter()
    exc_text: Optional[str] = None
    state: Optional[IterationState] = None
    try:
        state = await agent.solve(goal=task.goal, task_id=task.id)
    except Exception as e:
        import traceback
        exc_text = traceback.format_exc()
    elapsed = time.perf_counter() - start

    return {
        "task_id": task.id,
        "goal": task.goal,
        "difficulty": task.expected_difficulty,
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
    }


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------

def write_markdown_summary(results: List[Dict[str, Any]], path: Path) -> None:
    """Write a 1-page human summary suitable for docs/."""
    total = len(results)
    succeeded = sum(1 for r in results if r["status"] == "success")
    avg_iters = (
        sum(r["iterations"] or 0 for r in results if r["status"] == "success")
        / succeeded if succeeded else 0
    )
    total_time = sum(r["elapsed_seconds"] for r in results)

    lines = [
        "# Track C Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        f"- Tasks: {total}",
        f"- Succeeded: {succeeded}/{total}",
        f"- Avg iterations on success: {avg_iters:.2f}",
        f"- Total wall time: {total_time:.1f}s",
        "",
        "## Per-task",
        "",
        "| Task | Difficulty | Status | Iters | Patterns | Learnings | Elapsed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['task_id']} | {r['difficulty']} | {r['status']} | "
            f"{r['iterations'] or '-'} | {len(r['retrieved_patterns'])} | "
            f"{len(r['retrieved_learnings'])} | {r['elapsed_seconds']}s |"
        )

    lines.extend([
        "",
        "## Qualitative notes",
        "",
        "TODO: fill in by hand after a run —",
        "- Did retrieved learnings actually inform later tasks (e.g. did task 06 "
        "  benefit from a learning extracted on task 03)?",
        "- Any infrastructure bugs surfaced (fold into Track A follow-up)?",
        "- Which tasks failed and why? (LLM capability, agent infra, prompt)",
    ])

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

async def amain(args: argparse.Namespace) -> int:
    tasks = TASKS[: args.limit] if args.limit else TASKS

    llm = make_llm(args.llm, args.model)
    config = AgentConfig(
        loop=LoopConfig(max_iterations=args.max_iterations),
        workspace_path=Path(args.workspace),
        use_docker=True,
        docker_persistent=True,
    )
    agent = SelfImprovingAgent(llm_client=llm, config=config)

    results: List[Dict[str, Any]] = []

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("bench/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"track_c_{ts}.jsonl"
    md_path = out_dir / f"track_c_{ts}.md"

    with open(jsonl_path, "w") as fp:
        for task in tasks:
            result = await run_one(agent, task, verbose=not args.quiet)
            results.append(result)
            fp.write(json.dumps(result) + "\n")
            fp.flush()

    write_markdown_summary(results, md_path)

    print(f"\n✅ Track C run complete.")
    print(f"   JSONL: {jsonl_path}")
    print(f"   Report: {md_path}")
    succeeded = sum(1 for r in results if r["status"] == "success")
    print(f"   {succeeded}/{len(results)} succeeded.")
    return 0 if succeeded == len(results) else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="bench.track_c_runner")
    p.add_argument("--limit", type=int, default=0,
                   help="Run only the first N tasks (0 = all)")
    p.add_argument("--llm", choices=["ollama", "openai", "anthropic"],
                   default="ollama")
    p.add_argument("--model", default=None,
                   help="Model name override (default: provider's default)")
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--workspace", default="./workspace")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(parse_args())))
