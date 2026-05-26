# Long-term Memory Improvements – Design

**Status**: Draft  
**Date**: 2026-05-25  
**Focus**: Strengthening the self-improving capabilities of the agent

---

## 1. Goals

The primary goal is to make Long-term Memory a **first-class, actively used component** in the self-improving loop, rather than a passive store of past solutions.

Specific objectives:

- Move from shallow similarity search to **multi-signal retrieval** of relevant past experience.
- Store **richer, structured knowledge** (not just raw code + plans).
- Enable the **Reflector** to contribute learnings back into memory after successful tasks.
- Begin leveraging **persistent workspace state** (installed packages, project structure, etc.) as memory signals.
- Make the agent demonstrably better at similar tasks over time.

## 2. Current State & Problems

### Current Long-term Memory Behavior
- Stores successful solutions as `(goal, plan, code)` tuples.
- Retrieval is primarily based on embedding similarity of the goal text.
- The Planner receives the top-k similar solutions but usage is relatively passive.
- The Reflector does not systematically write structured insights back to memory.
- Workspace/environment state (what packages were installed, what files were created) is not captured.

### Key Weaknesses
- **Surface-level similarity**: Two tasks can have similar goals but very different solutions (e.g., "build a web API" in FastAPI vs Flask vs raw HTTP).
- **Lossy storage**: Only raw artifacts are stored; the *reasoning* and *patterns* that made something successful are lost.
- **One-way flow**: Memory is mostly read-only after a task succeeds. The Reflector’s insights are under-utilized.
- **Environment blindness**: The agent has no memory of the execution environment state across tasks.

## 3. Proposed Approach

We will evolve Long-term Memory from a **solution archive** into a **structured knowledge base** of patterns, decisions, and environmental context.

### Core Principles for Phase 1 of Memory Improvements
- Keep it **pragmatic and incremental** — avoid over-engineering a full knowledge graph.
- Focus on **high-signal, low-complexity** structures first.
- Ensure changes provide clear value to the Planner and Reflector.

## 4. Key Changes

### 4.1 Richer Memory Entry Model

Instead of storing raw `(goal, plan, code)`, we will store **Pattern** entries that capture:

- Goal summary + embedding
- Project type (from Plan)
- Key decisions made (e.g., framework chosen, architecture pattern)
- Dependencies that were required
- Environment context at success time (packages installed, workspace structure)
- Structured learnings / rationale from the Reflector
- Success metrics (iterations taken, whether auto-recovery was needed)

### 4.2 Improved Retrieval (Planner)

The Planner should be able to query memory using multiple signals, not just goal similarity:

- Project type match
- Dependency overlap
- Similar architectural patterns
- Goal embedding similarity (as one signal among several)

Retrieval should return not just past solutions, but **applicable patterns and lessons**.

### 4.3 Structured Writing from the Reflector

After a task succeeds, the Reflector should produce a structured `Learning` object that gets written to long-term memory. Examples:

- "For FastAPI projects, always include a `/health` endpoint"
- "When using SQLAlchemy with FastAPI, the recommended pattern is..."
- "This task required `pydantic` + `alembic` in addition to `fastapi`"

### 4.4 Environment State as Memory Signal

The persistent workspace now gives us a rich source of context. We should begin capturing:

- Packages that were installed during the task (especially via automatic recovery)
- High-level structure of the final workspace (key directories/files created)
- Any environment-specific decisions made

This information becomes valuable context for future similar tasks.

## 5. Phased Implementation Plan (Small Increments)

### Phase A – Foundation (Short term)
- Define a new `MemoryEntry` / `Pattern` data model
- Update storage in `LongTermMemory` to support richer fields
- Update `find_similar_solutions` to accept additional filters (project_type, dependencies)
- Modify the Planner to pass `project_type` when querying memory

### Phase B – Structured Reflection Writing
- Extend the Reflector to produce structured learnings on success
- Add method to write learnings into long-term memory
- Begin capturing installed packages from the persistent container at task completion

### Phase C – Better Retrieval & Usage
- Improve similarity search (multi-signal scoring)
- Update Planner prompt and logic to make better use of retrieved patterns
- Surface relevant past learnings more explicitly during planning

### Phase D – Environment Context Integration
- Capture workspace state at success
- Include environment context in memory entries
- Use installed packages / workspace patterns as retrieval signals

## 6. Open Questions

- What is the right granularity for stored patterns? (per-task vs abstracted patterns)
- How do we handle outdated or low-quality learnings over time?
- Should we also store *negative* learnings (what didn’t work)?
- How much structure vs free text should the Reflector produce when writing learnings?
- Do we need embeddings on multiple fields (goal + learnings + project_type), or is one sufficient?

## 7. Success Criteria

We will consider this work successful when:

- The Planner regularly surfaces relevant past patterns that influence its output.
- The agent requires meaningfully fewer iterations on repeated or similar project types over time.
- The Reflector produces usable, structured learnings that are stored and retrievable.
- We can demonstrate (even qualitatively) that the agent performs better on a second similar task than the first.

---

**Next Step Recommendation**

After this document is reviewed, the logical first implementation step is **Phase A – Foundation** (richer data model + improved retrieval interface in LongTermMemory + Planner updates).

Would you like to:
- Refine this document first?
- Move straight into defining the new memory data model?
- Or start with a different piece?

Let me know how you’d like to proceed.
