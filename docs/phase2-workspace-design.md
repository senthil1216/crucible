# Phase 2: Multi-File / Workspace Support – Design

**Status**: Draft  
**Date**: 2026-05-25  
**Related to**: Phase 1 Persistent Docker + Dependency Recovery

---

## 1. Goals

The primary objective of Phase 2 is to evolve Crucible from a **single-file script generator** into an agent that can work with **real project structures** inside its execution environment.

Specific goals:

- Allow the agent to create, modify, and reason about multiple files within a single task.
- Provide a proper workspace inside the persistent Docker container.
- Lay the technical foundation for **Phase 3 (Explicit Tool Use)**.
- Significantly increase the realism and usefulness of tasks the agent can handle (web apps, libraries, CLI tools, etc.).

## 2. Current State & Limitations

Today, the agent primarily operates in a "generate one script and run it" model:

- Code is generated as a single string.
- It is written to `main.py` (or similar) inside the container.
- Execution happens by running that one file.
- There is no concept of a project directory, multiple modules, or file system state across iterations.

This model breaks down quickly for anything beyond trivial scripts.

## 3. Proposed Approach

Introduce a **persistent workspace** inside the container for each task, combined with new filesystem capabilities in the `DockerExecutor`.

### Core Ideas

1. **Workspace per Task**
   - Every task gets its own isolated workspace directory inside the persistent container.
   - Example path: `/workspace/<task_id>/`

2. **Filesystem as First-Class Capability**
   - The agent (via the executor) gains the ability to:
     - List directories
     - Read files
     - Write / overwrite files
     - Create directories
   - These operations happen inside the running persistent container.

3. **Gradual Evolution of Behavior**
   - The agent should be able to move from "generate one file" → "generate a small project" → "iteratively build and modify a project" over time.

## 4. Workspace Model

### Recommended Structure

```
/workspace/
└── <task_id>/
    ├── src/                  # (optional) source code
    ├── tests/                # (optional)
    ├── main.py               # or app.py, etc.
    └── requirements.txt      # if dependencies are managed this way
```

Alternative (simpler starting point):

```
/workspace/<task_id>/
├── main.py
├── utils.py
├── models.py
└── ...
```

**Recommendation**: Start with a flat structure under `/workspace/<task_id>/` for MVP, then allow the agent to create subdirectories as needed.

### Workspace Lifecycle

- Created when a persistent container is started for a task (or lazily on first file write).
- Persists for the lifetime of the task (all iterations).
- Cleaned up when the persistent container is stopped (or optionally persisted for debugging).

## 5. DockerExecutor Interface Changes (Proposed)

We will need to extend `DockerExecutor` with new methods that work against the persistent container:

```python
class DockerExecutor:

    # Existing methods...

    # Phase 2 additions (only meaningful when persistent=True)
    def write_file(self, path: str, content: str) -> bool:
        """Write (or overwrite) a file inside the workspace."""
        ...

    def read_file(self, path: str) -> Optional[str]:
        """Read a file from the workspace."""
        ...

    def list_dir(self, path: str = ".") -> List[str]:
        """List contents of a directory in the workspace."""
        ...

    def create_directory(self, path: str) -> bool:
        """Create a directory (and parents) in the workspace."""
        ...

    def get_workspace_path(self) -> str:
        """Return the root workspace path for the current task (e.g. /workspace/<task_id>)."""
        ...
```

These methods should:
- Only work when a persistent container is active.
- Operate relative to the task’s workspace directory.
- Be safe (e.g., prevent escaping the workspace).

## 6. Impact on Agent Components

| Component       | Changes Needed |
|-----------------|----------------|
| **Planner**     | Should start planning in terms of files and project structure. May need to output file lists + contents. |
| **CodeGenerator** | Needs to support generating multiple files instead of one monolithic script. |
| **ExecutionLoop** | Should be able to execute specific files or modules within the workspace, not just one entry point. |
| **Reflector**   | Should be able to reason about file-level changes and project structure over iterations. |
| **Tester**      | May need to run tests using proper project commands (e.g. `python -m pytest`, `python -m mypackage`). |

## 7. Scope for Phase 2 (MVP)

**In Scope:**
- Basic workspace directory per task
- `write_file`, `read_file`, `list_dir` capabilities in the executor
- Ability for the agent to generate multiple files in one go
- Support for running code from within the workspace (e.g. `python main.py` or `python -m package`)
- Simple directory creation

**Out of Scope (for initial Phase 2):**
- Full tool-calling architecture (this is Phase 3)
- Git integration inside the container
- Complex build systems (Docker-in-Docker, etc.)
- Automatic project scaffolding / cookiecutter-style templates (can come later)
- Persistent workspaces across different agent runs (unless explicitly requested)

## 8. Open Questions

- Should the workspace be mounted from the host (for easier inspection) or stay fully inside the container?
- How should the agent communicate which files it wants to create vs. modify?
- Should there be a concept of "current working project" vs per-task workspaces?
- How do we handle large files or binary assets (if needed later)?

## 9. Suggested Implementation Order

1. Add workspace path management to `DockerExecutor` (when persistent mode is active).
2. Implement `write_file`, `read_file`, and `list_dir` using `exec_run` + `put_archive` / `get_archive`.
3. Update `SelfImprovingAgent` and `ExecutionLoop` to be workspace-aware.
4. Modify `CodeGenerator` (or add a new generator) to support multi-file output.
5. Update the Planner prompt to encourage thinking in terms of files and structure.
6. Add basic tests and a simple multi-file example.

## 10. Success Criteria

Phase 2 is considered successful when:

- The agent can successfully create a small multi-file project (e.g. a FastAPI app with separate `main.py`, `routers/`, and `models.py`).
- It can read and modify existing files across iterations.
- The persistent container + workspace model feels natural and stable.

---

**Status**: Ready for review and discussion.
