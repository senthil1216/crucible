# Phase 1: Robust Automatic Dependency Recovery – Design

**Status**: Draft  
**Owner**: —  
**Date**: 2026-05-25

## 1. Goals

The primary goal of this phase is to make the persistent Docker execution flow **solid and trustworthy**, with a strong emphasis on automatic recovery from missing dependencies.

Specific objectives:
- Significantly improve the reliability of automatic package installation when the agent encounters `ModuleNotFoundError` / `ImportError`.
- Make the self-improving behavior (detect → install → retry) observable and robust.
- Provide a clean, extensible foundation that supports future improvements (multi-language, smarter resolution, user confirmation, etc.).
- Reduce the number of times the circuit breaker triggers due to repeated import errors.

## 2. Current State & Problems

### Current Implementation
- Simple regex in `ExecutionLoop._extract_package_name()` to pull package names from stderr.
- Single automatic retry after calling `executor.install_packages()`.
- Very limited error classification from pip output.
- No support for `requirements.txt`.
- No distinction between different failure modes (package not found vs build failure vs network issue).

### Key Weaknesses
- Fragile package name extraction.
- Poor handling of real-world pip errors.
- No graceful degradation when installation fails.
- Limited visibility into what the system is doing during recovery.
- Hard to extend with smarter behavior.

## 3. Proposed Solution

Introduce a new component: **`DependencyManager`**

This class will own all logic related to detecting, planning, and executing dependency installations within the context of a persistent container.

### High-Level Responsibilities
- Extract missing packages from execution errors.
- Classify installation failure reasons.
- Decide what to install (with fallbacks and heuristics).
- Coordinate installation via the executor.
- Manage retry policy after installation attempts.
- Support installation from `requirements.txt`.
- Provide rich observability (what was attempted, what succeeded/failed).

### Integration Points
- `ExecutionLoop` will call into `DependencyManager` when it detects import-related failures.
- `DockerExecutor` (in persistent mode) will expose `install_packages()` (already partially implemented).
- `SelfImprovingAgent` can expose `ensure_packages()` as a higher-level API.

## 4. DependencyManager Interface (Proposed)

```python
class DependencyManager:
    def __init__(self, executor: DockerExecutor, config: AgentConfig):
        ...

    def handle_import_error(
        self, 
        error_message: str, 
        current_code: str
    ) -> RecoveryResult:
        """
        Main entry point called by the loop when an import error occurs.
        Returns whether recovery was attempted and whether it succeeded.
        """
        ...

    def install_packages(
        self, 
        packages: list[str]
    ) -> InstallResult:
        """Install a list of packages (delegates to executor)."""
        ...

    def install_from_requirements(
        self, 
        requirements_path: str
    ) -> InstallResult:
        """Install packages listed in a requirements.txt file."""
        ...

    def get_installed_packages(self) -> list[str]:
        """Return list of currently installed packages in the container."""
        ...
```

### Supporting Types

```python
@dataclass
class RecoveryResult:
    attempted: bool
    packages_installed: list[str]
    success: bool
    error: Optional[str] = None

@dataclass
class InstallResult:
    success: bool
    packages: list[str]
    stdout: str
    stderr: str
    failure_reason: Optional[str] = None   # e.g. "not_found", "build_error", "network"
```

## 5. Task Breakdown & Parallelization

All tasks below are scoped according to the Phase 1 decisions (Python packages only, resolve with `*`, max 4 attempts, persist successful history to long-term memory, and "simple & fast" implementation).

We will work on the following six tasks in parallel where possible. Clear interfaces will allow independent progress.

### Task 1: Improved Package Name Extraction
- Improve package name extraction logic (will live in `DependencyManager`).
- Reliably handle multiple packages in one error.
- Better handling of submodule imports (`foo.bar` → install `foo`).
- Support more error message formats (including from `exec_run` output).

**Parallelization**: Can start immediately. This is foundational for Tasks 2, 4, and 5. Good test coverage is critical.

### Task 2: Smarter Installation Logic
- Build a mapping of common import-name → PyPI-name mismatches.
- Implement fallback installation attempts.
- Consider using `pip install` with `--upgrade` or other flags intelligently.

**Parallelization**: Can start after basic `install_packages` interface is stable.

### Task 3: Support for `requirements.txt`
- Detect when generated code includes or references a `requirements.txt`.
- Implement `install_from_requirements()`.
- Decide policy: always install from requirements first? Only on demand?

**Parallelization**: Medium dependency on Task 2.

### Task 4: Better Post-Install Retry & Decision Logic
- Implement a retry policy with a hard maximum of **4 automatic install attempts** per task.
- Track installation attempts and outcomes within a task.
- Decide when to stop automatic recovery and hand off to the Reflector (after 4 attempts or repeated failures).
- Begin laying groundwork for persisting successful installations to long-term memory (cross-task learning).

**Parallelization**: Can be designed in parallel with Task 1 & 2, but has dependency on Task 1 for error detection.

### Task 5: Distinguish Installation Failure Types
- Parse pip output to categorize failures:
  - Package not found on PyPI
  - Build/compilation failure
  - Network / timeout
  - Permission / environment issues
- Surface categorized errors to the agent/reflector.

**Parallelization**: Good candidate for parallel work. Needs good test cases with real pip output.

### Task 6: Optional User Confirmation for Installs
- Add config: `docker_ask_before_install: bool`
- When enabled, pause and ask user for confirmation before running `pip install`.
- Support both CLI and programmatic usage.

**Parallelization**: Lowest dependency. Mostly UX + config work.

## 6. Phase 1 Scope Decisions

The following decisions have been made to keep Phase 1 focused, simple, and fast:

- **System packages**: Only Python packages (`pip install`). No `apt-get` or system-level packages in Phase 1.
- **Version resolution**: Resolve dependencies using `*` (e.g. `fastapi*`). Let pip choose the latest compatible version. No custom pinning or conflict resolution logic in Phase 1.
- **Installation history**: Persist successful installation history across tasks via long-term memory. This supports long-term learning.
- **Max automatic install attempts**: Maximum of **4** automatic install attempts per task.

## 8. Prioritized Task Breakdown (Recommended Order)

For implementation, the following order is recommended to maximize learning and reduce risk:

| Priority | Task | Focus Area | Dependencies | Notes |
|----------|------|------------|--------------|-------|
| 1 | **Task 1: Improved Package Name Extraction** | Error parsing & package detection | None | Highest leverage. Everything else depends on this. |
| 2 | **Task 5: Distinguish Installation Failure Types** | Error classification from pip output | Task 1 | Helps decide when *not* to retry installs. |
| 3 | **Task 2: Smarter Installation Logic** | Name mappings + fallback strategies | Task 1 | Makes recovery more effective in practice. |
| 4 | **Task 4: Better Post-Install Retry & Decision Logic** | Retry policy (max 4 attempts) + memory integration | Task 1, Task 5 | Includes groundwork for persisting install history. |
| 5 | **Task 3: Support for `requirements.txt`** | Installing from generated requirements files | Task 2, Task 4 | Nice-to-have for more mature code generation. |
| 6 | **Task 6: Optional User Confirmation for Installs** | `docker_ask_before_install` config | Low | Mostly independent UX work. Can be done anytime. |

**Guiding Principle**: Keep every task as simple and fast to implement as possible while still delivering clear value toward robust automatic recovery.

## 9. Next Steps After Design

1. Final review and approval of this document.
2. Create `agent/dependency_manager.py` with the core interface and skeleton.
3. Begin implementation starting with **Task 1**.
4. Add unit tests for package extraction and error classification early.
- **Overall philosophy**: Keep the `DependencyManager` implementation simple and fast. Prefer pragmatic solutions over sophisticated dependency resolution in this phase.

## 7. Open Questions (Deferred)

The following questions are deferred beyond Phase 1:

- How sophisticated should cross-task memory for installations become?
- Should we eventually support system packages or non-Python languages?
- What is the long-term strategy for dependency conflict resolution and versioning?

## 7. Next Steps

1. Review and refine this design document.
2. Define the final interface for `DependencyManager`.
3. Create the initial skeleton in `agent/dependency_manager.py`.
4. Split the six tasks across workstreams and begin implementation.
5. Add comprehensive tests (especially for error parsing and failure classification).

---

**Status**: Ready for review and refinement.
