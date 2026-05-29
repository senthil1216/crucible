"""
Execution Loop: Plan → Execute → Test → Reflect cycle.
Includes circuit breaker pattern and state persistence.
"""

import asyncio
import time
from typing import List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from agent.models import (
    IterationState, Plan, CodeArtifact, TestResults, 
    Reflection, Status, LoopConfig, AgentConfig
)
from agent.memory.short_term import ShortTermMemory
from agent.planner import Planner
from agent.code_generator import CodeGenerator
from agent.tester import Tester
from agent.reflector import Reflector
from agent.dependency_manager import DependencyManager  # Phase 2
from agent.test_generator import (
    TestGenerator, static_check_test_code, build_stub_files, MODULE_NAME,
)
from agent.profiling import StepProfiler
from agent.replay import ReplayEngine


def _make_fix_diff(before: Optional[str], after: Optional[str]) -> Optional[str]:
    """Build a small unified diff between the broken and fixed code blobs.

    Truncates aggressively — the diff is stored on a failure-memory entry
    purely for human inspection. None inputs return None.
    """
    if before is None or after is None:
        return None
    import difflib
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile="broken",
        tofile="fixed",
        n=2,
        lineterm="",
    )
    text = "\n".join(diff)
    return text[:2000] if text else None


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern to prevent infinite loops.
    Opens after threshold failures in a window.
    """
    failure_threshold: int = 3
    failure_window: int = 5  # Check last N iterations
    cooldown_period: int = 60  # Seconds
    
    def __post_init__(self):
        self.failure_history: List[bool] = []
        self.is_open = False
        self.opened_at: Optional[datetime] = None
    
    def record_result(self, success: bool) -> None:
        """Record the result of an iteration."""
        self.failure_history.append(not success)
        
        # Keep only recent history
        if len(self.failure_history) > self.failure_window:
            self.failure_history.pop(0)
        
        # Check if we should open the circuit
        if not success and not self.is_open:
            recent_failures = sum(self.failure_history[-self.failure_window:])
            if recent_failures >= self.failure_threshold:
                self.open()
    
    def open(self) -> None:
        """Open the circuit - stop execution."""
        self.is_open = True
        self.opened_at = datetime.now()
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if not self.is_open:
            return True
        
        # Check if cooldown has elapsed
        if self.opened_at:
            elapsed = (datetime.now() - self.opened_at).total_seconds()
            if elapsed > self.cooldown_period:
                self.close()
                return True
        
        return False
    
    def close(self) -> None:
        """Close the circuit - allow execution."""
        self.is_open = False
        self.opened_at = None
        self.failure_history.clear()
    
    def get_state(self) -> str:
        """Get circuit state as string."""
        return "OPEN" if self.is_open else "CLOSED"


class ExecutionLoop:
    """
    The core execution loop: Plan → Execute → Test → Reflect
    
    Features:
    - Configurable max iterations
    - Circuit breaker for infinite loops
    - State persistence for resumability
    - Progress callbacks
    """
    
    def __init__(
        self,
        planner: Planner,
        code_generator: CodeGenerator,
        tester: Tester,
        reflector: Reflector,
        short_term_memory: ShortTermMemory,
        config: LoopConfig = None,
        on_iteration: Callable[[IterationState], None] = None,
        on_plan: Callable[[Plan], None] = None,
        on_code: Callable[[CodeArtifact], None] = None,
        on_test: Callable[[TestResults], None] = None,
        on_reflect: Callable[[Reflection], None] = None,
        install_packages: Optional[Callable[[list[str]], bool]] = None,  # Phase 2 (legacy)
        dependency_manager: Optional[DependencyManager] = None,          # Phase 2
        test_generator: Optional[TestGenerator] = None,                  # real-pytest gate
        profiler: Optional[StepProfiler] = None,                         # per-step timing
        replay_engine: Optional["ReplayEngine"] = None,                  # Track D phase 2
    ):
        self.planner = planner
        self.code_generator = code_generator
        self.tester = tester
        self.reflector = reflector
        self.memory = short_term_memory
        self.config = config or LoopConfig()
        self.install_packages = install_packages
        self.dependency_manager = dependency_manager  # Preferred Phase 2 path
        # Track D phase 2: replays this task's failure predictions against the
        # passing code on success (record-only). None disables replay.
        self.replay_engine = replay_engine
        # When set, single-file Python tasks are gated on a real, frozen pytest
        # suite instead of "the script exited 0". When None, the legacy
        # exit-code behavior is preserved (used by the loop's own unit tests).
        self.test_generator = test_generator
        self.profiler = profiler or StepProfiler()
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.failure_threshold,
            failure_window=self.config.failure_window,
            cooldown_period=self.config.cooldown_period
        )
        
        # Callbacks for progress tracking
        self.on_iteration = on_iteration
        self.on_plan = on_plan
        self.on_code = on_code
        self.on_test = on_test
        self.on_reflect = on_reflect
    
    async def run(
        self,
        goal: str,
        task_id: Optional[str] = None,
        resume_from: Optional[IterationState] = None,
        plan: Optional[Plan] = None,
    ) -> IterationState:
        """
        Run the execution loop until success, failure, or max iterations.
        
        Args:
            goal: The coding task to accomplish
            task_id: Optional identifier for this task
            resume_from: Optional state to resume from
            plan: Optional pre-computed plan. When the caller already planned
                (e.g. core, using long-term memory), pass it here so the loop
                reuses it instead of making a second, memory-blind plan call.

        Returns:
            Final iteration state
        """
        # Initialize or restore state
        if resume_from:
            current_plan = resume_from.plan
            frozen_tests = resume_from.test_code
            iteration = resume_from.iteration + 1
            self.memory.add(resume_from)
        else:
            # Plan up front so the test-first suite can be generated before any
            # implementation exists. Reuse a caller-supplied plan when given
            # (avoids a duplicate, memory-blind planning call).
            if plan is not None:
                print("\n[PLAN] Using pre-computed plan")
                current_plan = plan
            else:
                print("\n[PLAN] Creating plan...")
                with self.profiler.track("planning"):
                    current_plan = await self.planner.create_plan(goal)
            print(f"Steps: {len(current_plan.steps)}")
            print(f"Tests: {len(current_plan.test_cases)}")
            if self.on_plan:
                self.on_plan(current_plan)
            frozen_tests = None
            iteration = 1

        # Eager dependency install (Docker persistent path only). The plan already
        # declares its pip packages, so install them once up front instead of
        # discovering them reactively via ImportErrors. This also stops
        # dependency-recovery iterations from eating the circuit-breaker budget.
        # Kicked off as a task so the (IO-bound) pip install overlaps the first
        # slow LLM call — genuine concurrency even on a single-GPU backend.
        install_task = None
        if self.dependency_manager and not resume_from:
            install_task = asyncio.create_task(self._eager_install(current_plan))

        # Test-first: generate and validate the frozen pytest suite once. The
        # suite is reused unchanged across fix iterations.
        if frozen_tests is None and self._pytest_gate_enabled(current_plan):
            frozen_tests, gate_failure = await self._prepare_frozen_tests(
                current_plan, goal, task_id, install_task=install_task
            )
            install_task = None  # awaited inside _prepare_frozen_tests
            if gate_failure is not None:
                return gate_failure

        # Legacy path had no test-generation call to overlap with — make sure the
        # eager install finished before we start iterating.
        if install_task is not None:
            await install_task

        final_state = None
        # Track the most recently stored failure so that when the next
        # iteration succeeds we can mark that failure was_fixed=True. The
        # diff field stores the broken→fixed transition for inspection.
        last_failure_id: Optional[str] = None
        last_failure_code: Optional[str] = None
        # Track D phase 2: every failure-memory id emitted during this task, so
        # that on success we can replay all of their predictions against the
        # final code. Reset per run() — this loop instance is reused across tasks.
        emitted_failure_ids: List[str] = []

        while iteration <= self.config.max_iterations:
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                print(f"⚠️ Circuit breaker is {self.circuit_breaker.get_state()}")

                # Try to surface the last real code attempt instead of an empty artifact.
                # This makes the final summary and persisted file much more useful.
                last_states = self.memory.get_recent(1)
                if last_states:
                    last = last_states[0]
                    code_to_use = last.code
                    plan_to_use = last.plan
                else:
                    code_to_use = CodeArtifact(source="", file_path="", language="python")
                    plan_to_use = current_plan or Plan(goal=goal, steps=[], test_cases=[])

                return self._create_final_state(
                    iteration=iteration - 1,
                    plan=plan_to_use,
                    code=code_to_use,
                    test_results=TestResults(passed=False, stderr="Circuit breaker opened"),
                    reflection=Reflection(
                        success=False,
                        analysis="Too many consecutive failures - circuit breaker opened",
                        should_continue=False
                    ),
                    status=Status.CIRCUIT_BREAKER,
                    task_id=task_id,
                    test_code=frozen_tests,
                )
            
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{self.config.max_iterations}")
            print(f"Goal: {goal}")
            print(f"{'='*60}")
            
            try:
                state = await self._run_iteration(
                    goal=goal,
                    iteration=iteration,
                    current_plan=current_plan,
                    frozen_tests=frozen_tests,
                    task_id=task_id
                )
                
                final_state = state
                
                # Callbacks
                if self.on_iteration:
                    self.on_iteration(state)
                
                # Store in memory
                self.memory.add(state)
                
                # Record result for circuit breaker
                self.circuit_breaker.record_result(state.status == Status.SUCCESS)
                
                # Check termination conditions
                if state.status == Status.SUCCESS:
                    print("\n✅ SUCCESS! Tests passed.")
                    # Mark the prior iteration's failure as fixed so future
                    # retrievals get a small boost — the failure→fix transition
                    # is the signal we want the agent to learn from.
                    failure_memory = getattr(self.reflector, "failure_memory", None)
                    if last_failure_id and failure_memory:
                        fix_diff = _make_fix_diff(last_failure_code, state.code.source)
                        try:
                            failure_memory.mark_fixed(last_failure_id, fix_diff)
                        except Exception as e:
                            print(f"⚠️ mark_fixed failed: {e}")
                    # Track D phase 2: replay this task's failure predictions
                    # against the passing code (record-only — never changes the
                    # SUCCESS result). Scoped to the single-file Python pytest
                    # path: the replay driver assumes the `solution` module
                    # contract, like _pytest_gate_enabled.
                    await self._maybe_replay_predictions(
                        state, current_plan, emitted_failure_ids, goal
                    )
                    return state

                if not state.reflection.should_continue:
                    print("\n❌ Stopping: Reflector indicates task may be hopeless")
                    return state

                if iteration == self.config.max_iterations:
                    print("\n⚠️ MAX ITERATIONS REACHED")
                    return state

                # Remember this iteration's failure so the next iteration can
                # mark it fixed if it succeeds.
                if state.reflection.failure_id:
                    last_failure_id = state.reflection.failure_id
                    last_failure_code = state.code.source
                    emitted_failure_ids.append(state.reflection.failure_id)

                # Prepare for next iteration
                current_plan = self._prepare_next_plan(state, goal)
                iteration += 1
                
            except Exception as e:
                print(f"\n💥 Exception in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                
                error_state = self._create_final_state(
                    iteration=iteration,
                    plan=current_plan or Plan(goal=goal, steps=[], test_cases=[]),
                    code=CodeArtifact(source="", file_path="", language="python"),
                    test_results=TestResults(
                        passed=False,
                        stderr=f"Exception: {str(e)}"
                    ),
                    reflection=Reflection(
                        success=False,
                        analysis=f"Exception during execution: {str(e)}",
                        should_continue=False
                    ),
                    status=Status.FAILED,
                    task_id=task_id,
                    test_code=frozen_tests,
                )
                return error_state
        
        return final_state
    
    @staticmethod
    def _error_blob(test_results: TestResults) -> str:
        """Combine all failure text so import errors are findable regardless of
        whether pytest wrote them to stdout, stderr, or a collection longrepr."""
        parts = [
            f.get("message", "") for f in (test_results.test_failures or [])
        ]
        parts.append(test_results.stderr or "")
        parts.append(test_results.stdout or "")
        return "\n".join(p for p in parts if p)

    async def _maybe_replay_predictions(
        self,
        state: IterationState,
        plan: Plan,
        failure_ids: List[str],
        goal: str,
    ) -> None:
        """Replay this task's failure predictions against the passing code and
        attach the report to `state.replay_report`. Best-effort and record-only:
        any error here is logged and swallowed — it must never turn a SUCCESS
        into a failure (that's the inert `predictions_gate_enabled` hook's job in
        a future phase, not phase 2)."""
        if not self.replay_engine or not failure_ids:
            return
        # Same `solution`-module contract the replay driver depends on.
        if not self._pytest_gate_enabled(plan):
            return
        try:
            with self.profiler.track("replay"):
                report = await self.replay_engine.replay_for_failures(
                    failure_ids, state.code.source, goal=goal
                )
            state.replay_report = report.to_dict()
            print(report.summary_line())
        except Exception as e:
            print(f"⚠️ Prediction replay failed (non-fatal): {e}")

    async def _eager_install(self, plan: Plan) -> None:
        """Install the plan's declared pip dependencies once, up front.

        Best-effort and Docker-persistent only (needs a DependencyManager).
        Failures are non-fatal — reactive ImportError recovery stays as the
        safety net for anything the plan under-declared (e.g. fastapi naming
        httpx implicitly). Runs the blocking install in a thread so it truly
        overlaps the concurrent test-generation LLM call.
        """
        if not self.dependency_manager:
            return
        deps = list(getattr(plan, "dependencies", None) or [])
        if not deps:
            return
        print(f"[DEPS] Pre-installing declared dependencies: {deps}")
        try:
            with self.profiler.track("dependency_install"):
                result = await asyncio.to_thread(
                    self.dependency_manager.install_packages, deps
                )
        except Exception as e:
            print(f"[DEPS] Pre-install error (non-fatal): {e}")
            return
        if getattr(result, "success", False):
            print(f"[DEPS] Installed {getattr(result, 'packages', deps)}")
        else:
            print(
                f"[DEPS] Pre-install incomplete: {getattr(result, 'stderr', '')} "
                "— relying on reactive recovery"
            )

    def _pytest_gate_enabled(self, plan: Plan) -> bool:
        """
        True when this task should be gated on a real, frozen pytest suite.

        Scoped to single-file Python tasks — the `solution` module contract and
        the stub-based vacuity check only make sense there. Multi-file / non-
        Python tasks fall back to the legacy exit-code behavior.
        """
        return (
            self.test_generator is not None
            and getattr(plan, "language", "python") == "python"
            and not getattr(plan, "use_multi_file", False)
        )

    async def _generate_tests(self, plan: Plan, **kwargs) -> CodeArtifact:
        """Profiled wrapper around the test generator."""
        with self.profiler.track("test_generation"):
            return await self.test_generator.generate_tests(plan, **kwargs)

    async def _prepare_frozen_tests(
        self,
        plan: Plan,
        goal: str,
        task_id: Optional[str],
        install_task: Optional["asyncio.Task"] = None,
    ) -> tuple[Optional[CodeArtifact], Optional[IterationState]]:
        """
        Generate, validate, and freeze the pytest suite (test-first).

        Returns (tests, None) on success, or (None, failure_state) when no
        structurally valid suite could be produced. A structurally valid but
        *vacuous* suite (passes against an empty stub) is non-fatal: we warn and
        proceed, because the suite is still real — we just couldn't prove it has
        teeth (this is also the common case under a weak/mock LLM).

        If `install_task` is given (eager dependency install kicked off in
        parallel), it is awaited before any stub pytest run so the vacuity check
        and every iteration see the declared packages.
        """
        print("\n[TESTS] Generating frozen pytest suite (test-first)...")
        attempts = max(0, self.config.max_test_regenerations)
        tests = await self._generate_tests(plan)

        # The eager install ran concurrently with the LLM call above; ensure it
        # has finished before the first stub pytest run below.
        if install_task is not None:
            await install_task

        for i in range(attempts + 1):
            ok, reasons = static_check_test_code(tests.source, MODULE_NAME)
            if not ok:
                print(f"  Test suite rejected (structure): {'; '.join(reasons)}")
                if i < attempts:
                    tests = await self._generate_tests(
                        plan, previous_tests=tests.source, feedback="; ".join(reasons)
                    )
                    continue
                # Exhausted retries with an unusable suite — fatal.
                return None, self._gate_failure_state(
                    plan, goal, task_id,
                    "Could not generate a structurally valid pytest suite: "
                    + "; ".join(reasons),
                )

            # Structurally valid — now check the tests actually have teeth.
            vacuous = await self._tests_pass_against_stub(tests, plan)
            if vacuous:
                print("  Test suite is vacuous (passes against an empty stub).")
                if i < attempts:
                    tests = await self._generate_tests(
                        plan,
                        previous_tests=tests.source,
                        feedback="the tests pass against an empty implementation; "
                                 "assert concrete behaviour so a stub fails them",
                    )
                    continue
                # Best-effort: proceed but flag it.
                print("  ⚠️  Proceeding with a vacuity warning.")
                tests.metadata["vacuity_warning"] = True
                return tests, None

            print(f"  Frozen suite accepted: {tests.file_path}")
            return tests, None

        return tests, None

    async def _tests_pass_against_stub(self, tests: CodeArtifact, plan: Plan) -> bool:
        """Run the frozen suite against an empty stub; True means the tests are vacuous."""
        try:
            stub_files = build_stub_files(tests.source, MODULE_NAME)
            with self.profiler.track("vacuity_check"):
                result = await self.tester.run_pytest(
                    stub_files, {tests.file_path: tests.source}
                )
            return result.passed
        except Exception as e:
            # If we can't run the stub check, don't block — we just can't prove
            # vacuity either way.
            print(f"  (stub vacuity check skipped: {e})")
            return False

    def _gate_failure_state(
        self, plan: Plan, goal: str, task_id: Optional[str], reason: str
    ) -> IterationState:
        return self._create_final_state(
            iteration=0,
            plan=plan,
            code=CodeArtifact(source="", file_path=f"{MODULE_NAME}.py", language="python"),
            test_results=TestResults(passed=False, stderr=reason, error_type="TestGenerationError"),
            reflection=Reflection(success=False, analysis=reason, should_continue=False),
            status=Status.FAILED,
            task_id=task_id,
        )

    async def _run_iteration(
        self,
        goal: str,
        iteration: int,
        current_plan: Plan,
        frozen_tests: Optional[CodeArtifact],
        task_id: Optional[str],
    ) -> IterationState:
        """Dispatch to the real-pytest path or the legacy exit-code path."""
        if frozen_tests is not None and self._pytest_gate_enabled(current_plan):
            return await self._run_iteration_pytest(
                goal, iteration, current_plan, frozen_tests, task_id
            )
        return await self._run_iteration_legacy(goal, iteration, current_plan, task_id)

    async def _run_iteration_pytest(
        self,
        goal: str,
        iteration: int,
        current_plan: Plan,
        frozen_tests: CodeArtifact,
        task_id: Optional[str],
    ) -> IterationState:
        """One iteration gated on the frozen pytest suite (single-file Python)."""
        print("\n[EXECUTE] Generating implementation...")
        previous_code = self.memory.get_last_code()
        last_reflection = self.memory.get_last_reflection()

        is_fix = bool(previous_code and last_reflection and not last_reflection.success)
        with self.profiler.track("code_generation"):
            if is_fix:
                code = await self.code_generator.generate_fix(
                    plan=current_plan,
                    broken_code=previous_code,
                    error_type=(last_reflection.error_signature.error_type
                                if last_reflection.error_signature else "TestFailure"),
                    error_message=(last_reflection.error_signature.error_message
                                   if last_reflection.error_signature else ""),
                    reflection=last_reflection.analysis,
                    test_code=frozen_tests.source,
                )
            else:
                code = await self.code_generator.generate(
                    plan=current_plan,
                    previous_attempt=previous_code,
                    error_feedback=last_reflection.suggested_fix if last_reflection else None,
                    test_code=frozen_tests.source,
                )
        print(f"Generated {len(code.source)} characters -> {code.file_path}")
        if self.on_code:
            self.on_code(code)

        # PHASE 3: TEST (real pytest against the frozen suite)
        print("\n[TEST] Running frozen pytest suite...")
        impl_files = {code.file_path: code.source}
        test_files = {frozen_tests.file_path: frozen_tests.source}
        with self.profiler.track("pytest_run"):
            test_results = await self.tester.run_pytest(impl_files, test_files)

        # Dependency recovery: a missing third-party package shows up as a
        # collection ImportError. pytest writes that to the collection longrepr /
        # stdout (not stderr), so search the combined output for the package.
        if not test_results.passed:
            is_import_error = test_results.error_type in ("ModuleNotFoundError", "ImportError")
            err_blob = self._error_blob(test_results)
            if (
                self.dependency_manager and is_import_error
                and self.dependency_manager.should_attempt_recovery(err_blob)
            ):
                packages = self.dependency_manager.extract_packages_from_error(err_blob)
                print(f"Missing dependency detected: {packages}. Installing...")
                with self.profiler.track("dependency_install"):
                    recovery = self.dependency_manager.handle_import_error(err_blob, code.source)
                if recovery.attempted and recovery.success:
                    print(f"Installed {recovery.packages_installed}. Retrying...")
                    with self.profiler.track("pytest_run"):
                        test_results = await self.tester.run_pytest(impl_files, test_files)
            print(
                f"pytest: collected={test_results.tests_collected} "
                f"passed={test_results.tests_passed} failed={test_results.tests_failed} "
                f"errors={test_results.tests_errors}"
            )

        if self.on_test:
            self.on_test(test_results)

        # PHASE 4: REFLECT
        print("\n[REFLECT] Analyzing results...")
        with self.profiler.track("reflection"):
            reflection = await self.reflector.analyze(
                test_results=test_results,
                code=code,
                plan=current_plan,
                iteration=iteration,
                previous_reflections=[s.reflection for s in self.memory.get_recent(3)],
            )
        if reflection.suggested_fix:
            print(f"Suggested fix: {reflection.suggested_fix[:100]}...")
        if self.on_reflect:
            self.on_reflect(reflection)

        status = self._determine_status(test_results, reflection, iteration)
        return self._create_final_state(
            iteration=iteration,
            plan=current_plan,
            code=code,
            test_results=test_results,
            reflection=reflection,
            status=status,
            task_id=task_id,
            test_code=frozen_tests,
        )

    async def _run_iteration_legacy(
        self,
        goal: str,
        iteration: int,
        current_plan: Optional[Plan],
        task_id: Optional[str]
    ) -> IterationState:
        """Run a single iteration of the loop (legacy exit-code behavior)."""

        # PHASE 1: PLAN
        if current_plan is None:
            print("\n[PLAN] Creating plan...")
            current_plan = await self.planner.create_plan(goal)
            print(f"Steps: {len(current_plan.steps)}")
            print(f"Tests: {len(current_plan.test_cases)}")
            
            if self.on_plan:
                self.on_plan(current_plan)
        else:
            print("\n[PLAN] Using refined plan from previous iteration")
        
        # PHASE 2: EXECUTE (Generate Code)
        print("\n[EXECUTE] Generating code...")
        
        # Get context from memory
        previous_code = self.memory.get_last_code()
        last_reflection = self.memory.get_last_reflection()
        
        is_multi_file = getattr(current_plan, 'use_multi_file', False)
        
        if previous_code and last_reflection and not last_reflection.success:
            # This is a fix attempt
            if is_multi_file:
                files = await self.code_generator.generate_files(
                    plan=current_plan,
                    previous_attempt=previous_code,
                    error_feedback=last_reflection.suggested_fix if last_reflection else None
                )
                code = None  # Will use files instead
            else:
                code = await self.code_generator.generate_fix(
                    plan=current_plan,
                    broken_code=previous_code,
                    error_type=last_reflection.error_signature.error_type if last_reflection.error_signature else "Unknown",
                    error_message=last_reflection.error_signature.error_message if last_reflection.error_signature else "",
                    reflection=last_reflection.analysis
                )
                files = None
        else:
            # Fresh generation
            if is_multi_file:
                files = await self.code_generator.generate_files(
                    plan=current_plan,
                    previous_attempt=previous_code,
                    error_feedback=last_reflection.suggested_fix if last_reflection else None
                )
                code = None
            else:
                code = await self.code_generator.generate(
                    plan=current_plan,
                    previous_attempt=previous_code,
                    error_feedback=last_reflection.suggested_fix if last_reflection else None
                )
                files = None
        
        if is_multi_file and files:
            print(f"Generated {len(files)} files")
            # Write files to workspace if executor supports it (Phase 2)
            if hasattr(self.tester.executor, 'write_files') and getattr(self.tester.executor, 'persistent', False):
                self.tester.executor.write_files(files)
                print(f"Written {len(files)} files to workspace")
            # Multi-file mode: write files to workspace and create a representative CodeArtifact
            first_file = next(iter(files))
            code = CodeArtifact(
                source=files[first_file],
                file_path=first_file,
                language=current_plan.language,
                metadata={"generated_files": list(files.keys())}
            )
        elif code:
            print(f"Generated {len(code.source)} characters")
        
        if self.on_code:
            self.on_code(code)
        
        # PHASE 3: TEST
        print("\n[TEST] Running tests...")
        test_results = await self.tester.run_tests(
            code, 
            current_plan, 
            run_from_workspace=is_multi_file
        )
        
        # Phase 2: Automatic package installation recovery (Docker persistent mode).
        # Missing dependencies on the *first* run are expected — we recover automatically
        # instead of treating them as hard failures.
        if not test_results.passed:
            is_import_error = test_results.error_type in ("ModuleNotFoundError", "ImportError")
            can_recover = (
                self.dependency_manager 
                and is_import_error
                and self.dependency_manager.should_attempt_recovery(test_results.stderr)
            )
            
            if can_recover:
                # Friendly path for the common case (FastAPI, pandas, etc. on first run)
                packages = self.dependency_manager.extract_packages_from_error(test_results.stderr)
                print(f"Missing dependency detected: {packages}. Installing inside container...")
                recovery = self.dependency_manager.handle_import_error(test_results.stderr, code.source)
                if recovery.attempted:
                    if recovery.success:
                        print(f"Successfully installed {recovery.packages_installed}. Retrying...")
                        test_results = await self.tester.run_tests(
                            code, current_plan, run_from_workspace=is_multi_file
                        )
                        if test_results.passed:
                            print("Retry passed.")
                        else:
                            print(f"Retry still failing (passed={test_results.passed}).")
                    else:
                        print(f"Package installation failed: {recovery.error}")
            elif self.install_packages and is_import_error:
                # Legacy fallback (non-Docker or older path)
                package = self._extract_package_name(test_results.stderr)
                if package:
                    print(f"Attempting to install missing package: {package}")
                    if self.install_packages([package]):
                        print(f"Successfully installed {package}. Retrying...")
                        test_results = await self.tester.run_tests(
                            code, current_plan, run_from_workspace=is_multi_file
                        )
                        print(f"Retry passed: {test_results.passed}")
            else:
                # Genuine failure (not a recoverable import error, or recovery exhausted)
                print(f"Passed: {test_results.passed}")
                if test_results.error_type:
                    print(f"Error: {test_results.error_type}")
                if test_results.stderr:
                    print(f"Details: {test_results.stderr[:300]}...")
        
        if self.on_test:
            self.on_test(test_results)
        
        # PHASE 4: REFLECT
        print("\n[REFLECT] Analyzing results...")
        reflection = await self.reflector.analyze(
            test_results=test_results,
            code=code,
            plan=current_plan,
            iteration=iteration,
            previous_reflections=[s.reflection for s in self.memory.get_recent(3)]
        )
        
        print(f"Analysis: {reflection.analysis[:100]}...")
        if reflection.suggested_fix:
            print(f"Suggested fix: {reflection.suggested_fix[:100]}...")
        
        if self.on_reflect:
            self.on_reflect(reflection)
        
        # Determine status
        status = self._determine_status(test_results, reflection, iteration)
        
        return self._create_final_state(
            iteration=iteration,
            plan=current_plan,
            code=code,
            test_results=test_results,
            reflection=reflection,
            status=status,
            task_id=task_id
        )
    
    def _prepare_next_plan(self, state: IterationState, goal: str) -> Plan:
        """Prepare plan for next iteration based on reflection."""
        if not state.reflection.suggested_fix:
            return state.plan
        
        # Augment plan with fix suggestion
        new_steps = [
            f"FIX: {state.reflection.suggested_fix}",
            *state.plan.steps
        ]
        
        return Plan(
            goal=goal,
            steps=new_steps,
            test_cases=state.plan.test_cases,
            language=state.plan.language,
            dependencies=state.plan.dependencies,
            context={
                **state.plan.context,
                "iteration": state.iteration,
                "fix_applied": state.reflection.suggested_fix
            }
        )
    
    def _determine_status(
        self,
        test_results: TestResults,
        reflection: Reflection,
        iteration: int
    ) -> Status:
        """Determine the status of the current iteration."""
        if test_results.passed:
            return Status.SUCCESS
        if not reflection.should_continue:
            return Status.FAILED
        if iteration >= self.config.max_iterations:
            return Status.MAX_ITERATIONS
        return Status.IN_PROGRESS
    
    def _create_final_state(
        self,
        iteration: int,
        plan: Plan,
        code: CodeArtifact,
        test_results: TestResults,
        reflection: Reflection,
        status: Status,
        task_id: Optional[str] = None,
        test_code: Optional[CodeArtifact] = None,
    ) -> IterationState:
        """Create an iteration state object."""
        return IterationState(
            iteration=iteration,
            plan=plan,
            code=code,
            test_results=test_results,
            reflection=reflection,
            status=status,
            task_id=task_id,
            test_code=test_code,
        )
    
    def get_stats(self) -> dict:
        """Get loop statistics."""
        return {
            "iterations_completed": len(self.memory),
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "error_history": self.memory.get_error_history()
        }

    def _extract_package_name(self, stderr: str) -> Optional[str]:
        """
        Legacy method - kept only for fallback path.
        New code should use DependencyManager.extract_packages_from_error() instead.
        """
        if not stderr:
            return None

        import re
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr)
        if match:
            name = match.group(1).split(".")[0]
            if name in {"os", "sys", "re", "json", "time", "pathlib"}:
                return None
            return name
        return None
