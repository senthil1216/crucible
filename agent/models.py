"""
Data models for the self-improving coding agent.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from datetime import datetime
from pathlib import Path
import hashlib
import json


class Status(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"
    CIRCUIT_BREAKER = "circuit_breaker"


class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGEROUS = "dangerous"


@dataclass
class Plan:
    """Output of Planner - represents the strategy to achieve a goal."""
    goal: str
    steps: List[str]
    test_cases: List[str]
    language: str = "python"
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    # Phase 2: Multi-file / workspace support
    project_type: str = "general"       # e.g. "fastapi", "python_package", "cli_tool", "general"
    use_multi_file: bool = False        # Whether this task should generate multiple files
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": self.steps,
            "test_cases": self.test_cases,
            "language": self.language,
            "dependencies": self.dependencies,
            "context": self.context,
            "project_type": self.project_type,
            "use_multi_file": self.use_multi_file,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        return cls(**data)


@dataclass
class CodeArtifact:
    """Output of Executor - represents generated code."""
    source: str
    file_path: str
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def write_to_disk(self, base_path: Path) -> Path:
        """Write code to disk and return the path."""
        path = base_path / self.file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.source)
        return path
    
    def get_hash(self) -> str:
        """Get a hash of the source code for deduplication."""
        return hashlib.sha256(self.source.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "file_path": self.file_path,
            "language": self.language,
            "metadata": self.metadata
        }


@dataclass
class TestResults:
    """Output of Tester - represents test execution results.

    When the run came from a real pytest invocation, the `tests_*` counts and
    `test_failures` carry per-test detail. `tests_collected == 0` means the
    suite was empty (or failed to collect) and must never be treated as a pass.
    `failed_tests` keeps the legacy list of failing test node ids for callers
    that only need names.
    """
    passed: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time: float = 0.0
    error_type: Optional[str] = None
    failed_tests: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Real-pytest detail (Phase: real-pytest success gate)
    tests_collected: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_errors: int = 0
    # Each entry: {"nodeid": str, "outcome": str, "message": str}
    test_failures: List[Dict[str, Any]] = field(default_factory=list)
    from_pytest: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "execution_time": self.execution_time,
            "error_type": self.error_type,
            "failed_tests": self.failed_tests,
            "warnings": self.warnings,
            "tests_collected": self.tests_collected,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_errors": self.tests_errors,
            "test_failures": self.test_failures,
            "from_pytest": self.from_pytest,
        }


@dataclass
class ErrorSignature:
    """Normalized error pattern for matching similar failures."""
    error_type: str
    error_message: str
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    
    def normalize(self) -> str:
        """Create a normalized string for similarity comparison.

        Strips only runtime-varying values (memory addresses, file paths,
        object reprs, large numerals) so that the remaining signal — type
        names, attribute names, identifiers — survives. The result is used
        as a grouping key in FailureMemory; collapsing all lowercase
        identifiers to {var} would erase the very thing we want to match on.
        """
        import re

        msg = self.error_message
        # Normalize quotes so "x" and 'x' group together.
        msg = msg.replace("'", '"')
        # Memory addresses: 0x7fabc123 -> {addr}.
        msg = re.sub(r"0x[0-9a-fA-F]+", "{addr}", msg)
        # POSIX absolute paths and Windows drive paths -> {path}.
        msg = re.sub(r"/[^\s'\"<>]+", "{path}", msg)
        msg = re.sub(r"[A-Za-z]:\\[^\s'\"<>]+", "{path}", msg)
        # Object reprs after address normalization: <pkg.Class object at {addr}>.
        msg = re.sub(r"<[\w.]+ object at \{addr\}>", "{obj}", msg)
        # Collapse runs of >=5 digits (timestamps, large IDs) but keep small
        # numbers (line numbers, indices) so "line 12" vs "line 47" still differ.
        msg = re.sub(r"\b\d{5,}\b", "{n}", msg)
        return f"{self.error_type}:{msg}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "line_number": self.line_number,
            "file_path": self.file_path
        }


@dataclass
class Reflection:
    """Output of Reflector - represents analysis of a failure."""
    success: bool
    analysis: str
    error_signature: Optional[ErrorSignature] = None
    root_cause: Optional[str] = None
    hypothesis: Optional[str] = None
    suggested_fix: Optional[str] = None
    should_continue: bool = True
    confidence: float = 0.5
    # ID of the FailureMemory entry written for this iteration's failure (if
    # any). The loop reads this on the next iteration's success to mark the
    # prior failure was_fixed=True.
    failure_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "analysis": self.analysis,
            "error_signature": self.error_signature.to_dict() if self.error_signature else None,
            "root_cause": self.root_cause,
            "hypothesis": self.hypothesis,
            "suggested_fix": self.suggested_fix,
            "should_continue": self.should_continue,
            "confidence": self.confidence,
            "failure_id": self.failure_id,
        }


@dataclass
class Learning:
    """
    A reusable lesson extracted by the Reflector on a successful run.

    Written to long-term memory and surfaced to the Planner on similar future
    tasks. Examples:
      "For FastAPI projects, always include a /health endpoint."
      "When parsing CSVs in Python, prefer csv.DictReader over manual splits."
    """
    lesson: str
    project_type: str = "general"
    language: str = "python"
    tags: List[str] = field(default_factory=list)
    source_task_id: Optional[str] = None
    source_goal: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lesson": self.lesson,
            "project_type": self.project_type,
            "language": self.language,
            "tags": self.tags,
            "source_task_id": self.source_task_id,
            "source_goal": self.source_goal,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Learning":
        ts = data.get("timestamp")
        return cls(
            lesson=data["lesson"],
            project_type=data.get("project_type", "general"),
            language=data.get("language", "python"),
            tags=data.get("tags", []) or [],
            source_task_id=data.get("source_task_id"),
            source_goal=data.get("source_goal"),
            timestamp=datetime.fromisoformat(ts) if ts else datetime.now(),
        )


@dataclass
class IterationState:
    """Complete state of one loop iteration - allows resumability."""
    iteration: int
    plan: Plan
    code: CodeArtifact
    test_results: TestResults
    reflection: Reflection
    status: Status
    timestamp: datetime = field(default_factory=datetime.now)
    task_id: Optional[str] = None
    # Frozen pytest suite for this task. Generated once (test-first) and reused
    # unchanged across fix iterations so the agent fixes code, never the tests.
    test_code: Optional[CodeArtifact] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "plan": self.plan.to_dict(),
            "code": self.code.to_dict(),
            "test_results": self.test_results.to_dict(),
            "reflection": self.reflection.to_dict(),
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "test_code": self.test_code.to_dict() if self.test_code else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationState":
        test_code_data = data.get("test_code")
        return cls(
            iteration=data["iteration"],
            plan=Plan.from_dict(data["plan"]),
            code=CodeArtifact(**data["code"]),
            test_results=TestResults(**data["test_results"]),
            reflection=Reflection(**data["reflection"]),
            status=Status(data["status"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            task_id=data.get("task_id"),
            test_code=CodeArtifact(**test_code_data) if test_code_data else None,
        )


@dataclass
class SafetyReport:
    """Output of SafetyChecker - represents safety analysis."""
    level: SafetyLevel
    warnings: List[str] = field(default_factory=list)
    dangerous_operations: List[str] = field(default_factory=list)
    requires_approval: bool = False
    
    def is_safe(self) -> bool:
        return self.level == SafetyLevel.SAFE


@dataclass
class MemoryEntry:
    """Base class for memory entries."""
    id: str
    content: Dict[str, Any]
    embedding: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass  
class LoopConfig:
    """Configuration for the execution loop."""
    max_iterations: int = 10
    stop_on_success: bool = True
    enable_reflection: bool = True
    checkpoint_interval: int = 1  # Save state every N iterations

    # Real-pytest gate: how many times to regenerate a rejected test suite
    # (structurally invalid, or vacuous against a stub) before giving up /
    # proceeding with a warning.
    max_test_regenerations: int = 2
    
    # Circuit breaker settings
    failure_threshold: int = 3
    failure_window: int = 5  # Check last N iterations
    cooldown_period: int = 60  # Seconds to wait after circuit opens


@dataclass
class AgentConfig:
    """Main configuration for the agent."""
    loop: LoopConfig = field(default_factory=LoopConfig)
    workspace_path: Path = field(default_factory=lambda: Path("./workspace"))
    state_path: Path = field(default_factory=lambda: Path("./.agent_state"))
    memory_path: Path = field(default_factory=lambda: Path("./.agent_memory"))
    
    # Safety settings
    enable_sandbox: bool = True
    sandbox_timeout: int = 30
    sandbox_memory_limit: str = "512m"
    sandbox_cpu_limit: float = 1.0

    # Docker sandbox settings (Phase 1+)
    use_docker: bool = False
    docker_image: str = "python:3.12-slim"

    # Phase 2 Persistent Docker settings
    docker_persistent: bool = False          # One container per task instead of ephemeral
    docker_enable_network: bool = True       # Needed for pip install etc. in persistent mode
    docker_install_build_tools: bool = True  # Install build-essential etc. on persistent start

    # Run-the-app: after a server task passes its tests, launch it and leave it
    # running, reachable from the host. Requires docker_persistent.
    run_app: bool = False
    app_port: int = 8000
    
    # LLM settings
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    
    def __post_init__(self):
        # Ensure paths are Path objects
        if isinstance(self.workspace_path, str):
            self.workspace_path = Path(self.workspace_path)
        if isinstance(self.state_path, str):
            self.state_path = Path(self.state_path)
        if isinstance(self.memory_path, str):
            self.memory_path = Path(self.memory_path)
