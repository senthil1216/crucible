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
    """Output of Tester - represents test execution results."""
    passed: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time: float = 0.0
    error_type: Optional[str] = None
    failed_tests: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "execution_time": self.execution_time,
            "error_type": self.error_type,
            "failed_tests": self.failed_tests,
            "warnings": self.warnings
        }


@dataclass
class ErrorSignature:
    """Normalized error pattern for matching similar failures."""
    error_type: str
    error_message: str
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    
    def normalize(self) -> str:
        """Create a normalized string for similarity comparison."""
        # Remove specific variable names, line numbers, etc.
        msg = self.error_message.lower()
        # Normalize quotes
        msg = msg.replace("'", '"')
        # Normalize specific values
        import re
        msg = re.sub(r'\b\d+\b', '{num}', msg)
        msg = re.sub(r'\b[a-z_][a-z0-9_]*\b', '{var}', msg)
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "analysis": self.analysis,
            "error_signature": self.error_signature.to_dict() if self.error_signature else None,
            "root_cause": self.root_cause,
            "hypothesis": self.hypothesis,
            "suggested_fix": self.suggested_fix,
            "should_continue": self.should_continue,
            "confidence": self.confidence
        }


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "plan": self.plan.to_dict(),
            "code": self.code.to_dict(),
            "test_results": self.test_results.to_dict(),
            "reflection": self.reflection.to_dict(),
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationState":
        return cls(
            iteration=data["iteration"],
            plan=Plan.from_dict(data["plan"]),
            code=CodeArtifact(**data["code"]),
            test_results=TestResults(**data["test_results"]),
            reflection=Reflection(**data["reflection"]),
            status=Status(data["status"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            task_id=data.get("task_id")
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
