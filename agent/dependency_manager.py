"""
DependencyManager

Handles automatic detection and installation of Python dependencies
during agent execution (Phase 1 focus: simple, fast, and reliable recovery).

Scope for Phase 1:
- Python packages only (pip)
- Resolve with wildcard (*) for simplicity
- Maximum 4 automatic install attempts per task
- Persist successful installations to long-term memory (future)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, List

from agent.executor.docker_executor import DockerExecutor


@dataclass
class RecoveryAttempt:
    """Represents a single automatic recovery attempt."""
    packages_tried: List[str]
    success: bool
    error: Optional[str] = None
    failure_reason: Optional[str] = None


@dataclass
class RecoveryResult:
    """Result of attempting to recover from a dependency-related failure."""
    attempted: bool = False
    packages_installed: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    attempts: List[RecoveryAttempt] = field(default_factory=list)


@dataclass
class InstallResult:
    """Result of a package installation attempt."""
    success: bool = False
    packages: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    # One of: "not_found", "build_error", "network", "permission", "unknown",
    # "not_persistent", "user_declined".
    failure_reason: Optional[str] = None


class DependencyManager:
    """
    Manages dependency detection and installation for the agent.

    Designed to be simple and fast in Phase 1 while providing a foundation
    for more sophisticated behavior later.
    """

    # Common import name -> PyPI package name mappings (Task 2)
    # Format: import_name -> list of possible PyPI package names (in preference order)
    PACKAGE_NAME_MAP: dict[str, list[str]] = {
        "cv2": ["opencv-python", "opencv-contrib-python"],
        "pil": ["Pillow"],
        "pillow": ["Pillow"],
        "yaml": ["PyYAML"],
        "sklearn": ["scikit-learn"],
        "bs4": ["beautifulsoup4"],
        "lxml": ["lxml"],
        "psycopg2": ["psycopg2-binary"],
        "mysql": ["mysql-connector-python"],
        "pymongo": ["pymongo"],
        "redis": ["redis"],
        "tensorflow": ["tensorflow", "tensorflow-cpu"],
        "torch": ["torch"],
        "torchvision": ["torchvision"],
        "cv": ["opencv-python"],
    }

    def __init__(
        self,
        executor: DockerExecutor,
        ask_before_install: bool = False,
        confirm_fn: Optional[Callable[[List[str]], bool]] = None,
    ):
        self.executor = executor
        self._attempt_count: int = 0
        self._max_attempts: int = 4
        self._attempts: List[RecoveryAttempt] = []
        # Track B: every package successfully installed this task, so the
        # Reflector / pattern store can persist them (via environment_context).
        self._installed_packages: List[str] = []
        # Track B: optional interactive confirmation before any pip install.
        self.ask_before_install = ask_before_install
        self._confirm_fn = confirm_fn or self._default_confirm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_import_error(
        self, error_message: str, current_code: str = ""
    ) -> RecoveryResult:
        """
        Main entry point when an import-related error is detected.

        This method owns the decision logic for whether to attempt recovery
        and tracks attempts for the current task.
        """
        # Check if we've already hit the attempt limit
        if self._attempt_count >= self._max_attempts:
            return RecoveryResult(
                attempted=False,
                success=False,
                error=f"Maximum automatic install attempts ({self._max_attempts}) reached for this task.",
                attempts=self._attempts.copy(),
            )

        # Try to extract packages from the error
        packages = self.extract_packages_from_error(error_message)
        if not packages:
            return RecoveryResult(
                attempted=False,
                success=False,
                error="Could not extract any package names from the error message.",
                attempts=self._attempts.copy(),
            )

        self._attempt_count += 1

        # Attempt installation
        install_result = self.install_packages(packages)

        # Record this attempt
        attempt = RecoveryAttempt(
            packages_tried=packages,
            success=install_result.success,
            error=install_result.stderr if not install_result.success else None,
            failure_reason=install_result.failure_reason,
        )
        self._attempts.append(attempt)

        return RecoveryResult(
            attempted=True,
            packages_installed=install_result.packages if install_result.success else [],
            success=install_result.success,
            error=install_result.stderr if not install_result.success else None,
            attempts=self._attempts.copy(),
        )

    def install_packages(self, packages: List[str]) -> InstallResult:
        """
        Install a list of Python packages using the executor.

        Strategy (simple & fast):
        - Use common name mappings (Task 2)
        - Try candidates in order until one succeeds
        - Let pip choose the latest compatible version (no pinning)
        - Categorize the failure (not_found / build_error / network / permission)
          from pip's stderr so the Reflector can react per-category
        """
        if not packages:
            return InstallResult(success=True)

        if not getattr(self.executor, "persistent", False):
            return InstallResult(
                success=False,
                packages=packages,
                stderr="install_packages() requires a persistent container (docker_persistent=True).",
                failure_reason="not_persistent",
            )

        # Track B: optional human-in-the-loop gate before touching the network.
        if self.ask_before_install and not self._confirm_fn(list(packages)):
            return InstallResult(
                success=False,
                packages=packages,
                stderr="Install declined by user.",
                failure_reason="user_declined",
            )

        all_installed: List[str] = []
        last_error = ""
        last_reason = "unknown"

        for pkg in packages:
            candidates = self._resolve_package_candidates(pkg)

            installed = False
            for candidate in candidates:
                try:
                    success, _stdout, stderr = self.executor.install_packages_detailed(
                        [candidate]
                    )
                    if success:
                        all_installed.append(candidate)
                        self._record_installed(candidate)
                        installed = True
                        break
                    else:
                        last_error = stderr or f"Failed to install {candidate}"
                        last_reason = self.classify_pip_failure(stderr)

                except Exception as e:
                    last_error = str(e)
                    last_reason = "unknown"

            if not installed:
                # None of the candidates for this package worked
                return InstallResult(
                    success=False,
                    packages=all_installed,  # return what we did manage to install
                    stderr=last_error,
                    failure_reason=last_reason,
                )

        return InstallResult(
            success=True,
            packages=all_installed,
        )

    def install_from_requirements(
        self, relative_path: str = "requirements.txt"
    ) -> InstallResult:
        """
        Install packages from a generated requirements.txt inside the workspace.

        On-demand by default — the loop decides when to call this (e.g. gated by
        `AgentConfig.docker_auto_install_requirements`). Failures are categorized
        the same way as `install_packages`.
        """
        if not getattr(self.executor, "persistent", False):
            return InstallResult(
                success=False,
                stderr="install_from_requirements() requires a persistent container.",
                failure_reason="not_persistent",
            )

        if self.ask_before_install and not self._confirm_fn([f"-r {relative_path}"]):
            return InstallResult(
                success=False,
                stderr="Install declined by user.",
                failure_reason="user_declined",
            )

        try:
            success, stdout, stderr = self.executor.install_requirements_file_detailed(
                relative_path
            )
        except Exception as e:
            return InstallResult(success=False, stderr=str(e), failure_reason="unknown")

        if not success:
            return InstallResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                failure_reason=self.classify_pip_failure(stderr),
            )
        return InstallResult(success=True, stdout=stdout)

    # ------------------------------------------------------------------
    # Failure categorization (Track B)
    # ------------------------------------------------------------------

    @staticmethod
    def classify_pip_failure(stderr: str) -> str:
        """Map pip's stderr to a coarse failure category.

        Returns one of: "permission", "not_found", "network", "build_error",
        "unknown". Order matters — the most specific / least ambiguous signals
        are checked first so a not_found error that happens to retry over the
        network isn't misread as a transient network failure.
        """
        if not stderr:
            return "unknown"
        text = stderr.lower()

        if "permission denied" in text or "errno 13" in text or "eacces" in text:
            return "permission"
        if (
            "no matching distribution found" in text
            or "could not find a version that satisfies" in text
        ):
            return "not_found"
        if (
            "temporary failure in name resolution" in text
            or "could not fetch url" in text
            or "connection refused" in text
            or "network is unreachable" in text
            or "failed to establish a new connection" in text
            or "timed out" in text
            or "retrying" in text
        ):
            return "network"
        if (
            "failed building wheel" in text
            or "subprocess-exited-with-error" in text
            or "microsoft visual c++" in text
            or "legacy-install-failure" in text
            or re.search(r"command '[^']+' failed", text)
            or "gcc" in text
            or "clang" in text
        ):
            return "build_error"
        return "unknown"

    # ------------------------------------------------------------------
    # Install confirmation + tracking (Track B)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_confirm(packages: List[str]) -> bool:
        """Interactive prompt used when ask_before_install is on and no custom
        confirm_fn was supplied. Defaults to 'no' on EOF/non-interactive."""
        try:
            answer = input(f"Install {', '.join(packages)}? [y/N] ").strip().lower()
        except EOFError:
            return False
        return answer in ("y", "yes")

    def _record_installed(self, candidate: str) -> None:
        if candidate and candidate not in self._installed_packages:
            self._installed_packages.append(candidate)

    @property
    def installed_packages(self) -> List[str]:
        """Packages successfully installed during the current task (cumulative
        until reset). Fed into the Pattern's environment_context on success."""
        return list(self._installed_packages)

    def _resolve_package_candidates(self, package: str) -> List[str]:
        """
        Given an import name, return a list of PyPI package names to try,
        in order of preference.
        """
        package = package.lower().strip()

        if package in self.PACKAGE_NAME_MAP:
            return self.PACKAGE_NAME_MAP[package]

        # Default: just try the name as-is
        return [package]

    # ------------------------------------------------------------------
    # Extraction Logic (Task 1)
    # ------------------------------------------------------------------

    def extract_packages_from_error(self, error_message: str) -> List[str]:
        """
        Extract top-level Python package names from an error message.

        This is an improved version for Phase 1 (Task 1).
        """
        if not error_message:
            return []

        packages = set()

        # Common patterns for missing modules / import errors
        patterns = [
            # Standard "No module named 'xxx'"
            r"No module named ['\"]([^'\"]+)['\"]",
            r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
            r"ImportError: No module named ['\"]([^'\"]+)['\"]",

            # "cannot import name 'xxx' from 'yyy'"
            r"cannot import name ['\"]?([^'\" ]+)['\"]? from ['\"]([^'\"]+)['\"]",

            # "from 'xxx' import yyy" style errors
            r"from ['\"]([^'\"]+)['\"] import",

            # More verbose Python 3.12+ style
            r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"] \(from ['\"]([^'\"]+)['\"]\)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, error_message, re.IGNORECASE)
            for match in matches:
                # `re.findall` can return tuples when there are multiple groups
                if isinstance(match, tuple):
                    # Prefer the imported-from module if present (e.g. "from 'foo' import bar")
                    candidate = match[1] if len(match) > 1 and match[1] else match[0]
                else:
                    candidate = match

                pkg = self._normalize_package_name(candidate)
                if pkg:
                    packages.add(pkg)

        return sorted(packages)

    def _normalize_package_name(self, name: str) -> Optional[str]:
        """
        Normalize a potential package name.

        - Takes only the top-level package (e.g. 'foo.bar' -> 'foo')
        - Filters out obvious stdlib / internal / built-in names
        """
        if not name:
            return None

        # Take top-level package only
        top_level = name.split(".")[0].strip().lower()

        # Common stdlib and internal modules to ignore
        stdlib_blacklist = {
            "os", "sys", "re", "json", "time", "datetime", "pathlib",
            "collections", "typing", "subprocess", "threading", "queue",
            "argparse", "logging", "unittest", "pytest", "builtins",
            "__main__", "abc", "copy", "functools", "itertools", "io",
            "urllib", "http", "email", "xml", "html", "sqlite3", "tkinter",
            "multiprocessing", "concurrent", "asyncio", "importlib",
        }

        if top_level in stdlib_blacklist:
            return None

        # Filter out names that are clearly not valid package names
        if not top_level.isidentifier() or top_level.startswith("_"):
            return None

        return top_level

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset_attempt_count(self) -> None:
        """Reset attempt tracking for a new task."""
        self._attempt_count = 0
        self._attempts = []
        self._installed_packages = []

    @property
    def attempts_remaining(self) -> int:
        return max(0, self._max_attempts - self._attempt_count)

    def should_attempt_recovery(self, error_message: str) -> bool:
        """
        Returns whether the manager thinks we should try to recover
        from the given error (based on attempt count and error type).
        """
        if self._attempt_count >= self._max_attempts:
            return False

        # Only attempt recovery for import-related errors
        if "ModuleNotFoundError" in error_message or "ImportError" in error_message:
            packages = self.extract_packages_from_error(error_message)
            return len(packages) > 0

        return False

    def get_recent_attempts(self, n: int = 5) -> List[RecoveryAttempt]:
        """Return the most recent recovery attempts."""
        return self._attempts[-n:]
