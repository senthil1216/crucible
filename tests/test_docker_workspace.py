"""
Basic tests for Phase 2 workspace methods in DockerExecutor.

Note: Full end-to-end tests require a running Docker daemon and are marked
as integration tests. These tests focus on logic and error handling.
"""

import pytest
from unittest.mock import MagicMock, patch

from agent.executor.docker_executor import DockerExecutor


class TestDockerWorkspaceMethods:
    """Tests for workspace-related methods in DockerExecutor."""

    @pytest.fixture
    def mock_executor(self):
        """Create a DockerExecutor with a mocked persistent container."""
        with patch("agent.executor.docker_executor.DOCKER_AVAILABLE", True), \
             patch("agent.executor.docker_executor.docker") as mock_docker:
            executor = DockerExecutor(docker_client=MagicMock())
            executor.persistent = True
            executor._persistent_container = MagicMock()
            executor._workspace_path = "/workspace/test-task-123"
            yield executor

    def test_get_workspace_path_returns_correct_path(self, mock_executor):
        assert mock_executor.get_workspace_path() == "/workspace/test-task-123"

    def test_get_workspace_path_returns_none_when_not_persistent(self, mock_executor):
        mock_executor.persistent = False
        assert mock_executor.get_workspace_path() is None

    def test_write_file_returns_false_when_not_persistent(self, mock_executor):
        mock_executor.persistent = False
        result = mock_executor.write_file("main.py", "print('hello')")
        assert result is False

    def test_read_file_returns_none_when_not_persistent(self, mock_executor):
        mock_executor.persistent = False
        result = mock_executor.read_file("main.py")
        assert result is None

    def test_list_dir_returns_empty_when_not_persistent(self, mock_executor):
        mock_executor.persistent = False
        result = mock_executor.list_dir(".")
        assert result == []

    def test_create_directory_returns_false_when_not_persistent(self, mock_executor):
        mock_executor.persistent = False
        result = mock_executor.create_directory("src")
        assert result is False

    def test_capture_environment_returns_empty_when_not_persistent(self, mock_executor):
        mock_executor.persistent = False
        result = mock_executor.capture_environment()
        assert result == {"installed_packages": [], "workspace_files": []}

    def test_capture_environment_parses_pip_freeze(self, mock_executor):
        """capture_environment should return pip names (lowercase, no versions)
        and shallow workspace listing."""
        pip_output = b"fastapi==0.110.0\nuvicorn==0.29.0\nPydantic==2.6.0\n"
        ls_output = b"main.py\nrequirements.txt\n"

        def fake_exec_run(cmd, **kwargs):
            result = MagicMock()
            joined = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "pip list" in joined:
                result.exit_code = 0
                result.output = (pip_output, b"")
            elif "ls" in joined:
                result.exit_code = 0
                result.output = (ls_output, b"")
            else:
                result.exit_code = 0
                result.output = (b"", b"")
            return result

        mock_executor._persistent_container.exec_run.side_effect = fake_exec_run

        result = mock_executor.capture_environment()
        assert "fastapi" in result["installed_packages"]
        assert "uvicorn" in result["installed_packages"]
        # Pydantic is lowercased on the way out
        assert "pydantic" in result["installed_packages"]
        # No versions
        assert all("==" not in pkg for pkg in result["installed_packages"])
        assert "main.py" in result["workspace_files"]
        assert "requirements.txt" in result["workspace_files"]
