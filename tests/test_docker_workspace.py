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

    # TODO: Add more tests with mocked exec_run and put_archive
    # for successful write/read/list/create scenarios.
