"""
Integration tests for the full agent.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path

from agent import SelfImprovingAgent, AgentConfig, LoopConfig
from agent.llm_clients import MockLLMClient
from agent.models import Status


class TestAgentIntegration:
    """Integration tests for the full agent."""
    
    @pytest.fixture
    def temp_workspace(self):
        path = Path(tempfile.mkdtemp())
        yield path
        shutil.rmtree(path)
    
    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient()
    
    @pytest.fixture
    def agent(self, temp_workspace, mock_llm):
        config = AgentConfig(
            workspace_path=temp_workspace,
            memory_path=temp_workspace / ".memory",
            state_path=temp_workspace / ".state",
            loop=LoopConfig(max_iterations=3)
        )
        return SelfImprovingAgent(llm_client=mock_llm, config=config)
    
    @pytest.mark.asyncio
    async def test_solve_simple_task(self, agent):
        result = await agent.solve("Create a function that adds two numbers")
        
        assert result is not None
        assert result.plan is not None
        assert result.code is not None
        assert len(result.code.source) > 0
    
    @pytest.mark.asyncio
    async def test_task_with_state_persistence(self, agent):
        task_id = "test_task_123"
        
        # Run task
        result = await agent.solve(
            "Create a sort function",
            task_id=task_id
        )
        
        # Check state was saved
        history = await agent.get_task_history(task_id)
        assert history is not None
        assert history["metadata"] is not None
    
    @pytest.mark.asyncio
    async def test_memory_learning(self, agent):
        # First task
        result1 = await agent.solve("Create add function")
        
        # Second similar task should retrieve pattern
        result2 = await agent.solve("Create a function to add numbers")
        
        # Stats should show patterns
        stats = agent.get_stats()
        assert stats["memory"]["patterns"]["total_patterns"] >= 1
    
    @pytest.mark.asyncio
    async def test_stats(self, agent):
        # Run a task
        await agent.solve("Test task")
        
        # Get stats
        stats = agent.get_stats()
        
        assert "memory" in stats
        assert "patterns" in stats["memory"]
        assert "failures" in stats["memory"]


@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test a complete workflow."""
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Setup
        llm = MockLLMClient()
        config = AgentConfig(
            workspace_path=temp_dir / "workspace",
            memory_path=temp_dir / "memory",
            state_path=temp_dir / "state",
            loop=LoopConfig(max_iterations=2)
        )
        
        agent = SelfImprovingAgent(llm_client=llm, config=config)
        
        # Execute
        result = await agent.solve("Create hello world function")
        
        # Verify
        assert result.plan.goal == "Create hello world function"
        assert result.code.language == "python"
        assert "def" in result.code.source or "print" in result.code.source
        
    finally:
        shutil.rmtree(temp_dir)
