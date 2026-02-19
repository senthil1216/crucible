"""
State Persistence: Saves and resumes agent state.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from agent.models import IterationState, Status


class StateManager:
    """
    Manages persistent state storage for the agent.
    Allows resuming after interruptions.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _get_task_dir(self, task_id: str) -> Path:
        """Get directory for a specific task."""
        task_dir = self.storage_path / task_id
        task_dir.mkdir(exist_ok=True)
        return task_dir
    
    async def save_checkpoint(
        self,
        task_id: str,
        state: IterationState
    ) -> Path:
        """
        Save a checkpoint of the current state.
        
        Args:
            task_id: Unique identifier for this task
            state: Current iteration state
        
        Returns:
            Path to saved checkpoint file
        """
        task_dir = self._get_task_dir(task_id)
        
        # Save with timestamp for history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = task_dir / f"checkpoint_{state.iteration}_{timestamp}.json"
        
        # Also save as "latest" for easy resuming
        latest_file = task_dir / "latest.json"
        
        data = state.to_dict()
        
        # Save checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Update latest
        with open(latest_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return checkpoint_file
    
    async def load_checkpoint(
        self,
        task_id: str,
        iteration: Optional[int] = None
    ) -> Optional[IterationState]:
        """
        Load a checkpoint for a task.
        
        Args:
            task_id: Task identifier
            iteration: Specific iteration to load, or None for latest
        
        Returns:
            IterationState if found, None otherwise
        """
        task_dir = self._get_task_dir(task_id)
        
        if iteration is not None:
            # Find specific iteration
            pattern = f"checkpoint_{iteration}_*.json"
            files = list(task_dir.glob(pattern))
            if files:
                checkpoint_file = sorted(files)[-1]  # Most recent
            else:
                return None
        else:
            # Load latest
            checkpoint_file = task_dir / "latest.json"
            if not checkpoint_file.exists():
                return None
        
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        return IterationState.from_dict(data)
    
    async def list_checkpoints(self, task_id: str) -> list[Dict[str, Any]]:
        """List all available checkpoints for a task."""
        task_dir = self._get_task_dir(task_id)
        
        checkpoints = []
        for file in task_dir.glob("checkpoint_*.json"):
            # Parse filename: checkpoint_{iteration}_{timestamp}.json
            parts = file.stem.split('_')
            if len(parts) >= 3:
                iteration = int(parts[1])
                timestamp = '_'.join(parts[2:])
                checkpoints.append({
                    "iteration": iteration,
                    "timestamp": timestamp,
                    "file": str(file)
                })
        
        return sorted(checkpoints, key=lambda x: x["iteration"])
    
    async def save_task_metadata(
        self,
        task_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Save metadata about a task."""
        task_dir = self._get_task_dir(task_id)
        metadata_file = task_dir / "metadata.json"
        
        data = {
            "task_id": task_id,
            "updated_at": datetime.now().isoformat(),
            **metadata
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def load_task_metadata(
        self,
        task_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load metadata for a task."""
        task_dir = self._get_task_dir(task_id)
        metadata_file = task_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def list_tasks(self) -> list[str]:
        """List all task IDs with saved state."""
        if not self.storage_path.exists():
            return []
        
        return [d.name for d in self.storage_path.iterdir() if d.is_dir()]
    
    async def cleanup_old_checkpoints(
        self,
        task_id: str,
        keep_last: int = 5
    ) -> int:
        """
        Clean up old checkpoints, keeping only the most recent N.
        
        Returns:
            Number of files removed
        """
        checkpoints = await self.list_checkpoints(task_id)
        
        if len(checkpoints) <= keep_last:
            return 0
        
        to_remove = checkpoints[:-keep_last]
        removed = 0
        
        for cp in to_remove:
            file_path = Path(cp["file"])
            if file_path.exists():
                file_path.unlink()
                removed += 1
        
        return removed
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete all state for a task."""
        import shutil
        
        task_dir = self._get_task_dir(task_id)
        
        if task_dir.exists():
            shutil.rmtree(task_dir)
            return True
        return False
