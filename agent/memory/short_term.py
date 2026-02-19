"""
Short-term memory: Rolling window of recent context.
Keeps the agent focused on the current task.
"""

from collections import deque
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from agent.models import IterationState, Reflection, Plan


class ShortTermMemory:
    """
    Maintains a rolling window of the last N iterations.
    Provides quick access to recent context.
    """
    
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.context_window: deque[IterationState] = deque(maxlen=max_history)
        self.current_plan: Optional[Plan] = None
    
    def add(self, state: IterationState) -> None:
        """Add a new iteration state to memory."""
        self.context_window.append(state)
        self.current_plan = state.plan
    
    def get_recent(self, n: int = None) -> List[IterationState]:
        """Get the last N iteration states."""
        if n is None:
            n = self.max_history
        return list(self.context_window)[-n:]
    
    def get_last_reflection(self) -> Optional[Reflection]:
        """Get the most recent reflection."""
        if not self.context_window:
            return None
        return self.context_window[-1].reflection
    
    def get_last_code(self) -> Optional[str]:
        """Get the most recent code attempt."""
        if not self.context_window:
            return None
        return self.context_window[-1].code.source
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """Get a summary of recent errors."""
        errors = []
        for state in self.context_window:
            if not state.test_results.passed:
                errors.append({
                    "iteration": state.iteration,
                    "error_type": state.test_results.error_type,
                    "error_message": state.test_results.stderr[:200]
                })
        return errors
    
    def is_repeating_errors(self, window: int = 3) -> bool:
        """Check if we're seeing the same error repeatedly."""
        recent = list(self.context_window)[-window:]
        if len(recent) < 2:
            return False
        
        error_types = [s.test_results.error_type for s in recent 
                      if not s.test_results.passed]
        return len(error_types) >= 2 and len(set(error_types)) == 1
    
    def get_context_string(self, include_code: bool = False) -> str:
        """Get a formatted string of recent context for LLM prompts."""
        lines = []
        for state in self.context_window:
            lines.append(f"\n--- Iteration {state.iteration} ---")
            lines.append(f"Status: {state.status.value}")
            lines.append(f"Tests passed: {state.test_results.passed}")
            
            if not state.test_results.passed:
                lines.append(f"Error: {state.test_results.error_type}")
                lines.append(f"Error details: {state.test_results.stderr[:300]}")
            
            if state.reflection.suggested_fix:
                lines.append(f"Suggested fix: {state.reflection.suggested_fix}")
            
            if include_code:
                lines.append(f"Code:\n{state.code.source}")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all short-term memory."""
        self.context_window.clear()
        self.current_plan = None
    
    def __len__(self) -> int:
        return len(self.context_window)
