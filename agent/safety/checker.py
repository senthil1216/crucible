"""
Safety checker - analyzes code for dangerous operations before execution.
"""

import ast
import re
from typing import List, Set
from pathlib import Path

from agent.models import CodeArtifact, SafetyReport, SafetyLevel


class SafetyChecker:
    """
    Static analysis to detect potentially dangerous operations.
    Implements defense in depth: catch issues before they reach sandbox.
    """
    
    # Dangerous builtins that require scrutiny
    DANGEROUS_BUILTINS = {
        'eval', 'exec', 'compile', '__import__', 'open',
        'input', 'raw_input', 'reload'
    }
    
    # Dangerous modules/patterns
    DANGEROUS_MODULES = {
        'os.system', 'os.popen', 'os.spawn', 'os.exec',
        'subprocess.call', 'subprocess.run', 'subprocess.Popen',
        'subprocess.check_output', 'subprocess.check_call',
        'sys.exit', 'quit', 'exit',
        'socket.socket', 'socket.create_connection',
        'urllib.request.urlopen', 'urllib.urlopen',
        'requests.get', 'requests.post', 'requests.put', 'requests.delete',
        'httpx.get', 'httpx.post',
        'ftplib.FTP', 'smtplib.SMTP',
        'shutil.rmtree', 'shutil.move',
        'pathlib.Path.unlink', 'os.remove', 'os.unlink', 'os.rmdir',
        'os.rename', 'os.replace',
    }
    
    # File system operations that need approval
    FILESYSTEM_OPERATIONS = {
        'open', 'os.open', 'os.mkdir', 'os.makedirs',
        'os.remove', 'os.unlink', 'os.rmdir', 'os.removedirs',
        'shutil.copy', 'shutil.move', 'shutil.rmtree',
        'pathlib.Path.write_text', 'pathlib.Path.write_bytes',
    }
    
    # Network operations
    NETWORK_OPERATIONS = {
        'socket', 'urllib', 'http.client', 'ftplib', 'smtplib',
        'requests', 'httpx', 'aiohttp'
    }
    
    def __init__(self, project_dir: Path = None):
        self.project_dir = project_dir or Path.cwd()
    
    def analyze(self, code: CodeArtifact) -> SafetyReport:
        """
        Analyze code and return safety report.
        """
        warnings = []
        dangerous_ops = []
        requires_approval = False
        
        try:
            tree = ast.parse(code.source)
        except SyntaxError as e:
            return SafetyReport(
                level=SafetyLevel.WARNING,
                warnings=[f"Syntax error in code: {e}"],
                requires_approval=False
            )
        
        # Walk the AST
        for node in ast.walk(tree):
            # Check for dangerous calls
            if isinstance(node, ast.Call):
                op = self._get_call_name(node)
                if op:
                    if self._is_dangerous(op):
                        dangerous_ops.append(op)
                        warnings.append(f"Dangerous operation detected: {op}")
                        requires_approval = True
                    elif self._is_filesystem_op(op):
                        # Check if it's within project directory
                        if not self._is_safe_filesystem_call(node):
                            warnings.append(f"Filesystem operation may access outside project: {op}")
                            requires_approval = True
                    elif self._is_network_op(op):
                        warnings.append(f"Network operation detected: {op}")
                        requires_approval = True
            
            # Check for imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.NETWORK_OPERATIONS:
                        warnings.append(f"Network module imported: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base = node.module.split('.')[0]
                    if base in self.NETWORK_OPERATIONS:
                        warnings.append(f"Network module imported: {node.module}")
        
        # Check for patterns in raw text (catches obfuscated code)
        text_warnings = self._pattern_analysis(code.source)
        warnings.extend(text_warnings)
        
        # Determine safety level
        if dangerous_ops:
            level = SafetyLevel.DANGEROUS
        elif warnings:
            level = SafetyLevel.WARNING
        else:
            level = SafetyLevel.SAFE
        
        return SafetyReport(
            level=level,
            warnings=warnings,
            dangerous_operations=dangerous_ops,
            requires_approval=requires_approval
        )
    
    def _get_call_name(self, node: ast.Call) -> str:
        """Extract the full name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return None
    
    def _is_dangerous(self, op: str) -> bool:
        """Check if operation is in dangerous list."""
        return any(op.startswith(d) or op == d for d in self.DANGEROUS_BUILTINS) or \
               any(op.startswith(d) for d in self.DANGEROUS_MODULES)
    
    def _is_filesystem_op(self, op: str) -> bool:
        """Check if operation is a filesystem operation."""
        return any(op.startswith(f) or op == f for f in self.FILESYSTEM_OPERATIONS)
    
    def _is_network_op(self, op: str) -> bool:
        """Check if operation is a network operation."""
        return any(op.startswith(n) for n in self.NETWORK_OPERATIONS)
    
    def _is_safe_filesystem_call(self, node: ast.Call) -> bool:
        """
        Check if a filesystem call appears to be within project directory.
        This is a heuristic - not foolproof.
        """
        # For now, be conservative and require approval for all file ops
        # In production, could analyze path arguments
        return False

    # SAFETY CAVEAT: This whole module is best-effort static analysis, not a
    # security boundary. The AST walker only inspects direct ast.Call nodes,
    # so the following all bypass it:
    #   o = open; o(...)             # local rebinding
    #   import subprocess as s; s.run(...)   # import aliasing
    #   getattr(__builtins__, "exec")(...)   # dynamic attribute access
    #   __import__("os").system(...)         # dynamic imports
    # Treat warnings as advisory. The real isolation boundary is the Docker
    # executor (see agent/executor/sandbox.py and docker/).
    
    def _pattern_analysis(self, source: str) -> List[str]:
        """Analyze source code for suspicious patterns."""
        warnings = []
        
        # Check for encoded/encrypted code (simple heuristics)
        suspicious_patterns = [
            (r'base64\.(b64decode|decode)', "Base64 decoding detected"),
            (r'exec\s*\(', "Dynamic code execution with exec"),
            (r'eval\s*\(', "Dynamic code execution with eval"),
            (r'__import__', "Dynamic import detected"),
            (r'importlib', "Dynamic import module usage"),
            (r'getattr\s*\([^,]+,\s*[\'"]__', "Reflection accessing private attributes"),
            (r'\.pyc[\'\"\\]', "Compiled Python file reference"),
        ]
        
        for pattern, warning in suspicious_patterns:
            if re.search(pattern, source, re.IGNORECASE):
                warnings.append(warning)
        
        return warnings
    
    def quick_check(self, code: str) -> bool:
        """
        Quick safety check - returns True if code appears safe.
        Used for fast-path approval.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                op = self._get_call_name(node)
                if op and self._is_dangerous(op):
                    return False
        
        return True
