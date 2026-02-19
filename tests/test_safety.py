"""
Tests for safety checker.
"""

import pytest
from pathlib import Path

from agent.safety.checker import SafetyChecker
from agent.models import CodeArtifact, SafetyLevel


class TestSafetyChecker:
    """Tests for safety checker."""
    
    @pytest.fixture
    def checker(self):
        return SafetyChecker()
    
    def test_safe_code(self, checker):
        code = CodeArtifact(
            source="""
def add(a, b):
    return a + b

result = add(2, 3)
print(result)
""",
            file_path="test.py",
            language="python"
        )
        
        report = checker.analyze(code)
        assert report.level == SafetyLevel.SAFE
        assert not report.requires_approval
    
    def test_dangerous_eval(self, checker):
        code = CodeArtifact(
            source="""
user_input = input("Enter code: ")
result = eval(user_input)
print(result)
""",
            file_path="test.py",
            language="python"
        )
        
        report = checker.analyze(code)
        assert report.level == SafetyLevel.DANGEROUS
        assert "eval" in str(report.dangerous_operations).lower()
    
    def test_dangerous_exec(self, checker):
        code = CodeArtifact(
            source="""
exec("import os; os.system('rm -rf /')")
""",
            file_path="test.py",
            language="python"
        )
        
        report = checker.analyze(code)
        assert report.level == SafetyLevel.DANGEROUS
    
    def test_network_imports(self, checker):
        code = CodeArtifact(
            source="""
import requests
import socket

requests.get("https://evil.com")
""",
            file_path="test.py",
            language="python"
        )
        
        report = checker.analyze(code)
        # Should have warnings about network modules
        assert len(report.warnings) > 0
    
    def test_filesystem_operations(self, checker):
        code = CodeArtifact(
            source="""
with open("/etc/passwd", "r") as f:
    data = f.read()
print(data)
""",
            file_path="test.py",
            language="python"
        )
        
        report = checker.analyze(code)
        # Should require approval for file operations
        assert report.requires_approval
    
    def test_obfuscated_code(self, checker):
        code = CodeArtifact(
            source="""
import base64
code = base64.b64decode("cHJpbnQoJ2hlbGxvJyk=")
eval(code)
""",
            file_path="test.py",
            language="python"
        )
        
        report = checker.analyze(code)
        # Should detect base64 decoding
        assert any("base64" in w.lower() for w in report.warnings)
    
    def test_syntax_error(self, checker):
        code = CodeArtifact(
            source="""
def broken(
    print("missing parenthesis"
""",
            file_path="test.py",
            language="python"
        )
        
        report = checker.analyze(code)
        # Should handle syntax errors gracefully
        assert report.level == SafetyLevel.WARNING
    
    def test_quick_check(self, checker):
        safe_code = "x = 1 + 2"
        dangerous_code = "eval(user_input)"
        
        assert checker.quick_check(safe_code) is True
        assert checker.quick_check(dangerous_code) is False
