"""
LLM Client implementations for the agent.
"""

import os
from typing import Optional
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """Generate completion from the LLM."""
        pass


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing.
    Returns predetermined responses based on prompt patterns.
    """
    
    def __init__(self, responses: dict = None):
        self.responses = responses or {}
        self.call_count = 0
    
    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        self.call_count += 1
        system_lower = (system or "").lower()

        # Check code generation BEFORE plan detection because system prompts
        # for code generation may mention the word "plan" in passing.
        if "Generate complete, runnable code" in prompt or "Please fix" in prompt:
            return self._code_response(prompt)

        if "Create a detailed plan" in prompt or "coding agent planner" in system_lower:
            return self._plan_response(prompt)

        if "Analyze this test failure" in prompt or "debugging" in system_lower:
            return self._reflection_response(prompt)

        # Default response
        return self.responses.get("default", '{"success": true, "analysis": "OK"}')
    
    def _plan_response(self, prompt: str) -> str:
        """Generate a plan response."""
        if "add" in prompt.lower() or "sum" in prompt.lower():
            return '''{
                "steps": [
                    "Define a function that takes two parameters",
                    "Return the sum of the two parameters",
                    "Include error handling for non-numeric inputs"
                ],
                "test_cases": [
                    "add(2, 3) should return 5",
                    "add(-1, 1) should return 0",
                    "add(0, 0) should return 0"
                ],
                "language": "python",
                "dependencies": [],
                "estimated_complexity": "low"
            }'''
        
        if "sort" in prompt.lower() or "order" in prompt.lower():
            return '''{
                "steps": [
                    "Define a function that accepts a list",
                    "Implement a sorting algorithm",
                    "Return the sorted list"
                ],
                "test_cases": [
                    "sort([3, 1, 2]) should return [1, 2, 3]",
                    "sort([]) should return []",
                    "sort([1]) should return [1]"
                ],
                "language": "python",
                "dependencies": [],
                "estimated_complexity": "medium"
            }'''
        
        return '''{
            "steps": [
                "Implement the requested functionality",
                "Add input validation",
                "Include test cases"
            ],
            "test_cases": [
                "Test basic functionality",
                "Test edge cases"
            ],
            "language": "python",
            "dependencies": [],
            "estimated_complexity": "medium"
        }'''
    
    def _code_response(self, prompt: str) -> str:
        """Generate code response."""
        if "add" in prompt.lower():
            return '''
def add(a, b):
    """Add two numbers."""
    return a + b

# Test cases
if __name__ == "__main__":
    assert add(2, 3) == 5, "Basic addition failed"
    assert add(-1, 1) == 0, "Negative number test failed"
    assert add(0, 0) == 0, "Zero test failed"
    print("All tests passed!")
'''
        
        if "sort" in prompt.lower():
            return '''
def sort_list(arr):
    """Sort a list using bubble sort."""
    if not isinstance(arr, list):
        raise TypeError("Input must be a list")
    
    result = arr.copy()
    n = len(result)
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    
    return result

# Test cases
if __name__ == "__main__":
    assert sort_list([3, 1, 2]) == [1, 2, 3], "Basic sort failed"
    assert sort_list([]) == [], "Empty list test failed"
    assert sort_list([1]) == [1], "Single element test failed"
    assert sort_list([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5], "Reverse sort failed"
    print("All tests passed!")
'''
        
        return '''
def solution():
    """Generic solution."""
    pass

if __name__ == "__main__":
    print("Hello, World!")
'''
    
    def _reflection_response(self, prompt: str) -> str:
        """Generate reflection response."""
        if "SyntaxError" in prompt:
            return '''{
                "success": false,
                "analysis": "There is a syntax error in the code",
                "root_cause": "Missing colon or incorrect indentation",
                "hypothesis": "The generated code has a Python syntax error",
                "suggested_fix": "Check for missing colons after function definitions and ensure proper indentation",
                "should_continue": true,
                "confidence": 0.8
            }'''
        
        if "AssertionError" in prompt:
            return '''{
                "success": false,
                "analysis": "Tests are failing with assertion errors",
                "root_cause": "Logic error in implementation",
                "hypothesis": "The algorithm is not correctly implemented",
                "suggested_fix": "Review the algorithm logic and fix the implementation",
                "should_continue": true,
                "confidence": 0.7
            }'''
        
        return '''{
            "success": false,
            "analysis": "Tests failed",
            "root_cause": "Unknown",
            "hypothesis": "There may be a logic error",
            "suggested_fix": "Review the code and try again",
            "should_continue": true,
            "confidence": 0.5
        }'''


class OpenAIClient(LLMClient):
    """
    OpenAI API client.
    Requires: pip install openai
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        max_tokens: int = 2000,
        base_url: Optional[str] = None
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI client requires: pip install openai")
        
        self.client = openai.AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
        self.model = model
        self.max_tokens = max_tokens
    
    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """
    Anthropic Claude API client.
    Requires: pip install anthropic
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 2000
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic client requires: pip install anthropic")
        
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.max_tokens = max_tokens
    
    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature
        }
        
        if system:
            kwargs["system"] = system
        
        response = await self.client.messages.create(**kwargs)
        
        return response.content[0].text


class KimiClient(LLMClient):
    """
    Kimi (Moonshot AI) API client.
    Kimi has an OpenAI-compatible API.
    
    Requires: pip install openai
    
    Get API key from: https://platform.moonshot.cn/
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "moonshot-v1-8k",
        max_tokens: int = 2000,
        base_url: str = "https://api.moonshot.cn/v1"
    ):
        """
        Initialize Kimi client.
        
        Args:
            api_key: Kimi API key (or set KIMI_API_KEY env var)
            model: Model to use. Options:
                - moonshot-v1-8k (8K context)
                - moonshot-v1-32k (32K context)  
                - moonshot-v1-128k (128K context)
            max_tokens: Maximum tokens in response
            base_url: Kimi API base URL
        """
        try:
            import openai
        except ImportError:
            raise ImportError("Kimi client requires: pip install openai")
        
        self.api_key = api_key or os.getenv("KIMI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Kimi API key required. Set KIMI_API_KEY environment variable "
                "or pass api_key parameter. Get key from https://platform.moonshot.cn/"
            )
        
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        self.model = model
        self.max_tokens = max_tokens
    
    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """
        Generate completion using Kimi API.
        
        Args:
            prompt: The user prompt
            system: Optional system message
            temperature: Sampling temperature (0-1)
        
        Returns:
            Generated text response
        """
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"Kimi API error: {e}")


class DeepSeekClient(LLMClient):
    """
    DeepSeek API client.
    DeepSeek has an OpenAI-compatible API.
    
    Requires: pip install openai
    
    Get API key from: https://platform.deepseek.com/
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        max_tokens: int = 2000,
        base_url: str = "https://api.deepseek.com"
    ):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            model: Model to use (deepseek-chat or deepseek-coder)
            max_tokens: Maximum tokens in response
            base_url: DeepSeek API base URL
        """
        try:
            import openai
        except ImportError:
            raise ImportError("DeepSeek client requires: pip install openai")
        
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter. Get key from https://platform.deepseek.com/"
            )
        
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        self.model = model
        self.max_tokens = max_tokens
    
    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """Generate completion using DeepSeek API."""
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"DeepSeek API error: {e}")


class OllamaClient(LLMClient):
    """
    Ollama local LLM client.
    
    Ollama runs open-source models locally with an OpenAI-compatible API.
    
    Setup:
        1. Install Ollama: https://ollama.com/
        2. Pull a model: ollama pull qwen2.5-coder:7b
        3. Start server: ollama serve
    
    Recommended models for coding:
        - qwen2.5-coder:7b (best balance)
        - qwen2.5-coder:14b (better quality)
        - codellama:7b
        - codellama:13b
        - deepseek-coder-v2:16b
        - llama3.1:8b
    
    Requires: pip install openai
    """
    
    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        timeout: int = 120,
        num_ctx: int = 8192
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (must be pulled first with 'ollama pull <model>')
            base_url: Ollama API base URL
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            num_ctx: Context window size (model dependent)
        """
        try:
            import openai
            import httpx
        except ImportError:
            raise ImportError("Ollama client requires: pip install openai httpx")
        
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_ctx = num_ctx
        
        # Create client with timeout
        http_client = httpx.AsyncClient(timeout=timeout)
        
        # Ollama uses OpenAI-compatible API at /v1 endpoint
        base_url_with_v1 = base_url.rstrip('/') + "/v1"
        
        self.client = openai.AsyncOpenAI(
            base_url=base_url_with_v1,
            api_key="ollama",  # Ollama doesn't need real API key
            http_client=http_client
        )
    
    async def complete(self, prompt: str, system: str = None, temperature: float = None) -> str:
        """
        Generate completion using local Ollama model.
        
        Args:
            prompt: The user prompt
            system: Optional system message
            temperature: Sampling temperature (uses default if not set)
        
        Returns:
            Generated text response
        """
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        temp = temperature if temperature is not None else self.temperature
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=self.max_tokens,
                extra_body={
                    "num_ctx": self.num_ctx  # Ollama-specific parameter
                }
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Make sure Ollama is running: ollama serve"
                )
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                raise RuntimeError(
                    f"Model '{self.model}' not found. "
                    f"Pull it first: ollama pull {self.model}"
                )
            else:
                raise RuntimeError(f"Ollama error: {e}")


class LlamaCppClient(LLMClient):
    """
    llama.cpp local LLM client.
    
    For running GGUF models directly with llama-cpp-python.
    Best for CPU-only inference.
    
    Setup:
        pip install llama-cpp-python
    
    Download models from TheBloke on HuggingFace:
        https://huggingface.co/TheBloke
    """
    
    def __init__(
        self,
        model_path: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        verbose: bool = False
    ):
        """
        Initialize llama.cpp client.
        
        Args:
            model_path: Path to .gguf model file
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
            verbose: Print loading info
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama.cpp client requires: pip install llama-cpp-python")
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Load model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose
        )
    
    async def complete(self, prompt: str, system: str = None, temperature: float = None) -> str:
        """Generate completion using local llama.cpp model."""
        
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Generate
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response["choices"][0]["message"]["content"]
