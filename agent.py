"""
Core Agent: model configuration + Python REPL tool + Agent creation.
"""
import os
import subprocess
import sys
import tempfile
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()


def get_llm(temperature: float = 0.6, max_tokens: int = 32768, seed: int = 42):
    """Initialize LLM based on environment configuration."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model_name = os.getenv("LLM_MODEL", "qwen2.5:14b")
    base_url = os.getenv("LLM_BASE_URL") or None

    if provider == "ollama":
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": seed,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)
    elif provider in ("openai", "vllm"):
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "timeout": 60,  # fail fast instead of hanging indefinitely
        }
        if base_url:
            kwargs["base_url"] = base_url
        api_key = os.getenv("OPENAI_API_KEY", "sk-no-key-required")
        kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


@tool
def python_calculator(code: str) -> str:
    """Execute Python code with math and sympy for precise calculation.

    Use this tool when you need to perform numerical calculations, symbolic
    mathematics, or any computation requiring precision. The code runs in a
    sandboxed subprocess with access to math, sympy, numpy, and itertools.

    Args:
        code: Python code string to execute. Print or assign the final result
            to a variable named `result` for clarity.
    """
    timeout = 30
    preamble = """import math
import sympy as sp
import numpy as np
import itertools
"""
    full_code = preamble + code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() if proc.stderr else "Unknown error"
            return f"Error: {err}"
        out = proc.stdout.strip() if proc.stdout else "(no output)"
        # Limit output length to avoid flooding context
        if len(out) > 4000:
            out = out[:4000] + "\n... (truncated)"
        return out
    except subprocess.TimeoutExpired:
        return f"Error: Execution timed out after {timeout} seconds"
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. Solve the given problem step by step.\n\n"
    "You have access to a python_calculator tool that executes Python code (including sympy) "
    "for precise calculations. Use it for any non-trivial numerical or symbolic computation.\n\n"
    "Important:\n"
    "1. Think step by step and explain your reasoning.\n"
    "2. Use the python_calculator tool for every calculation; do not guess.\n"
    "3. When you reach the final answer, output it in the format \\boxed{answer}.\n"
)


def create_math_agent(llm):
    """Create a ReAct agent with the Python calculator tool."""
    return create_agent(
        model=llm,
        tools=[python_calculator],
        system_prompt=SYSTEM_PROMPT,
    )
