"""
LLM client for the pipeline
"""

import time, os
from typing import Dict, Any, Optional
from openai import OpenAI
from braintrust import init_logger, traced, current_span
from pipeline_core.utils import timestamp


# Try to load from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()

except ImportError:
    pass  # dotenv not installed, that's fine

# ---------------------------------------
# Global client (OpenAI by default)
# ---------------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # used with openrouter throug openai SDK
    # base_url="http://localhost:4000/",  # Your LiteLLM Proxy URL (Runnin in a local docker continer)
    # api_key=os.getenv("LITE_LLM_VKEY_MINI_PRO_02") or "your_openrouter_api_key_here",
    api_key=os.getenv("OPENROUTER_API_KEY") or "your_openrouter_api_key_here",
)

# ---------------------------------------
# Default model + params
# ---------------------------------------
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_PARAMS = {
    "temperature": 0.2,
    "max_tokens": 512,
}

# ---------------------------------------
# Retry settings
# ---------------------------------------
MAX_RETRIES = 1
RETRY_DELAY = 2  # seconds


# ---------------------------------------
# Initialize Braintrust logger
# ---------------------------------------
logger = init_logger(
    project="mini-project-01",
    api_key=os.getenv("BRAINTRUST_API_KEY") or "NOT SET",
)


# ---------------------------------------
# Core chat function
# ---------------------------------------
@traced(type="llm", name="generate_varied_response", notrace_io=True)
def chat(
    prompt: str,
    params: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Unified LLM call for all pipeline phases.
    Handles:
    - model selection
    - retries
    - error handling
    - consistent API usage
    - Braintrust logging
    """
    merged_params = {**DEFAULT_PARAMS, **(params or {})}
    model = model or DEFAULT_MODEL

    if logger is None:
        raise ValueError("Braintrust logger not initialized")

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                # **merged_params,
            )

            content = response.choices[0].message.content

            # -----------------------------
            # Braintrust logging
            # -----------------------------
            current_span().log(
                input=messages,
                output=content,
                metrics={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost": response.usage.total_tokens
                    * 0.000001
                    * 0.002,  # 0.002 is the cost per 1000 tokens for gpt-4o-mini,
                },
                metadata={
                    "model": model,
                    "params": merged_params,
                    "phase": metadata.get("phase") if metadata else None,
                    "phase_args": metadata.get("phase_args") if metadata else None,
                    # add more metadata here if needed
                    "timestamp": timestamp(),
                    "attempt": attempt,
                },
                tags=metadata.get("phase_tags") if metadata else None,
            )

            return content

        except Exception as e:
            print(f"[LLM] Error on attempt {attempt}: {e}")

            if attempt == MAX_RETRIES:
                raise RuntimeError(f"LLM failed after {MAX_RETRIES} attempts") from e

            time.sleep(RETRY_DELAY)
