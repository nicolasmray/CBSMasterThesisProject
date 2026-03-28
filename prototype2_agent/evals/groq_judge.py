"""Custom DeepEval judge model using Groq instead of OpenAI.

DeepEval metrics need an LLM judge. This wraps Groq so you can use your
existing GROQ_API_KEY — no OpenAI key needed.

Usage:
    from groq_judge import get_judge_model
    metric = FaithfulnessMetric(threshold=0.5, model=get_judge_model())
"""

import json
import os
import re
from typing import Optional

from deepeval.models.base_model import DeepEvalBaseLLM


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval-compatible wrapper around the Groq chat API."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "DEEPEVAL_GROQ_MODEL",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        )
        self._client = None

    def load_model(self, api_key: str | None = None):
        from groq import Groq
        key = api_key or os.getenv("GROQ_API_KEY")
        self._client = Groq(api_key=key)
        return self._client

    def generate(self, prompt: str, schema: Optional[object] = None) -> str:
        """Synchronous generation with auto key rotation on rate limit."""
        from llm_config import get_groq_key, rotate_groq_key

        last_error = None
        for _ in range(5):
            try:
                client = self.load_model(api_key=get_groq_key())
                return self._do_generate(client, prompt, schema)
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate_limit" in err or "rate limit" in err:
                    last_error = e
                    new_key = rotate_groq_key()
                    if new_key is None:
                        raise
                    print(f"  [Judge key rotation] Rate limited, switching key...")
                    continue
                raise
        raise last_error

    def _do_generate(self, client, prompt: str, schema: Optional[object] = None) -> str:
        """Actual generation logic."""

        messages = [{"role": "user", "content": prompt}]

        # If a Pydantic schema is requested, instruct the model to return JSON
        if schema is not None:
            messages[0]["content"] = (
                prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON that matches "
                "the requested schema. No markdown fences, no explanation."
            )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=4096,
        )
        text = response.choices[0].message.content.strip()

        # If schema provided, parse the JSON into the Pydantic model
        if schema is not None:
            try:
                # Strip markdown fences if present
                cleaned = text
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
                    cleaned = re.sub(r"\n?```$", "", cleaned)
                data = json.loads(cleaned)
                return schema(**data)
            except (json.JSONDecodeError, TypeError, Exception):
                # Fallback: try to extract JSON from the text
                try:
                    match = re.search(r"\{.*\}", text, re.DOTALL)
                    if match:
                        data = json.loads(match.group())
                        return schema(**data)
                except Exception:
                    pass
                return text

        return text

    async def a_generate(self, prompt: str, schema: Optional[object] = None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name


# Singleton
_judge: GroqJudge | None = None


def get_judge_model() -> GroqJudge:
    global _judge
    if _judge is None:
        _judge = GroqJudge()
    return _judge
