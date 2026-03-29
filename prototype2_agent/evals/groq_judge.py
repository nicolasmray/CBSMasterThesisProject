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

    def _call_groq(self, prompt: str, json_mode: bool = False) -> str:
        """Make a Groq API call with automatic key rotation on rate limit."""
        from llm_config import get_groq_key, rotate_groq_key

        last_error = None
        for _ in range(5):
            try:
                client = self.load_model(api_key=get_groq_key())
                messages = [{"role": "user", "content": prompt}]
                if json_mode:
                    messages[0]["content"] += (
                        "\n\nIMPORTANT: Respond with ONLY valid JSON. "
                        "No markdown fences, no explanation, no text before or after the JSON."
                    )
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=4096,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate_limit" in err or "rate limit" in err:
                    last_error = e
                    new_key = rotate_groq_key()
                    if new_key is None:
                        raise
                    continue
                raise
        raise last_error

    def _clean_json(self, text: str) -> str:
        """Strip markdown fences and extract JSON from LLM output."""
        cleaned = text.strip()
        # Remove ```json ... ``` fences
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?\s*```$", "", cleaned)
        return cleaned.strip()

    def generate(self, prompt: str, schema: Optional[object] = None) -> str:
        """Generate text. When schema is provided, return a parsed Pydantic instance."""
        if schema is not None:
            return self._generate_with_schema(prompt, schema)
        return self._call_groq(prompt, json_mode=False)

    def _generate_with_schema(self, prompt: str, schema) -> object:
        """Generate and parse into a Pydantic schema. Returns the schema instance."""
        text = self._call_groq(prompt, json_mode=True)
        cleaned = self._clean_json(text)

        # Try direct parse
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return schema(**data)
        except Exception:
            pass

        # Try extracting first JSON object from the text
        try:
            match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return schema(**data)
        except Exception:
            pass

        # Try extracting any JSON (including nested)
        try:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return schema(**data)
        except Exception:
            pass

        # Last resort: return raw text and let DeepEval's trimAndLoadJson handle it
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
