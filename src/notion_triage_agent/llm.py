"""LLM client wrapper.

The only file that knows which model vendor we use. Nodes depend on the
LLMClient protocol below, not on any SDK -- swapping providers means adding
a class here and changing one line in cli.py.
"""

from typing import Protocol, TypeVar

from google import genai
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-3.6-flash"


class LLMError(Exception):
    """Raised when the model call fails or returns unusable output."""


class LLMClient(Protocol):
    """What the nodes need from a language model. Nothing else."""

    def generate(self, prompt: str, schema: type[T]) -> T:
        """Return an instance of `schema` produced from `prompt`."""
        ...


class GeminiClient:
    """LLMClient backed by the Google Gemini API."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, schema: type[T]) -> T:
        """Call Gemini in JSON mode with `schema` as the response schema.

        Raises LLMError if the call fails or the response cannot be parsed
        into `schema`.
        """
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
        except Exception as exc:  # network, auth, quota, safety blocks
            raise LLMError(f"{self._model} call failed: {exc}") from exc

        parsed = response.parsed
        if not isinstance(parsed, schema):
            raise LLMError(
                f"{self._model} returned output that did not match "
                f"{schema.__name__}: {response.text!r}"
            )
        return parsed
