"""VLM backend abstraction — Groq API for generation."""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class VLMBackend(ABC):
    """Abstract base class for vision-language model backends."""

    @abstractmethod
    def generate(self, content: List[Dict], max_tok: int = 1024) -> str:
        """Send multimodal content and return generated text.

        Args:
            content: List of dicts with type "image" or "text".
                     Images use {"type": "image", "image": "data:image/jpeg;base64,..."}.
                     Text uses {"type": "text", "text": "..."}.
            max_tok: Maximum tokens to generate.

        Returns:
            Generated text string.
        """

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the backend is ready to generate."""


class GroqBackend(VLMBackend):
    """Groq API backend (Llama 4 Scout or similar vision model)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    ):
        from groq import Groq

        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._model = model
        self._client = Groq(api_key=self._api_key) if self._api_key else None

    def set_api_key(self, key: str):
        """Update API key at runtime (e.g., from Streamlit sidebar)."""
        from groq import Groq

        self._api_key = key
        self._client = Groq(api_key=key)

    def generate(self, content: List[Dict], max_tok: int = 1024) -> str:
        if not self._client:
            raise RuntimeError("Groq API key not set.")

        # Convert content to Groq message format
        groq_content = []
        for item in content:
            if item["type"] == "text":
                groq_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                groq_content.append({
                    "type": "image_url",
                    "image_url": {"url": item["image"]},
                })

        response = self._client.chat.completions.create(
            messages=[{"role": "user", "content": groq_content}],
            model=self._model,
            temperature=0.1,
            max_tokens=max_tok,
        )
        return response.choices[0].message.content.strip()

    def is_loaded(self) -> bool:
        return self._client is not None
