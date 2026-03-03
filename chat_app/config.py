"""Application configuration and device detection."""

import os
from dataclasses import dataclass, field
from typing import Optional


def detect_device() -> str:
    """Detect best available compute device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class AppConfig:
    """Application configuration.

    Mac-side: no GPU needed. Groq API handles all generation.
    Retrieval results come from pre-computed bundles.
    """

    device: str = field(default_factory=detect_device)
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY")
    )
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    top_k: int = 3
    upload_dir: str = "uploads"
    results_dir: str = "results"

    def __post_init__(self):
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
