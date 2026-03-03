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


HF_HOME = os.environ.get(
    "HF_HOME",
    "/lustre/project/ki-qarbs/nwang01/VisualRagPipeline/models",
)
INDEX_NAME = "portfolio_eval_index"


def find_local_colpali() -> str:
    """Find ColPali model -- local path or HuggingFace ID."""
    direct = os.path.join(HF_HOME, "hub/colpali-v1.3")
    if os.path.exists(direct):
        return direct
    cache = os.path.join(HF_HOME, "hub/models--vidore--colpali-v1.3/snapshots")
    if os.path.exists(cache):
        snaps = os.listdir(cache)
        if snaps:
            return os.path.join(cache, snaps[0])
    return "vidore/colpali-v1.3"


@dataclass
class AppConfig:
    """Application configuration.

    ColPali runs on cluster GPU for retrieval.
    Groq API handles all generation.
    """

    device: str = field(default_factory=detect_device)
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY")
    )
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    top_k: int = 3
    index_name: str = INDEX_NAME
    upload_dir: str = "uploads"
    results_dir: str = "results"

    def __post_init__(self):
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
