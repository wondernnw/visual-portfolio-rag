"""FastMCP server wrapping EvaluationEngine methods.

Runs in-process (no separate server) -- no GPU needed on Mac.
Bundle-based: load a pre-computed retrieval bundle, then evaluate via Groq.
"""

from typing import Optional

from fastmcp import FastMCP

from chat_app.config import AppConfig
from chat_app.engine import EvaluationEngine
from chat_app.vlm_backends import VLMBackend

# Module-level references set during create_server()
_engine: Optional[EvaluationEngine] = None

mcp = FastMCP("PortfolioEvaluation")


def create_server(config: AppConfig, vlm: VLMBackend) -> FastMCP:
    """Create and configure the MCP server with a shared engine instance."""
    global _engine
    _engine = EvaluationEngine(config, vlm)
    return mcp


def get_engine() -> EvaluationEngine:
    """Get the shared engine instance."""
    if _engine is None:
        raise RuntimeError("MCP server not initialized. Call create_server() first.")
    return _engine


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def load_bundle(path: str) -> dict:
    """Load a pre-computed retrieval bundle (from cluster export).

    Args:
        path: Path to the bundle JSON file.

    Returns:
        Status with portfolio name and criteria list.
    """
    return get_engine().load_bundle(path)


@mcp.tool()
def load_checkliste(path: str) -> dict:
    """Load checkliste PDF images for evaluation context.

    Args:
        path: Path to the checkliste PDF file.

    Returns:
        Status with page count.
    """
    return get_engine().load_checkliste(path)


@mcp.tool()
def evaluate_criterion(criterion_index: int) -> dict:
    """Evaluate the portfolio for a single criterion.

    Args:
        criterion_index: Zero-based index of the criterion to evaluate.

    Returns:
        Evaluation result with points, comment, page references, and quote.
    """
    return get_engine().evaluate_criterion(criterion_index)


@mcp.tool()
def evaluate_all() -> dict:
    """Evaluate the portfolio against all extracted criteria at once.

    Returns:
        All evaluation results with total score.
    """
    return get_engine().evaluate_all()


@mcp.tool()
def generate_report(output_path: str = "") -> dict:
    """Generate a PDF evaluation report.

    Args:
        output_path: Output PDF path (auto-generated if empty).

    Returns:
        Path to the generated PDF and total score.
    """
    return get_engine().generate_report(output_path or None)
