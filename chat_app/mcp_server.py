"""FastMCP server with 6 tools wrapping EvaluationEngine methods.

Runs in-process (no separate server) -- shares GPU state with the app.
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
def index_portfolio(path: str, force: bool = False) -> dict:
    """Index a portfolio PDF with ColPali for visual retrieval.

    Args:
        path: Path to the portfolio PDF file.
        force: Force re-indexing even if index exists.

    Returns:
        Status with page count.
    """
    return get_engine().index_portfolio(path, force)


@mcp.tool()
def index_checkliste(path: str) -> dict:
    """Extract evaluation criteria from a checkliste PDF.

    Args:
        path: Path to the checkliste PDF file.

    Returns:
        Extracted criteria list with count.
    """
    return get_engine().index_checkliste(path)


@mcp.tool()
def search_portfolio(query: str = "", top_k: int = 3) -> dict:
    """Search the indexed portfolio for relevant pages.

    If query is empty, searches for all extracted criteria.

    Args:
        query: Search query (leave empty to search all criteria).
        top_k: Number of pages to retrieve per query.

    Returns:
        Search results with page numbers and scores.
    """
    return get_engine().search_portfolio(query or None, top_k)


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
