"""Streamlit Chat UI for Portfolio Evaluation Agent.

Mac-side app: loads a pre-computed retrieval bundle, evaluates via Groq API.
"""

import glob
import json
import os
import subprocess
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Bundle directory (scp target from cluster)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUNDLE_DIR = os.path.join(PROJECT_ROOT, "bundles")

# Cluster configuration
CLUSTER_HOST = "mogon-nhr"
CLUSTER_PROJECT = "/lustre/project/ki-qarbs/nwang01/PortfolioEvalTool"

from chat_app.chat_agent import ChatEvent, DeterministicAgent, EventType, GroqChatAgent
from chat_app.config import AppConfig
from chat_app.mcp_client import MCPBridge
from chat_app.mcp_server import create_server, get_engine


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session():
    """Initialize session state on first run."""
    if "initialized" in st.session_state:
        return

    st.session_state.initialized = True
    st.session_state.config = AppConfig()
    st.session_state.messages = []
    st.session_state.bridge = None
    st.session_state.agent = None
    st.session_state.bundle_path = None
    st.session_state.checkliste_path = None
    st.session_state.evaluations = None
    st.session_state.report_path = None


def init_backend():
    """Initialize or re-initialize the MCP backend."""
    config = st.session_state.config

    from chat_app.vlm_backends import GroqBackend

    vlm = GroqBackend(api_key=config.groq_api_key, model=config.groq_model)

    server = create_server(config, vlm)
    bridge = MCPBridge(server)
    st.session_state.bridge = bridge

    st.session_state.agent = GroqChatAgent(
        api_key=config.groq_api_key,
        bridge=bridge,
        model=config.groq_model,
    )


# ---------------------------------------------------------------------------
# File upload handler
# ---------------------------------------------------------------------------

def save_uploaded_file(uploaded_file, subdir: str = "") -> str:
    """Save an uploaded file and return its path."""
    config = st.session_state.config
    target_dir = os.path.join(config.upload_dir, subdir) if subdir else config.upload_dir
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def find_latest_bundle() -> str | None:
    """Find the latest bundle JSON in the bundles/ directory.

    Keeps only the latest bundle and removes older ones.
    """
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(BUNDLE_DIR, "bundle_*.json")))
    if not files:
        return None
    # Clean up old bundles, keep only the latest
    for old in files[:-1]:
        os.remove(old)
    return files[-1]


def upload_to_cluster(local_path: str) -> tuple[bool, str]:
    """Upload a portfolio PDF to the cluster via scp."""
    filename = os.path.basename(local_path)
    remote = f"{CLUSTER_HOST}:{CLUSTER_PROJECT}/uploads/portfolios/{filename}"
    result = subprocess.run(
        ["scp", local_path, remote],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode == 0:
        return True, filename
    return False, result.stderr.strip()


def submit_cluster_job(filename: str) -> tuple[bool, str]:
    """Submit the export_bundle SLURM job on the cluster."""
    cmd = f"cd {CLUSTER_PROJECT} && sbatch --cluster=mogonki submit_export.slurm uploads/portfolios/{filename}"
    result = subprocess.run(
        ["ssh", CLUSTER_HOST, cmd],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        return True, result.stdout.strip()
    return False, result.stderr.strip()


def fetch_bundle_from_cluster() -> tuple[bool, str]:
    """Fetch the latest bundle from the cluster to local bundles/ dir."""
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    # Clean old local bundles
    for old in glob.glob(os.path.join(BUNDLE_DIR, "bundle_*.json")):
        os.remove(old)
    remote = f"{CLUSTER_HOST}:{CLUSTER_PROJECT}/bundles/bundle_*.json"
    result = subprocess.run(
        ["scp", remote, BUNDLE_DIR + "/"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        latest = find_latest_bundle()
        if latest:
            return True, os.path.basename(latest)
        return False, "Bundle kopiert aber nicht gefunden."
    return False, result.stderr.strip()


# ---------------------------------------------------------------------------
# Chat display helpers
# ---------------------------------------------------------------------------

def display_message(role: str, content: str):
    """Display a chat message."""
    with st.chat_message(role):
        st.markdown(content)


def display_tool_event(event: ChatEvent):
    """Display a tool call or result event."""
    if event.type == EventType.TOOL_CALL:
        args_str = ", ".join(f"{k}={v!r}" for k, v in event.tool_args.items())
        st.caption(f"Werkzeug: `{event.tool_name}({args_str})`")

    elif event.type == EventType.TOOL_RESULT:
        result = event.tool_result
        if isinstance(result, dict):
            status = result.get("status", "unknown")
            if status == "ok":
                _display_tool_result_ok(event.tool_name, result)
            else:
                st.error(f"Fehler: {result.get('message', 'Unbekannter Fehler')}")
        else:
            st.json(result)

    elif event.type == EventType.ERROR:
        st.error(f"Fehler bei `{event.tool_name}`: {event.content}")


def _display_tool_result_ok(tool_name: str, result: dict):
    """Display successful tool results with appropriate formatting."""
    if tool_name == "load_bundle":
        name = result.get("portfolio_name", "?")
        count = result.get("criteria_count", 0)
        st.success(f"Bundle geladen: {name} ({count} Kriterien)")
        criteria = result.get("criteria", [])
        if criteria:
            for i, c in enumerate(criteria):
                st.caption(f"  {i+1}. {c['kriterium']} (max. {c['max_punkte']} Pkt.)")

    elif tool_name == "load_checkliste":
        pages = result.get("pages", 0)
        st.success(f"Checkliste geladen: {pages} Seiten")

    elif tool_name == "evaluate_criterion":
        ev = result.get("evaluation", {})
        _display_single_evaluation(ev)

    elif tool_name == "evaluate_all":
        total = result.get("total", 0)
        total_max = result.get("total_max", 0)
        st.metric("Gesamtergebnis", f"{total} / {total_max} Punkte")
        for ev in result.get("evaluations", []):
            _display_single_evaluation(ev)
        st.session_state.evaluations = result.get("evaluations")

    elif tool_name == "generate_report":
        path = result.get("path", "")
        st.success(f"PDF-Bericht erstellt: {os.path.basename(path)}")
        st.session_state.report_path = path


def _display_single_evaluation(ev: dict):
    """Display a single criterion evaluation."""
    if not ev:
        return
    punkte = ev.get("punkte", 0)
    max_p = ev.get("max_punkte", 0)
    kriterium = ev.get("kriterium", "")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{kriterium}**")
    with col2:
        st.metric("Punkte", f"{punkte}/{max_p}", delta=None)

    if ev.get("kommentar"):
        st.warning(f"Kommentar: {ev['kommentar']}")

    refs = ev.get("seiten_referenzen", [])
    zitat = ev.get("zitat", "")
    if refs:
        st.caption(f"Seiten: {', '.join(str(r) for r in refs)}")
    if zitat:
        st.info(f'Zitat: "{zitat}"')


def process_agent_events(events):
    """Process and display a stream of agent events."""
    for event in events:
        if event.type == EventType.TEXT:
            st.session_state.messages.append(
                {"role": "assistant", "content": event.content}
            )
            display_message("assistant", event.content)
        elif event.type in (EventType.TOOL_CALL, EventType.TOOL_RESULT, EventType.ERROR):
            display_tool_event(event)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with configuration and file upload."""
    with st.sidebar:
        st.header("Einstellungen")

        config = st.session_state.config

        # Groq API key
        api_key = st.text_input(
            "Groq API Key",
            value=config.groq_api_key or "",
            type="password",
            help="Beginnt mit gsk_...",
        )
        if api_key != config.groq_api_key:
            config.groq_api_key = api_key
            st.session_state.bridge = None

        st.divider()

        # Portfolio upload & cluster export
        st.subheader("1. Portfolio hochladen")

        portfolio_file = st.file_uploader(
            "Portfolio PDF",
            type=["pdf"],
            key="portfolio_upload",
            help="Portfolio-PDF wird zum Cluster gesendet",
        )
        if portfolio_file:
            # Save locally first
            local_path = save_uploaded_file(portfolio_file, "portfolios")
            st.session_state.portfolio_local_path = local_path
            st.success(f"Portfolio: {portfolio_file.name}")

        if st.button(
            "An Cluster senden & Export starten",
            disabled=not st.session_state.get("portfolio_local_path"),
            use_container_width=True,
        ):
            local_path = st.session_state.portfolio_local_path
            with st.spinner("Sende an Cluster..."):
                ok, msg = upload_to_cluster(local_path)
            if ok:
                st.success(f"Hochgeladen: {msg}")
                with st.spinner("SLURM-Job wird gestartet..."):
                    ok2, msg2 = submit_cluster_job(msg)
                if ok2:
                    st.success(msg2)
                else:
                    st.error(f"Job-Fehler: {msg2}")
            else:
                st.error(f"Upload-Fehler: {msg}")

        st.divider()

        # Bundle fetch
        st.subheader("2. Bundle laden")

        if st.button("Bundle vom Cluster holen", use_container_width=True):
            with st.spinner("Lade Bundle vom Cluster..."):
                ok, msg = fetch_bundle_from_cluster()
            if ok:
                st.success(f"Bundle: {msg}")
                st.rerun()
            else:
                st.error(f"Fehler: {msg}")

        # Auto-detect bundle from bundles/ directory
        latest = find_latest_bundle()
        if latest:
            st.session_state.bundle_path = latest
            st.success(f"Bundle: {os.path.basename(latest)}")
        else:
            st.info("Kein Bundle vorhanden.")

        st.divider()

        # Checkliste upload (optional)
        st.subheader("Checkliste (optional)")
        checkliste_file = st.file_uploader(
            "Checkliste PDF",
            type=["pdf"],
            key="checkliste_upload",
            help="Fuer Checklisten-Bilder in der Bewertung",
        )
        if checkliste_file:
            path = save_uploaded_file(checkliste_file, "checklisten")
            st.session_state.checkliste_path = path
            st.success(f"Checkliste: {checkliste_file.name}")

        st.divider()

        # Action buttons
        st.subheader("Aktionen")

        bundle_ready = bool(st.session_state.bundle_path)

        if st.button(
            "Bewertung starten",
            disabled=not bundle_ready,
            use_container_width=True,
        ):
            st.session_state.run_full_pipeline = True

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Checkliste laden", disabled=not st.session_state.checkliste_path):
                st.session_state.run_step = ("load_checkliste", {"path": st.session_state.checkliste_path})
        with col2:
            if st.button("PDF-Bericht", disabled=not st.session_state.evaluations):
                st.session_state.run_step = ("generate_report", {"output_path": ""})

        # Download report
        if st.session_state.get("report_path") and os.path.exists(st.session_state.report_path):
            with open(st.session_state.report_path, "rb") as f:
                st.download_button(
                    "PDF herunterladen",
                    data=f,
                    file_name=os.path.basename(st.session_state.report_path),
                    mime="application/pdf",
                    use_container_width=True,
                )

        # Status
        st.divider()
        st.caption(f"Device: {config.device}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Portfolio-Bewertung",
        page_icon="📋",
        layout="wide",
    )

    init_session()
    render_sidebar()

    # Lazy init backend
    if st.session_state.bridge is None:
        config = st.session_state.config
        if not config.groq_api_key:
            st.info("Bitte Groq API Key in der Seitenleiste eingeben.")
            return
        with st.spinner("Backend wird initialisiert..."):
            init_backend()

    st.title("Portfolio-Bewertung")
    st.caption("Bewertung von Studierenden-Portfolios mit Visual RAG")

    # Display chat history
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])

    # Display last evaluation results (persists after rerun)
    if st.session_state.get("evaluations") and not st.session_state.get("run_full_pipeline"):
        with st.chat_message("assistant"):
            evals = st.session_state.evaluations
            total = sum(e["punkte"] for e in evals)
            total_max = sum(e["max_punkte"] for e in evals)
            st.metric("Gesamtergebnis", f"{total} / {total_max} Punkte")
            for ev in evals:
                _display_single_evaluation(ev)
            if st.session_state.get("report_path"):
                st.success(f"PDF-Bericht erstellt: {os.path.basename(st.session_state.report_path)}")

    # Handle full pipeline run (from button)
    if st.session_state.get("run_full_pipeline"):
        st.session_state.run_full_pipeline = False
        _run_full_pipeline()

    # Handle single step run (from button)
    if st.session_state.get("run_step"):
        tool_name, args = st.session_state.run_step
        st.session_state.run_step = None
        _run_single_step(tool_name, args)

    # Chat input
    if user_input := st.chat_input("Nachricht eingeben..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_message("user", user_input)
        _handle_chat_input(user_input)


def _handle_chat_input(user_input: str):
    """Handle a user chat message via GroqChatAgent."""
    agent = st.session_state.agent

    with st.chat_message("assistant"):
        with st.status("Verarbeite...", expanded=True):
            events = list(agent.chat(user_input))
        for event in events:
            if event.type == EventType.TEXT:
                st.markdown(event.content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": event.content}
                )
            elif event.type in (EventType.TOOL_CALL, EventType.TOOL_RESULT, EventType.ERROR):
                display_tool_event(event)


def _run_full_pipeline():
    """Run the full evaluation pipeline: load bundle, (optionally) checkliste, evaluate, report."""
    bundle_path = st.session_state.bundle_path
    checkliste_path = st.session_state.checkliste_path

    if not bundle_path:
        st.error("Bundle muss hochgeladen sein.")
        return

    pipeline = DeterministicAgent(bridge=st.session_state.bridge)

    with st.chat_message("assistant"):
        with st.status("Bewertung laeuft...", expanded=True):
            for event in pipeline.run_full_pipeline(bundle_path, checkliste_path):
                display_tool_event(event)
                if (event.type == EventType.TOOL_RESULT
                        and event.tool_name == "evaluate_all"
                        and isinstance(event.tool_result, dict)):
                    st.session_state.evaluations = event.tool_result.get("evaluations")
                if (event.type == EventType.TOOL_RESULT
                        and event.tool_name == "generate_report"
                        and isinstance(event.tool_result, dict)):
                    st.session_state.report_path = event.tool_result.get("path")
        st.success("Bewertung abgeschlossen!")
        st.session_state.messages.append(
            {"role": "assistant", "content": "Bewertung abgeschlossen!"}
        )
    st.rerun()


def _run_single_step(tool_name: str, args: dict):
    """Run a single tool step via deterministic execution."""
    pipeline = DeterministicAgent(bridge=st.session_state.bridge)

    with st.chat_message("assistant"):
        with st.status(f"Fuehre {tool_name} aus...", expanded=True):
            for event in pipeline.run_step(tool_name, args):
                display_tool_event(event)
                if (event.type == EventType.TOOL_RESULT
                        and event.tool_name == "evaluate_all"
                        and isinstance(event.tool_result, dict)):
                    st.session_state.evaluations = event.tool_result.get("evaluations")
                if (event.type == EventType.TOOL_RESULT
                        and event.tool_name == "generate_report"
                        and isinstance(event.tool_result, dict)):
                    st.session_state.report_path = event.tool_result.get("path")


if __name__ == "__main__":
    main()
