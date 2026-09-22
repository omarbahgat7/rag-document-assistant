"""
frontend/app.py
==================
Enterprise-grade Streamlit chat interface for the RAG Document Assistant.

Run with:
    streamlit run app.py
"""

import html as html_lib

import streamlit as st
import streamlit.components.v1 as components

from api_client import (
    API_BASE_URL,
    BackendRequestError,
    BackendUnavailableError,
    ask_question,
    check_backend_health,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Assistant",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="expanded",
)

STARTER_PROMPTS = [
    "Summarize the main document",
    "What are the core findings?",
    "Explain the methodology",
    "List any dates or deadlines mentioned",
]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
# Design direction: obsidian glass — deep near-black surfaces, translucent
# panels with a soft blur, and one considered accent (violet) for anything
# interactive, plus a cooler cyan reserved specifically for evidence
# (citations), so color always tells the user what kind of thing they're
# looking at rather than decorating everything uniformly.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #09090b;
        --panel: #121214;
        --panel-glass: rgba(24, 24, 27, 0.68);
        --border: #27272a;
        --border-soft: rgba(255, 255, 255, 0.06);
        --text: #F4F4F5;
        --text-muted: #A1A1AA;
        --text-faint: #71717A;
        --accent: #8B5CF6;
        --accent-soft: rgba(139, 92, 246, 0.14);
        --accent-line: rgba(139, 92, 246, 0.35);
        --cyan: #22D3EE;
        --cyan-soft: rgba(34, 211, 238, 0.10);
        --cyan-line: rgba(34, 211, 238, 0.30);
        --green: #34D399;
        --red: #F87171;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(139, 92, 246, 0.08), transparent 40%),
            radial-gradient(circle at 85% 15%, rgba(34, 211, 238, 0.05), transparent 35%),
            var(--bg);
    }

    #MainMenu, footer, header[data-testid="stHeader"] { background: transparent; }

    /* ---- Top bar ---- */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.25rem 0 1.4rem 0;
        margin-bottom: 1.4rem;
        border-bottom: 1px solid var(--border);
    }
    .topbar-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .topbar-mark {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: linear-gradient(135deg, var(--accent), var(--cyan));
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: #0a0a0d;
        font-size: 1rem;
    }
    .topbar-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.28rem;
        letter-spacing: -0.01em;
        line-height: 1.1;
    }
    .topbar-subtitle {
        font-size: 0.8rem;
        color: var(--text-faint);
    }

    /* ---- Hero (empty state) ---- */
    .hero {
        text-align: center;
        padding: 2.6rem 1rem 1.6rem 1rem;
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 2.3rem;
        letter-spacing: -0.015em;
        margin: 0 0 0.6rem 0;
        background: linear-gradient(90deg, #F4F4F5, #C4B5FD 60%, #67E8F9);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: var(--text-muted);
        font-size: 1rem;
        max-width: 34rem;
        margin: 0 auto;
        line-height: 1.55;
    }

    /* ---- Glass panel base ---- */
    .glass {
        background: var(--panel-glass);
        border: 1px solid var(--border);
        border-radius: 14px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }

    /* ---- Chat messages ---- */
    [data-testid="stChatMessage"] {
        background: var(--panel-glass);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        animation: rise 0.28s ease-out;
    }
    @keyframes rise {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, var(--accent), #6D28D9) !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, var(--cyan), #0891B2) !important;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
        line-height: 1.6;
        font-size: 0.95rem;
    }
    [data-testid="stChatMessage"] code {
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        padding: 0.1rem 0.35rem;
    }

    /* ---- Starter chips ---- */
    div[data-testid="stButton"] button[kind="secondary"].chip-btn,
    .stButton button {
        border-radius: 10px;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.2rem;
    }
    .sidebar-brand .topbar-mark { width: 28px; height: 28px; font-size: 0.85rem; border-radius: 8px; }
    .sidebar-brand span {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.02rem;
    }

    /* ---- Status pill with glow ---- */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 0.82rem;
        color: var(--text-muted);
        padding: 0.45rem 0.85rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: rgba(255,255,255,0.03);
        width: 100%;
        box-sizing: border-box;
    }
    .status-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .status-dot.online {
        background: var(--green);
        box-shadow: 0 0 8px 1px rgba(52, 211, 153, 0.7);
        animation: glow-pulse 2.4s ease-in-out infinite;
    }
    .status-dot.offline {
        background: var(--red);
        box-shadow: 0 0 8px 1px rgba(248, 113, 113, 0.6);
    }
    @keyframes glow-pulse {
        0%, 100% { box-shadow: 0 0 6px 1px rgba(52, 211, 153, 0.55); }
        50%      { box-shadow: 0 0 12px 3px rgba(52, 211, 153, 0.85); }
    }

    /* ---- Metadata badge ---- */
    .meta-badge {
        font-size: 0.74rem;
        color: var(--text-faint);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.5rem 0.65rem;
        line-height: 1.6;
        background: rgba(255,255,255,0.02);
    }
    .meta-badge b { color: var(--text-muted); font-weight: 600; }

    /* ---- Citation cards ---- */
    .citation-card {
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        border: 1px solid var(--cyan-line);
        background: var(--cyan-soft);
        border-radius: 12px;
        padding: 0.75rem 0.95rem;
        margin-bottom: 0.6rem;
    }
    .citation-badge {
        flex-shrink: 0;
        width: 1.6rem;
        height: 1.6rem;
        border-radius: 7px;
        background: var(--cyan);
        color: #06282E;
        font-weight: 700;
        font-size: 0.78rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .citation-body { min-width: 0; flex: 1; }
    .citation-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .citation-label {
        font-weight: 600;
        font-size: 0.88rem;
        color: var(--text);
    }
    .citation-score {
        font-size: 0.7rem;
        font-weight: 600;
        color: #06282E;
        background: var(--cyan);
        border-radius: 999px;
        padding: 0.12rem 0.55rem;
        white-space: nowrap;
    }
    .citation-detail {
        font-size: 0.83rem;
        color: var(--text-muted);
        line-height: 1.5;
        margin-top: 0.3rem;
    }

    /* ---- Error banner ---- */
    .error-banner {
        border: 1px solid rgba(248, 113, 113, 0.35);
        background: rgba(248, 113, 113, 0.08);
        border-radius: 12px;
        padding: 0.9rem 1.05rem;
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .error-banner strong { color: var(--red); }

    /* ---- Buttons ---- */
    .stButton button {
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
        color: var(--text);
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton button:hover {
        border-color: var(--accent-line);
        color: var(--accent);
        background: var(--accent-soft);
    }

    /* ---- Chat input ---- */
    [data-testid="stChatInput"] {
        border-radius: 12px;
        border: 1px solid var(--border) !important;
    }

    /* ---- Slider accent ---- */
    [data-testid="stSlider"] [role="slider"] {
        background-color: var(--accent) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources", "is_error"}]

if "backend_online" not in st.session_state:
    st.session_state.backend_online = check_backend_health()

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="topbar-mark">◆</div>
            <span>Document Assistant</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Retrieval-grounded Q&A over your ingested documents.")

    st.divider()

    st.markdown("**System status**")
    status_label = "Backend online" if st.session_state.backend_online else "Backend unreachable"
    status_class = "online" if st.session_state.backend_online else "offline"
    st.markdown(
        f"""
        <div class="status-pill">
            <span class="status-dot {status_class}"></span>
            {status_label}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"`{API_BASE_URL}`")

    if st.button("Recheck connection", use_container_width=True):
        st.session_state.backend_online = check_backend_health()
        st.rerun()

    st.divider()

    st.markdown("**Retrieval settings**")
    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=10,
        value=4,
        help="How many document chunks the backend pulls before generating an answer.",
    )

    st.divider()

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    st.markdown(
        """
        <div class="meta-badge">
            <b>RAG Document Assistant</b><br/>
            Embeddings: all-MiniLM-L6-v2<br/>
            Vector store: ChromaDB (persistent)<br/>
            Phase: 4 — Frontend
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="topbar">
        <div class="topbar-brand">
            <div class="topbar-mark">◆</div>
            <div>
                <div class="topbar-title">Document Assistant</div>
                <div class="topbar-subtitle">Grounded answers, cited sources</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_copy_button(text: str, key: str):
    """Small floating-style copy-to-clipboard button rendered via a sandboxed
    component (script tags inside st.markdown are unreliable across Streamlit
    versions, so this uses components.html which executes JS properly)."""
    safe_text = html_lib.escape(text).replace("`", "&#96;")
    components.html(
        f"""
        <div style="font-family: Inter, sans-serif;">
        <button id="copy-{key}" style="
            background: rgba(255,255,255,0.05);
            border: 1px solid #27272a;
            color: #A1A1AA;
            border-radius: 8px;
            font-size: 12px;
            padding: 4px 10px;
            cursor: pointer;
        ">Copy</button>
        <pre id="src-{key}" style="display:none">{safe_text}</pre>
        <script>
            const btn = document.getElementById("copy-{key}");
            const src = document.getElementById("src-{key}");
            btn.addEventListener("click", function() {{
                navigator.clipboard.writeText(src.textContent);
                btn.textContent = "Copied";
                setTimeout(() => {{ btn.textContent = "Copy"; }}, 1400);
            }});
        </script>
        </div>
        """,
        height=34,
    )


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"View sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, start=1):
            detail = src.get("detail")
            detail_html = ""
            if detail:
                trimmed = detail[:280] + ("…" if len(detail) > 280 else "")
                detail_html = f'<div class="citation-detail">{html_lib.escape(trimmed)}</div>'

            score = src.get("score")
            score_html = ""
            if isinstance(score, (int, float)):
                score_html = f'<span class="citation-score">score {score:.3f}</span>'

            st.markdown(
                f"""
                <div class="citation-card">
                    <div class="citation-badge">{i}</div>
                    <div class="citation-body">
                        <div class="citation-top">
                            <span class="citation-label">{html_lib.escape(src["label"])}</span>
                            {score_html}
                        </div>
                        {detail_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_error_banner(text_html: str):
    st.markdown(f'<div class="error-banner">{text_html}</div>', unsafe_allow_html=True)


def process_question(question: str, top_k_value: int):
    """Shared path for both chat-input submissions and starter-chip clicks."""
    st.session_state.messages.append({"role": "user", "content": question, "sources": [], "is_error": False})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer…"):
            try:
                result = ask_question(question, top_k=top_k_value)
                st.session_state.backend_online = True

                st.markdown(result["answer"])
                render_copy_button(result["answer"], key=f"msg-{len(st.session_state.messages)}")
                render_sources(result["sources"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "is_error": False,
                })

            except BackendUnavailableError as e:
                st.session_state.backend_online = False
                error_text = f"<strong>Backend unreachable.</strong><br/>{html_lib.escape(str(e))}"
                render_error_banner(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text, "sources": [], "is_error": True}
                )

            except BackendRequestError as e:
                error_text = f"<strong>Backend returned an error.</strong><br/>{html_lib.escape(str(e))}"
                render_error_banner(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text, "sources": [], "is_error": True}
                )

            except Exception as e:  # last-resort safety net so the UI never hard-crashes
                error_text = f"<strong>Unexpected error.</strong><br/>{html_lib.escape(str(e))}"
                render_error_banner(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text, "sources": [], "is_error": True}
                )


# ---------------------------------------------------------------------------
# Empty state: hero + starter chips
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        """
        <div class="hero">
            <h1>What do you want to know?</h1>
            <p>Ask anything about your ingested documents — every answer is grounded
            in retrieved passages, with sources cited so you can verify the claim.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for idx, prompt in enumerate(STARTER_PROMPTS):
        with cols[idx % 2]:
            if st.button(prompt, use_container_width=True, key=f"chip-{idx}"):
                st.session_state.pending_question = prompt
                st.rerun()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("is_error"):
            render_error_banner(message["content"])
        else:
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_copy_button(message["content"], key=f"hist-{i}")
                render_sources(message.get("sources", []))

# ---------------------------------------------------------------------------
# Handle a starter-chip click queued from the empty state
# ---------------------------------------------------------------------------
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    process_question(q, top_k)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_question = st.chat_input("Ask a question about your documents…")
if user_question:
    process_question(user_question, top_k)
