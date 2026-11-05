"""PrivacyLens — Understand before you agree.

Streamlit front end. All policy-derived and model-generated text is HTML-escaped
before rendering, because both come from untrusted sources.
"""

from __future__ import annotations

import html
import logging

import streamlit as st

from config import SAMPLE_POLICY_PATH, get_settings
from core.analyzer import ProcessedPolicy, analyze_category, analyze_policy, process_policy
from core.embeddings import EmbeddingError, load_embedder
from core.extractor import ExtractionError, extract_from_url
from core.llm import LLMClient, LLMError, LLMUnavailableError
from core.preprocessor import Chunk, PolicyValidationError
from core.qa import answer_question
from models.schemas import CategorySummary, EvidenceStatus, PolicyOverview, QAAnswer

logging.basicConfig(level=logging.INFO)
settings = get_settings()

st.set_page_config(page_title="PrivacyLens", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

SUGGESTED_QUESTIONS = [
    "Does this company share my information with advertisers?",
    "Can I delete my personal data?",
    "Does this service collect my location?",
    "How long does the company keep my information?",
    "Can I opt out of personalized advertising?",
    "What information does the company collect about me?",
]
CATEGORY_ICONS = {
    "Data Collection": "📥",
    "Data Usage": "⚙️",
    "Data Sharing": "🔗",
    "Data Retention": "🗄️",
    "User Rights and Choices": "🧭",
}
CATEGORY_QUESTIONS = {
    "Data Collection": "What personal information does the service say it collects?",
    "Data Usage": "Why does the service say it uses information?",
    "Data Sharing": "Who might receive the information?",
    "Data Retention": "How long is information kept?",
    "User Rights and Choices": "What can users access, change, delete or opt out of?",
}

SHIELD_SVG = """<svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<path d="M20 3.5 6.5 8.6v10.2c0 8.6 5.8 15.4 13.5 17.7 7.7-2.3 13.5-9.1 13.5-17.7V8.6L20 3.5Z" fill="#E3F0F0" stroke="#1F6F78" stroke-width="2" stroke-linejoin="round"/>
<circle cx="18.6" cy="19" r="5.4" stroke="#1F6F78" stroke-width="2.2"/>
<path d="m22.6 23 4.4 4.4" stroke="#1F6F78" stroke-width="2.4" stroke-linecap="round"/>
</svg>"""

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --pl-bg: #F7F6F2; --pl-surface: #FFFFFF; --pl-ink: #1C2A39; --pl-muted: #5B6676;
  --pl-line: #E2E1DA; --pl-accent: #1F6F78; --pl-accent-soft: #E3F0F0;
  --pl-clear: #1D6B4F; --pl-clear-bg: #E3F2EA; --pl-unclear: #8A5A00; --pl-unclear-bg: #FBF0D9;
  --pl-missing: #4A5565; --pl-missing-bg: #ECEDEF;
}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: var(--pl-bg); color: var(--pl-ink); }
.block-container { max-width: 1180px; padding-top: 3.6rem; padding-bottom: 4rem; }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }
h1, h2, h3, h4 { color: var(--pl-ink); letter-spacing: -0.01em; }

.pl-header { display: flex; align-items: center; gap: 14px; padding: 6px 0 18px; border-bottom: 1px solid var(--pl-line); margin-bottom: 22px; }
.pl-brand { font-size: 1.55rem; font-weight: 700; color: var(--pl-ink); line-height: 1.1; }
.pl-tagline { color: var(--pl-muted); font-size: 0.98rem; margin-top: 2px; }

.pl-hero h2 { font-size: 1.9rem; margin: 0 0 8px; }
.pl-hero p { color: var(--pl-muted); font-size: 1.04rem; max-width: 640px; line-height: 1.6; }
.pl-steps { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin: 18px 0 6px; }
.pl-step { background: var(--pl-surface); border: 1px solid var(--pl-line); border-radius: 14px; padding: 14px 16px; }
.pl-step b { display: block; color: var(--pl-ink); margin-bottom: 4px; }
.pl-step span { color: var(--pl-muted); font-size: 0.92rem; }
@media (max-width: 760px) { .pl-steps { grid-template-columns: 1fr; } }

[data-testid="stVerticalBlockBorderWrapper"] { background: var(--pl-surface); border-radius: 16px !important; border-color: var(--pl-line) !important; box-shadow: 0 1px 2px rgba(28,42,57,0.04); }

.pl-pill { display: inline-flex; align-items: center; gap: 6px; font-size: 0.78rem; font-weight: 600; padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
.pl-pill::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.pl-pill.clearly_stated { color: var(--pl-clear); background: var(--pl-clear-bg); }
.pl-pill.unclear { color: var(--pl-unclear); background: var(--pl-unclear-bg); }
.pl-pill.not_found { color: var(--pl-missing); background: var(--pl-missing-bg); }

.pl-card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; margin-bottom: 6px; }
.pl-card-title { font-weight: 650; font-size: 1.08rem; color: var(--pl-ink); }
.pl-card-q { color: var(--pl-muted); font-size: 0.86rem; margin-bottom: 10px; }
.pl-summary { color: var(--pl-ink); line-height: 1.55; margin-bottom: 8px; }
.pl-finding { border-top: 1px solid var(--pl-line); padding-top: 10px; margin-top: 10px; }
.pl-finding-text { font-weight: 560; color: var(--pl-ink); line-height: 1.5; }
.pl-plain { color: var(--pl-muted); line-height: 1.5; margin-top: 3px; font-size: 0.95rem; }
.pl-note { color: var(--pl-muted); font-size: 0.86rem; background: #F3F2EC; border-radius: 10px; padding: 8px 10px; margin-top: 8px; line-height: 1.45; }
.pl-note.warn { background: var(--pl-unclear-bg); color: #6B4700; }

.pl-meta { color: var(--pl-muted); font-size: 0.9rem; }
.pl-meta a { color: var(--pl-accent); }
.pl-policy-title { font-size: 1.35rem; font-weight: 700; color: var(--pl-ink); margin-bottom: 2px; }
.pl-demo-badge { display: inline-block; font-size: 0.75rem; font-weight: 600; color: #6B4700; background: var(--pl-unclear-bg); padding: 2px 9px; border-radius: 999px; margin-left: 8px; vertical-align: middle; }

.pl-excerpt { background: #FBFAF7; border: 1px solid var(--pl-line); border-left: 3px solid var(--pl-accent); border-radius: 10px; padding: 12px 14px; color: var(--pl-ink); line-height: 1.6; font-size: 0.94rem; white-space: pre-wrap; max-height: 420px; overflow-y: auto; }
.pl-source-head { font-weight: 650; color: var(--pl-ink); }
.pl-source-id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; color: var(--pl-muted); }
.pl-answer { line-height: 1.6; color: var(--pl-ink); }

.pl-legend { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; color: var(--pl-muted); font-size: 0.86rem; margin: 4px 0 16px; }
.pl-disclaimer { color: var(--pl-muted); font-size: 0.82rem; border-top: 1px solid var(--pl-line); margin-top: 36px; padding-top: 14px; }

.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 550; border-color: var(--pl-line); }
.stButton > button[kind="primary"] { background: var(--pl-accent); border-color: var(--pl-accent); }
.stButton > button[kind="primary"]:hover { background: #185960; border-color: #185960; }
[data-testid="stPopover"] button { border-radius: 999px; font-size: 0.8rem; padding: 2px 12px; min-height: 0; border-color: #C9DCDD; color: var(--pl-accent); background: var(--pl-accent-soft); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.stTextInput input, .stTextArea textarea { border-radius: 10px; background: var(--pl-surface); }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def esc(text: str | None) -> str:
    return html.escape(text or "").replace("\n", "<br>")


def pill(status: EvidenceStatus) -> str:
    return f'<span class="pl-pill {status.value}">{status.label}</span>'


@st.cache_resource(show_spinner=False)
def get_embedder():
    return load_embedder(settings.embedding_model)


def get_llm() -> tuple[LLMClient | None, str | None]:
    try:
        return LLMClient.from_settings(settings), None
    except LLMUnavailableError as exc:
        return None, str(exc)


def init_state() -> None:
    defaults = {"policy": None, "overview": None, "chat": [], "pending_question": None, "flash_error": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_policy_state() -> None:
    """Clear everything tied to a previous policy (chat history and retrieval context)."""
    st.session_state.policy = None
    st.session_state.overview = None
    st.session_state.chat = []
    st.session_state.pending_question = None


def run_pipeline(source: str, value: str) -> None:
    reset_policy_state()
    llm, llm_reason = get_llm()
    with st.status("Analyzing policy…", expanded=True) as status:
        try:
            title, url, is_demo = None, None, False
            if source == "url":
                status.write("Fetching the page securely…")
                extracted = extract_from_url(value, settings=settings)
                text, title, url = extracted.text, extracted.title, extracted.source_url
            elif source == "demo":
                status.write("Loading the fictional demo policy…")
                text, is_demo = SAMPLE_POLICY_PATH.read_text(encoding="utf-8"), True
            else:
                text = value
            status.write("Cleaning text and splitting it into sections…")
            embedder = get_embedder()
            status.write("Embedding sections and building the search index…")
            policy = process_policy(text, embedder, title=title, source_url=url, settings=settings, is_demo=is_demo)
            st.session_state.policy = policy
            status.write(f"Indexed {len(policy.chunks)} sections ({policy.word_count:,} words).")
            if llm is None:
                status.update(label="Policy processed — AI features need an API key", state="complete", expanded=False)
                return
            status.write("Reading the policy for each privacy category…")
            st.session_state.overview = analyze_policy(policy, llm, settings)
            status.update(label="Analysis complete", state="complete", expanded=False)
        except (ExtractionError, PolicyValidationError) as exc:
            st.session_state.flash_error = str(exc)
            status.update(label="Couldn't analyze that policy", state="error", expanded=False)
        except EmbeddingError as exc:
            st.session_state.flash_error = str(exc)
            status.update(label="Search index unavailable", state="error", expanded=False)
        except Exception:  # noqa: BLE001 - keep the app alive on unexpected failures
            logging.exception("Unexpected pipeline failure")
            st.session_state.flash_error = "Something unexpected went wrong while processing the policy. Please try again."
            status.update(label="Unexpected error", state="error", expanded=False)


# ---------------------------------------------------------------------------
# Evidence rendering
# ---------------------------------------------------------------------------

def render_source(chunk: Chunk, policy: ProcessedPolicy) -> None:
    heading = esc(chunk.heading) if chunk.heading else "<i>No section heading</i>"
    st.markdown(
        f'<div class="pl-source-head">Source {chunk.source_number} — {heading}</div>'
        f'<div class="pl-source-id">ID: {esc(chunk.id)} · ~{chunk.approx_tokens} tokens</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="pl-excerpt">{html.escape(chunk.text)}</div>', unsafe_allow_html=True)
    if policy.source_url:
        st.markdown(
            f'<div class="pl-meta" style="margin-top:6px">From <a href="{html.escape(policy.source_url, quote=True)}" '
            f'target="_blank" rel="noopener noreferrer">{esc(policy.source_url)}</a></div>',
            unsafe_allow_html=True,
        )


def render_citations(source_ids: list[str], policy: ProcessedPolicy) -> None:
    chunks = [policy.index.get(sid) for sid in source_ids]
    chunks = [c for c in chunks if c is not None]
    if not chunks:
        return
    with st.container(horizontal=True, gap="small"):
        for chunk in chunks[:4]:
            with st.popover(f"Source {chunk.source_number}"):
                render_source(chunk, policy)


def render_category(cat: CategorySummary, policy: ProcessedPolicy) -> None:
    with st.container(border=True):
        st.markdown(
            f'<div class="pl-card-head"><div class="pl-card-title">{CATEGORY_ICONS.get(cat.category.value, "")} '
            f"{esc(cat.category.value)}</div>{pill(cat.evidence_status)}</div>"
            f'<div class="pl-card-q">{esc(CATEGORY_QUESTIONS.get(cat.category.value))}</div>'
            f'<div class="pl-summary">{esc(cat.summary)}</div>',
            unsafe_allow_html=True,
        )
        if cat.error:
            st.markdown(f'<div class="pl-note warn">{esc(cat.error)}</div>', unsafe_allow_html=True)
        for f in cat.findings:
            st.markdown(
                f'<div class="pl-finding"><div style="margin-bottom:4px">{pill(f.evidence_status)}</div>'
                f'<div class="pl-finding-text">{esc(f.finding)}</div>'
                f'<div class="pl-plain">{esc(f.plain_language_explanation)}</div></div>',
                unsafe_allow_html=True,
            )
            if f.verification_note:
                st.markdown(f'<div class="pl-note warn">{esc(f.verification_note)}</div>', unsafe_allow_html=True)
            render_citations(f.source_ids, policy)
        if not cat.findings and not cat.error:
            st.markdown(
                '<div class="pl-note">No supporting statement was identified in the processed policy for this category. '
                "This does not mean the practice doesn't happen — only that the policy text we processed doesn't describe it.</div>",
                unsafe_allow_html=True,
            )
        if cat.limitations:
            st.markdown(f'<div class="pl-note"><b>What the policy doesn\'t say:</b> {esc(cat.limitations)}</div>',
                        unsafe_allow_html=True)


def render_answer(ans: QAAnswer, policy: ProcessedPolicy, key: str) -> None:
    st.markdown(f"{pill(ans.evidence_status)}<div class='pl-answer' style='margin-top:6px'>{esc(ans.answer)}</div>",
                unsafe_allow_html=True)
    if ans.limitations:
        st.markdown(f'<div class="pl-note"><b>Limitations:</b> {esc(ans.limitations)}</div>', unsafe_allow_html=True)
    if ans.verification_note:
        st.markdown(f'<div class="pl-note warn">{esc(ans.verification_note)}</div>', unsafe_allow_html=True)
    if ans.source_ids:
        st.caption("Sources — open one to compare with the original wording")
        render_citations(ans.source_ids, policy)
    with st.expander("Copy answer"):
        sources = ", ".join(f"Source {policy.index.get(s).source_number}" for s in ans.source_ids if policy.index.get(s))
        text = ans.answer + (f"\n\nLimitations: {ans.limitations}" if ans.limitations else "")
        text += f"\n\nEvidence: {ans.evidence_status.label}" + (f" ({sources})" if sources else "")
        st.code(text, language=None, wrap_lines=True)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        f'<div class="pl-header">{SHIELD_SVG}<div><div class="pl-brand">PrivacyLens</div>'
        f'<div class="pl-tagline">Understand before you agree.</div></div></div>',
        unsafe_allow_html=True,
    )


def render_landing() -> None:
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown(
            """<div class="pl-hero"><h2>Read the fine print — in plain language.</h2>
<p>PrivacyLens explains what a privacy policy says, flags what it leaves unclear, and links every finding back
to the original wording so you can check it yourself. It doesn't rate companies or tell you what to do.</p></div>
<div class="pl-steps">
<div class="pl-step"><b>1 · What does it say?</b><span>Five categories summarised from the policy text.</span></div>
<div class="pl-step"><b>2 · What does it mean?</b><span>Legal wording translated into everyday language.</span></div>
<div class="pl-step"><b>3 · Where can I verify it?</b><span>Every claim cites the exact excerpt.</span></div>
</div>""",
            unsafe_allow_html=True,
        )
    with right:
        with st.container(border=True):
            mode = st.radio("Input", ["Website URL", "Paste policy text"], horizontal=True, label_visibility="collapsed")
            if mode == "Website URL":
                url = st.text_input("Privacy policy URL", placeholder="https://example.com/privacy")
                st.caption("Public http(s) pages only. Pages that need JavaScript to show text may not work — paste the text instead.")
                if st.button("Analyze Policy", type="primary", width="stretch", key="analyze_url"):
                    run_pipeline("url", url)
                    st.rerun()
            else:
                text = st.text_area("Privacy policy text", height=230, placeholder="Paste the full privacy policy here…")
                st.caption(f"{len(text.split()):,} words")
                if st.button("Analyze Policy", type="primary", width="stretch", key="analyze_text"):
                    run_pipeline("text", text)
                    st.rerun()
            st.divider()
            st.markdown('<div class="pl-meta">No policy handy? Explore a <b>fictional</b> sample through the same pipeline.</div>',
                        unsafe_allow_html=True)
            if st.button("Try the demo policy", width="stretch", key="demo"):
                run_pipeline("demo", "")
                st.rerun()
    if not settings.llm_enabled:
        st.info(
            "**AI features are off.** Set `OPENAI_API_KEY` (see README) to enable the category overview and Q&A. "
            "Without it you can still process a policy and browse its indexed sections.",
            icon="🔑",
        )


def render_policy_bar(policy: ProcessedPolicy) -> None:
    left, right = st.columns([4, 1], vertical_alignment="center")
    with left:
        demo = '<span class="pl-demo-badge">Fictional demo</span>' if policy.is_demo else ""
        title = esc(policy.title or "Submitted privacy policy")
        if policy.source_url:
            origin = (f'<a href="{html.escape(policy.source_url, quote=True)}" target="_blank" rel="noopener noreferrer">'
                      f"{esc(policy.source_url)}</a>")
        else:
            origin = "Bundled sample document" if policy.is_demo else "Pasted text"
        st.markdown(
            f'<div class="pl-policy-title">{title}{demo}</div>'
            f'<div class="pl-meta">{origin} · {policy.word_count:,} words · {len(policy.chunks)} indexed sections</div>',
            unsafe_allow_html=True,
        )
    with right:
        if st.button("↺ Analyze another policy", width="stretch", key="reset"):
            reset_policy_state()
            st.rerun()


def render_overview_tab(policy: ProcessedPolicy, overview: PolicyOverview | None, llm_reason: str | None) -> None:
    if overview is None:
        if llm_reason:
            st.info(f"{llm_reason}\n\nThe policy has been indexed — open **Source viewer** to browse its sections.", icon="🔑")
        else:
            st.warning("The overview hasn't been generated yet.")
            if st.button("Generate overview", type="primary", key="gen_overview"):
                llm, _ = get_llm()
                with st.spinner("Reading the policy for each privacy category…"):
                    st.session_state.overview = analyze_policy(policy, llm, settings)
                st.rerun()
        return

    st.markdown(
        f'<div class="pl-legend"><span>Evidence status:</span>{pill(EvidenceStatus.CLEARLY_STATED)} the text directly supports it'
        f"{pill(EvidenceStatus.UNCLEAR)} wording is vague or conditional"
        f"{pill(EvidenceStatus.NOT_FOUND)} not identified in the processed policy</div>",
        unsafe_allow_html=True,
    )
    cats = overview.categories
    for row in (cats[0:2], cats[2:4], cats[4:5]):
        cols = st.columns(2, gap="medium")
        for col, cat in zip(cols, row):
            with col:
                render_category(cat, policy)
    failed = [c for c in cats if c.error]
    if failed and st.button(f"Retry {len(failed)} failed categor{'y' if len(failed) == 1 else 'ies'}", key="retry"):
        llm, _ = get_llm()
        with st.spinner("Retrying…"):
            updated = {c.category: c for c in cats}
            for c in failed:
                updated[c.category] = analyze_category(policy, c.category.value, llm, settings)
            st.session_state.overview = PolicyOverview(categories=[updated[c.category] for c in cats])
        st.rerun()


def render_chat_tab(policy: ProcessedPolicy, llm_reason: str | None) -> None:
    if llm_reason:
        st.info(f"{llm_reason}", icon="🔑")
        return

    st.markdown('<div class="pl-meta">Answers use only retrieved excerpts from this policy. '
                "If the policy doesn't address something, PrivacyLens says so instead of guessing.</div>",
                unsafe_allow_html=True)
    st.write("")
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 3].button(q, key=f"suggest_{i}", width="stretch"):
            st.session_state.pending_question = q

    for i, msg in enumerate(st.session_state.chat):
        with st.chat_message(msg["role"], avatar="🙂" if msg["role"] == "user" else "🛡️"):
            if msg["role"] == "user":
                st.markdown(f"<div class='pl-answer'>{esc(msg['content'])}</div>", unsafe_allow_html=True)
            elif msg.get("error"):
                st.markdown(f'<div class="pl-note warn">{esc(msg["error"])}</div>', unsafe_allow_html=True)
            else:
                render_answer(msg["answer"], policy, key=f"ans_{i}")

    typed = st.chat_input("Ask a question about this policy…")
    question = typed or st.session_state.pending_question
    if question:
        st.session_state.pending_question = None
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat if not m.get("error")]
        st.session_state.chat.append({"role": "user", "content": question})
        llm, _ = get_llm()
        with st.spinner("Searching the policy and drafting an answer…"):
            try:
                ans = answer_question(policy, question, llm, history=history, settings=settings)
                st.session_state.chat.append({"role": "assistant", "content": ans.answer, "answer": ans})
            except (LLMError, ValueError) as exc:
                st.session_state.chat.append({"role": "assistant", "content": "", "error": str(exc)})
            except Exception:  # noqa: BLE001
                logging.exception("Unexpected QA failure")
                st.session_state.chat.append(
                    {"role": "assistant", "content": "", "error": "Something went wrong while answering. Please try again."}
                )
        st.rerun()


def render_sources_tab(policy: ProcessedPolicy) -> None:
    st.markdown('<div class="pl-meta">Every citation in PrivacyLens points to one of these sections of the processed policy.</div>',
                unsafe_allow_html=True)
    st.write("")
    labels = {c.id: c.label for c in policy.chunks}
    selected = st.selectbox("Section", options=list(labels), format_func=lambda cid: labels[cid])
    with st.container(border=True):
        render_source(policy.index.get(selected), policy)
    with st.expander("View the full processed policy text"):
        st.markdown(f'<div class="pl-excerpt" style="max-height:600px">{html.escape(policy.text)}</div>',
                    unsafe_allow_html=True)


def render_analysis(policy: ProcessedPolicy) -> None:
    _, llm_reason = get_llm()
    render_policy_bar(policy)
    st.write("")
    tab_overview, tab_chat, tab_sources = st.tabs(["Privacy at a glance", "Ask the policy", "Source viewer"])
    with tab_overview:
        render_overview_tab(policy, st.session_state.overview, llm_reason)
    with tab_chat:
        render_chat_tab(policy, llm_reason)
    with tab_sources:
        render_sources_tab(policy)


def main() -> None:
    init_state()
    render_header()
    if st.session_state.flash_error:
        st.error(st.session_state.flash_error, icon="⚠️")
        st.session_state.flash_error = None
    if st.session_state.policy is None:
        render_landing()
    else:
        render_analysis(st.session_state.policy)
    st.markdown(
        '<div class="pl-disclaimer">PrivacyLens explains what a policy says; it is not legal advice and does not rate '
        "companies. AI summaries can be wrong — use the cited sources to verify. Submitted policies are processed in "
        "memory for this session only and are not stored.</div>",
        unsafe_allow_html=True,
    )


main()
