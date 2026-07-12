"""
VERDICT WIRE
================================================================
A regional-language misinformation detector for social media
dispatches, filed by two independent desks:

  STYLE DESK  — a local, multilingual transformer reads HOW a
                dispatch is written (zero-shot classification,
                no API key required).

  FACT DESK   — Google Gemini checks WHAT a dispatch claims
                against real-world knowledge (needs a free
                Gemini API key, stored in Streamlit secrets).

No dispatch text is stored beyond the current browser session.
"""

import re
import logging
from datetime import datetime
from dataclasses import dataclass

import streamlit as st
from transformers import pipeline
import google.generativeai as genai

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("verdict_wire")

# ============================================================
# CONSTANTS
# ============================================================
APP_NAME = "VERDICT WIRE"
APP_ICON = "🗞️"
STYLE_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
STYLE_LABELS = ["credible news", "misinformation", "satire", "propaganda"]
CONFIDENCE_THRESHOLD = 0.45
MAX_LOG_ITEMS = 25
GEMINI_MODEL_NAME = "gemini-2.5-flash"

INK = {
    "true": "#3F8768",     # verified — muted forest green
    "false": "#B23A2E",    # flagged — wire-service crimson
    "caution": "#C98A2C",  # satire / propaganda / partial — amber
    "neutral": "#888C93",  # unverifiable — fog grey
}

STYLE_META = {
    "credible news":  {"ink": INK["true"],    "stamp": "CREDIBLE"},
    "misinformation": {"ink": INK["false"],   "stamp": "MISINFORMATION"},
    "satire":         {"ink": INK["caution"], "stamp": "SATIRE"},
    "propaganda":     {"ink": INK["false"],   "stamp": "PROPAGANDA"},
}

VERDICT_INK = {
    "True": INK["true"],
    "False": INK["false"],
    "Partially True": INK["caution"],
    "Unverifiable": INK["neutral"],
}

st.set_page_config(
    page_title=f"{APP_NAME} · Regional Desk",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE INJECTION
# ============================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --ink-black: #15171B;
        --panel: #1E2127;
        --panel-2: #24272E;
        --rule: #34373F;
        --paper: #EDEAE2;
        --fog: #8B8E95;
        --crimson: #B23A2E;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--ink-black);
        color: var(--paper);
        font-family: 'IBM Plex Sans', sans-serif;
    }

    #MainMenu, footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }

    [data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--rule);
    }
    [data-testid="stSidebar"] * { color: var(--paper); }

    /* ---------- Masthead ---------- */
    .masthead {
        display:flex; align-items:center; justify-content:space-between;
        padding-bottom:0.9rem; margin-bottom:0.6rem;
        border-bottom:3px double var(--rule);
        flex-wrap: wrap; gap: 0.4rem;
    }
    .masthead-word {
        font-family:'IBM Plex Mono', monospace; font-weight:700;
        letter-spacing:0.12em; font-size:1.05rem; color:var(--paper);
    }
    .masthead-word span { color:var(--crimson); }
    .masthead-meta {
        font-family:'IBM Plex Mono', monospace; font-size:0.72rem;
        color:var(--fog); letter-spacing:0.06em;
    }
    .live-dot {
        display:inline-block; width:7px; height:7px; border-radius:50%;
        background:var(--crimson); margin-right:0.4rem;
    }
    @media (prefers-reduced-motion: no-preference) {
        .live-dot { animation: pulse 2.2s infinite; }
    }
    @keyframes pulse {
        0% { box-shadow:0 0 0 0 rgba(178,58,46,0.5); }
        70% { box-shadow:0 0 0 6px rgba(178,58,46,0); }
        100% { box-shadow:0 0 0 0 rgba(178,58,46,0); }
    }

    /* ---------- Hero ---------- */
    .eyebrow {
        font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.18em;
        color:var(--crimson); margin-bottom:0.5rem; text-transform:uppercase;
    }
    .hero-title {
        font-family:'IBM Plex Mono', monospace; font-weight:700;
        font-size:clamp(1.6rem, 6vw, 2.4rem); letter-spacing:0.01em;
        line-height:1.2; margin:0 0 0.6rem 0; color:var(--paper);
    }
    .hero-title .accent { color:var(--crimson); }
    .hero-sub { color:var(--fog); font-size:0.96rem; max-width:600px; margin-bottom:0.3rem; }

    /* ---------- Tabs ---------- */
    [data-testid="stTabs"] button {
        font-family:'IBM Plex Mono',monospace; color:var(--fog);
        font-size:0.85rem; letter-spacing:0.04em;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color:var(--paper); border-bottom:2px solid var(--crimson);
    }

    /* ---------- Inputs ---------- */
    [data-testid="stTextArea"] textarea {
        background:var(--panel) !important; color:var(--paper) !important;
        border:1px solid var(--rule) !important; border-radius:4px !important;
        font-family:'IBM Plex Mono', monospace !important; font-size:0.9rem !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color:var(--crimson) !important;
        box-shadow:0 0 0 2px rgba(178,58,46,0.25) !important;
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        font-family:'IBM Plex Mono',monospace; letter-spacing:0.04em; font-weight:600;
        border-radius:4px !important; border:1px solid var(--rule) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton>button:hover { transform:translateY(-1px); }
    button[kind="primary"] {
        background:var(--crimson) !important; border-color:var(--crimson) !important;
        color:var(--paper) !important;
    }
    button[kind="primary"]:hover { box-shadow:0 6px 16px rgba(178,58,46,0.35); }

    /* ---------- Stamp (signature element) ---------- */
    .stamp-wrap { margin:0.6rem 0 1rem 0; }
    .stamp {
        display:inline-block; font-family:'IBM Plex Mono', monospace; font-weight:700;
        font-size:1.05rem; letter-spacing:0.06em; text-transform:uppercase;
        padding:0.55rem 1.1rem; border:3px solid currentColor;
        border-radius:6px; transform:rotate(-3deg); position:relative;
    }
    .stamp::after {
        content:""; position:absolute; inset:3px; border:1px solid currentColor;
        border-radius:3px; opacity:0.5;
    }
    @media (prefers-reduced-motion: no-preference) {
        .stamp { animation: stampdown 0.4s cubic-bezier(.2,.8,.3,1.2) both; }
    }
    @keyframes stampdown {
        0% { opacity:0; transform:rotate(-3deg) scale(1.6); }
        100% { opacity:1; transform:rotate(-3deg) scale(1); }
    }

    /* ---------- Panels ---------- */
    .desk-panel {
        background:var(--panel); border:1px solid var(--rule); border-radius:6px;
        padding:1.1rem 1.3rem; margin-bottom:1rem;
    }
    .desk-label {
        font-family:'IBM Plex Mono',monospace; font-size:0.75rem; letter-spacing:0.14em;
        color:var(--fog); text-transform:uppercase; margin-bottom:0.5rem;
    }

    /* ---------- Redaction bars ---------- */
    .bar-row { display:flex; align-items:center; gap:0.7rem; margin-bottom:0.5rem; }
    .bar-label { font-family:'IBM Plex Mono',monospace; font-size:0.76rem; width:130px; flex-shrink:0; }
    .bar-track { flex:1; background:var(--panel-2); border-radius:2px; height:9px; overflow:hidden; }
    .bar-fill { height:100%; border-radius:2px; }
    .bar-pct { font-family:'IBM Plex Mono',monospace; font-size:0.76rem; color:var(--fog); width:44px; text-align:right; flex-shrink:0; }

    /* ---------- Log rows ---------- */
    .log-row { padding:0.65rem 0; border-bottom:1px solid var(--rule); }
    .log-row:last-child { border-bottom:none; }
    .log-meta { font-family:'IBM Plex Mono',monospace; font-size:0.74rem; color:var(--fog); margin-bottom:0.25rem; }
    .chip {
        display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
        letter-spacing:0.05em; padding:0.12rem 0.5rem; border-radius:3px;
        margin-right:0.4rem; border:1px solid currentColor;
    }
    .log-text { color:var(--fog); font-size:0.87rem; margin-top:0.3rem; }

    /* ---------- Footer ---------- */
    .site-footer {
        text-align:center; color:var(--fog); font-family:'IBM Plex Mono',monospace;
        font-size:0.7rem; letter-spacing:0.04em; margin-top:2rem; padding-top:1rem;
        border-top:1px solid var(--rule);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# DATA MODELS
# ============================================================
@dataclass
class StyleResult:
    top_label: str
    top_score: float
    all_labels: list
    all_scores: list

@dataclass
class FactVerdict:
    verdict: str
    explanation: str
    confidence: str

@dataclass
class LogEntry:
    text: str
    timestamp: str
    style: StyleResult
    fact: FactVerdict = None

# ============================================================
# GEMINI API KEY
# ============================================================
def get_gemini_api_key() -> str:
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return st.session_state.get("manual_api_key", "")

def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL_NAME)

# ============================================================
# STYLE DESK
# ============================================================
@st.cache_resource(show_spinner=False)
def load_classifier():
    logger.info("Loading style model: %s", STYLE_MODEL_NAME)
    return pipeline("zero-shot-classification", model=STYLE_MODEL_NAME)

def analyze_style(classifier, text: str) -> StyleResult:
    result = classifier(text, candidate_labels=STYLE_LABELS, multi_label=False)
    return StyleResult(
        top_label=result["labels"][0],
        top_score=result["scores"][0],
        all_labels=result["labels"],
        all_scores=result["scores"]
    )

# ============================================================
# FACT DESK
# ============================================================
FACT_CHECK_PROMPT = """You are a careful fact-checking assistant. Analyze the following claim/post \
for factual accuracy based on real-world knowledge.

Claim:
\"\"\"{text}\"\"\"

Respond in EXACTLY this format, with no extra commentary:
Verdict: <True | False | Partially True | Unverifiable>
Confidence: <High | Medium | Low>
Explanation: <2-3 sentence explanation of why, citing the relevant real facts>
"""

def run_fact_verification(model, text: str) -> FactVerdict:
    prompt = FACT_CHECK_PROMPT.format(text=text)
    response = model.generate_content(prompt)
    raw = response.text.strip()

    verdict, confidence, explanation = "Unverifiable", "Low", raw
    v_match = re.search(r"Verdict:\s*(.+)", raw)
    c_match = re.search(r"Confidence:\s*(.+)", raw)
    e_match = re.search(r"Explanation:\s*(.+)", raw, re.DOTALL)

    if v_match: verdict = v_match.group(1).strip()
    if c_match: confidence = c_match.group(1).strip()
    if e_match: explanation = e_match.group(1).strip()

    return FactVerdict(verdict=verdict, explanation=explanation, confidence=confidence)

# ============================================================
# RENDER HELPERS
# ============================================================
def render_masthead():
    now_str = datetime.now().strftime("%d %b %Y · %H:%M")
    st.markdown(f"""
    <div class="masthead">
        <span class="masthead-word">VERDICT <span>WIRE</span></span>
        <span class="masthead-meta"><span class="live-dot"></span>REGIONAL DESK · {now_str}</span>
    </div>
    """, unsafe_allow_html=True)

def render_hero():
    st.markdown("""
    <div class="eyebrow">DISPATCH REVIEW // REGIONAL SOCIAL MEDIA</div>
    <div class="hero-title">TWO DESKS.<br>ONE <span class="accent">VERDICT</span>.</div>
    <p class="hero-sub">Paste a post, headline, or forward in any of 100+ languages.
    The Style Desk reads how it's written; the Fact Desk checks what it claims.</p>
    """, unsafe_allow_html=True)

def render_stamp(text: str, ink: str):
    st.markdown(f'<div class="stamp-wrap"><div class="stamp" style="color:{ink};">{text}</div></div>',
                unsafe_allow_html=True)

def chip_html(text: str, ink: str) -> str:
    return f'<span class="chip" style="color:{ink};">{text}</span>'

def render_bar(label: str, pct: float, ink: str):
    st.markdown(f"""
    <div class="bar-row">
        <div class="bar-label">{label.upper()}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{pct*100:.1f}%; background:{ink};"></div></div>
        <div class="bar-pct">{pct*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "log" not in st.session_state:
    st.session_state.log = []
if "totals" not in st.session_state:
    st.session_state.totals = {"count": 0, "True": 0, "False": 0, "Partially True": 0, "Unverifiable": 0}

def add_to_log(entry: LogEntry):
    st.session_state.log.insert(0, entry)
    st.session_state.log = st.session_state.log[:MAX_LOG_ITEMS]
    st.session_state.totals["count"] += 1
    if entry.fact and entry.fact.verdict in st.session_state.totals:
        st.session_state.totals[entry.fact.verdict] += 1

# ============================================================
# MAIN
# ============================================================
inject_css()

with st.sidebar:
    st.markdown('<div class="desk-label">About this desk</div>', unsafe_allow_html=True)
    st.write("VERDICT WIRE runs every dispatch past two independent desks before filing a result — built for regional and multilingual social media text.")
    st.markdown("---")
    st.markdown('<div class="desk-label">Style Desk</div>', unsafe_allow_html=True)
    st.write("A local transformer reads phrasing across 100+ languages. No key needed.")
    st.markdown('<div class="desk-label">Fact Desk</div>', unsafe_allow_html=True)
    st.write("Google Gemini checks the claim against real-world knowledge.")
    st.markdown("---")
    st.markdown('<div class="desk-label">Gemini API key</div>', unsafe_allow_html=True)

    key_present = "GOOGLE_API_KEY" in st.secrets
    if key_present:
        st.markdown(chip_html("● KEY LOADED", INK["true"]), unsafe_allow_html=True)
    else:
        st.markdown(chip_html("○ NO KEY FOUND", INK["false"]), unsafe_allow_html=True)
        st.caption("Paste a free key for this session, or add `GOOGLE_API_KEY` under Settings → Secrets on Streamlit Cloud.")
        manual_key = st.text_input("Gemini API key", type="password", label_visibility="collapsed",
                                    placeholder="Paste key for this session only")
        if manual_key:
            st.session_state["manual_api_key"] = manual_key
            st.success("Key set for this session.")

    st.markdown("---")
    st.caption("Style Desk judges HOW a dispatch is written, not whether it's true. Fact Desk checks the claim. Read both stamps before drawing a conclusion.")

render_masthead()
render_hero()

tab_analyze, tab_log, tab_manual = st.tabs(["Analyze", "Log", "Manual"])

# ---------------- ANALYZE ----------------
with tab_analyze:
    st.markdown('<div class="desk-label">Paste the dispatch</div>', unsafe_allow_html=True)
    text_input = st.text_area(
        label="Dispatch text",
        height=170,
        placeholder="e.g. Ronaldo has won a world cup.",
        label_visibility="collapsed",
        key="dispatch_text"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_btn = st.button("Run Analysis", use_container_width=True, type="primary")
    with col2:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state.dispatch_text = ""
        st.rerun()

    if analyze_btn:
        api_key = get_gemini_api_key()
        stripped = text_input.strip()

        if not stripped:
            st.warning("This dispatch is empty. Paste a post, headline, or message before running the analysis.")
        elif len(stripped) < 10:
            st.warning("This dispatch is too short to read reliably. Add a full sentence or more.")
        elif not api_key:
            st.error("No Gemini key on file. Add one in the sidebar, or set `GOOGLE_API_KEY` in Streamlit secrets, to run the Fact Desk.")
        else:
            try:
                with st.spinner("Style Desk reading the dispatch..."):
                    classifier = load_classifier()
                    style_result = analyze_style(classifier, stripped)

                with st.spinner("Fact Desk checking the claim..."):
                    gemini_model = configure_gemini(api_key)
                    fact_result = run_fact_verification(gemini_model, stripped)

                entry = LogEntry(
                    text=stripped,
                    timestamp=datetime.now().strftime("%d %b %Y, %H:%M"),
                    style=style_result,
                    fact=fact_result
                )
                add_to_log(entry)

                # ---- FACT DESK ----
                st.markdown('<div class="desk-label">Fact Desk — Gemini verdict</div>', unsafe_allow_html=True)
                fact_ink = VERDICT_INK.get(fact_result.verdict, INK["neutral"])
                render_stamp(fact_result.verdict.upper(), fact_ink)
                st.markdown(f"""
                <div class="desk-panel">
                    <div style="color:{fact_ink}; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; margin-bottom:0.5rem;">
                        CONFIDENCE: {fact_result.confidence.upper()}
                    </div>
                    <div>{fact_result.explanation}</div>
                </div>
                """, unsafe_allow_html=True)

                # ---- STYLE DESK ----
                st.markdown('<div class="desk-label">Style Desk — writing pattern</div>', unsafe_allow_html=True)
                smeta = STYLE_META[style_result.top_label]
                if style_result.top_score < CONFIDENCE_THRESHOLD:
                    st.info(f"Inconclusive. Closest reading is {style_result.top_label.upper()} at only {style_result.top_score*100:.1f}%.")
                else:
                    st.markdown(f"""
                    <div class="desk-panel">
                        <span class="chip" style="color:{smeta['ink']};">{smeta['stamp']}</span>
                        <span style="font-family:'IBM Plex Mono',monospace; color:var(--fog); font-size:0.82rem;"> · {style_result.top_score*100:.1f}% confidence</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="desk-label">Full readout</div>', unsafe_allow_html=True)
                for label, score in zip(style_result.all_labels, style_result.all_scores):
                    render_bar(label, score, STYLE_META[label]["ink"])

                st.markdown("---")
                st.caption("This is an automated read, not a certified fact-check. For high-stakes claims, check a certified fact-checking desk such as BOOM, Alt News, or PolitiFact.")

            except Exception as e:
                logger.exception("Analysis failed")
 
