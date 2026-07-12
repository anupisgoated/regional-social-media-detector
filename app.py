"""
Misinformation & Fake News Detector for Regional Social Media
================================================================
Combines:
  1. Transformer-based linguistic style analysis (zero-shot, local, keyless)
  2. Gemini API-based real fact verification (free tier, requires API key)

The API key is NEVER hardcoded. It is read from Streamlit secrets
(st.secrets) or entered securely by the user in the sidebar.

Author: (your name)
"""

import re
import logging
from datetime import datetime
from dataclasses import dataclass, field

import streamlit as st
from transformers import pipeline
import google.generativeai as genai

# ============================================================
# LOGGING CONFIG
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("misinfo_detector")

# ============================================================
# APP CONFIG
# ============================================================
APP_TITLE = "Misinformation Detector"
APP_ICON = "🛡️"
STYLE_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
STYLE_LABELS = ["credible news", "misinformation", "satire", "propaganda"]
CONFIDENCE_THRESHOLD = 0.45
MAX_HISTORY_ITEMS = 25
GEMINI_MODEL_NAME = "gemini-1.5-flash"  # fast + free-tier friendly

LABEL_META = {
    "credible news":  {"emoji": "✅", "color": "#16a34a"},
    "misinformation": {"emoji": "⚠️", "color": "#dc2626"},
    "satire":         {"emoji": "🎭", "color": "#ca8a04"},
    "propaganda":     {"emoji": "📢", "color": "#dc2626"},
}

st.set_page_config(
    page_title=f"{APP_TITLE} | Regional Social Media",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
    <style>
    .main-header { font-size: 2.3rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { color: #6b7280; font-size: 1rem; margin-top: 0.2rem; margin-bottom: 1.5rem; }
    .result-card { padding: 1.2rem 1.5rem; border-radius: 12px; margin: 1rem 0; }
    .fact-card { padding: 1rem 1.2rem; border-radius: 10px; background-color: #f3f4f6;
                 border-left: 5px solid #2563eb; margin-bottom: 0.8rem; }
    .footer-note { text-align: center; color: #9ca3af; font-size: 0.8rem; margin-top: 2rem; }
    .history-item { padding: 0.6rem 0.8rem; border-radius: 8px; background-color: #f9fafb;
                    margin-bottom: 0.5rem; border: 1px solid #e5e7eb; }
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
    verdict: str          # "True", "False", "Partially True", "Unverifiable"
    explanation: str
    confidence: str        # "High", "Medium", "Low"

@dataclass
class AnalysisRecord:
    text: str
    timestamp: str
    style: StyleResult
    fact: FactVerdict = None

# ============================================================
# API KEY HANDLING  (Google Gemini — free tier)
# ============================================================
def get_gemini_api_key() -> str:
    """
    Retrieves the Gemini API key from Streamlit secrets first
    (recommended for deployment), falling back to a sidebar input
    for local/manual testing. The key is never stored in code.
    """
    # Preferred: Streamlit Cloud secrets (Settings -> Secrets ->
    # GOOGLE_API_KEY = "your-key-here")
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]

    # Fallback: manual entry in sidebar (session-only, not saved anywhere)
    return st.session_state.get("manual_api_key", "")


def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


# ============================================================
# MODEL LOADING — STYLE ANALYSIS (local, keyless)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_classifier():
    logger.info("Loading zero-shot classification model: %s", STYLE_MODEL_NAME)
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
# FACT VERIFICATION MODULE (Gemini API)
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
    """
    Sends the text to Gemini for real factual verification.
    Unlike the local style model, Gemini has actual world knowledge
    and can correctly judge claims like "Messi won the 2022 World Cup".
    """
    prompt = FACT_CHECK_PROMPT.format(text=text)
    response = model.generate_content(prompt)
    raw = response.text.strip()

    verdict = "Unverifiable"
    confidence = "Low"
    explanation = raw

    verdict_match = re.search(r"Verdict:\s*(.+)", raw)
    confidence_match = re.search(r"Confidence:\s*(.+)", raw)
    explanation_match = re.search(r"Explanation:\s*(.+)", raw, re.DOTALL)

    if verdict_match:
        verdict = verdict_match.group(1).strip()
    if confidence_match:
        confidence = confidence_match.group(1).strip()
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    return FactVerdict(verdict=verdict, explanation=explanation, confidence=confidence)


VERDICT_META = {
    "True":            {"emoji": "✅", "color": "#16a34a"},
    "False":           {"emoji": "❌", "color": "#dc2626"},
    "Partially True":  {"emoji": "🟡", "color": "#ca8a04"},
    "Unverifiable":    {"emoji": "❔", "color": "#6b7280"},
}

# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

def add_to_history(record: AnalysisRecord):
    st.session_state.history.insert(0, record)
    st.session_state.history = st.session_state.history[:MAX_HISTORY_ITEMS]

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header(f"{APP_ICON} About this tool")
    st.write(
        "This tool combines two independent checks on social media text "
        "or news snippets, particularly for **regional / multilingual content**:"
    )
    st.markdown(
        "1. **Style Analysis** — a local transformer model estimates whether "
        "the *writing style* resembles credible news, misinformation, satire, "
        "or propaganda.\n"
        "2. **Fact Verification** — Google Gemini checks the claim against "
        "real-world knowledge and returns a True/False/Partially True verdict."
    )
    st.markdown("---")

    st.subheader("🔑 API Key (Google Gemini — free)")
    api_key_from_secrets = "GOOGLE_API_KEY" in st.secrets

    if api_key_from_secrets:
        st.success("API key loaded from Streamlit secrets ✅")
    else:
        st.info(
            "No key found in secrets. Paste your free Gemini API key below "
            "for this session, or add it permanently in **Settings → Secrets** "
            "on Streamlit Cloud as `GOOGLE_API_KEY`."
        )
        manual_key = st.text_input(
            "Paste your Gemini API key",
            type="password",
            help="Get a free key at https://aistudio.google.com/apikey"
        )
        if manual_key:
            st.session_state["manual_api_key"] = manual_key
            st.success("Key set for this session.")

    st.markdown("---")
    st.warning(
        "**Note:** Style Analysis judges *how* text is written, not whether "
        "it's true. Fact Verification (Gemini) checks the actual claim. "
        "Use both together for the most reliable read.",
        icon="⚠️"
    )

# ============================================================
# HEADER
# ============================================================
st.markdown(f'<p class="main-header">{APP_ICON} {APP_TITLE}</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Style analysis + Gemini-powered fact verification for regional social media content.</p>',
    unsafe_allow_html=True
)

tab_analyze, tab_history, tab_about = st.tabs(["🔍 Analyze", "🕘 History", "📘 Methodology"])

# ============================================================
# TAB 1: ANALYZE
# ============================================================
with tab_analyze:
    text_input = st.text_area(
        "Paste a post, headline, or message to analyze:",
        height=180,
        placeholder="e.g. 'Lionel Messi won the 2022 FIFA World Cup with Argentina.'"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_btn = st.button("🔍 Run Full Analysis", use_container_width=True, type="primary")
    with col2:
        clear_btn = st.button("🧹 Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    if analyze_btn:
        api_key = get_gemini_api_key()

        if not text_input.strip():
            st.warning("Please enter some text before analyzing.")
        elif len(text_input.strip()) < 10:
            st.warning("Text is too short for reliable analysis. Please provide a full sentence or more.")
        elif not api_key:
            st.error(
                "No Gemini API key found. Add one in the sidebar, or set "
                "`GOOGLE_API_KEY` in Streamlit Cloud secrets, to enable fact verification."
            )
        else:
            try:
                with st.spinner("Loading style model (first run may take a moment)..."):
                    classifier = load_classifier()

                with st.spinner("Running style analysis..."):
                    style_result = analyze_style(classifier, text_input)

                with st.spinner("Verifying facts with Gemini..."):
                    gemini_model = configure_gemini(api_key)
                    fact_result = run_fact_verification(gemini_model, text_input)

                record = AnalysisRecord(
                    text=text_input,
                    timestamp=datetime.now().strftime("%d %b %Y, %H:%M"),
                    style=style_result,
                    fact=fact_result
                )
                add_to_history(record)

                # ---------- FACT VERIFICATION OUTPUT (shown first — most important) ----------
                st.markdown("### 🧠 Fact Verification (Gemini)")
                vmeta = VERDICT_META.get(fact_result.verdict, VERDICT_META["Unverifiable"])
                st.markdown(
                    f"""
                    <div class="result-card" style="background-color:{vmeta['color']}15; border-left: 5px solid {vmeta['color']};">
                        <span style="font-size:1.3rem;">{vmeta['emoji']} <b>{fact_result.verdict}</b></span>
                        <span style="color:#6b7280;"> · Confidence: {fact_result.confidence}</span><br><br>
                        <span>{fact_result.explanation}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ---------- STYLE ANALYSIS OUTPUT ----------
                st.markdown("### 📊 Style Analysis")
                st.caption("Estimates writing style — separate from factual accuracy above.")

                meta = LABEL_META[style_result.top_label]
                if style_result.top_score < CONFIDENCE_THRESHOLD:
                    st.info(
                        f"🤔 **Low confidence.** Leans toward **{style_result.top_label.upper()}** "
                        f"at only {style_result.top_score*100:.1f}% — treat as inconclusive."
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card" style="background-color:{meta['color']}15; border-left: 5px solid {meta['color']};">
                            <span style="font-size:1.3rem;">{meta['emoji']} <b>{style_result.top_label.title()}</b></span><br>
                            <span style="color:{meta['color']}; font-size:1.1rem;">{style_result.top_score*100:.1f}% confidence</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with st.expander("Full style confidence breakdown"):
                    for label, score in zip(style_result.all_labels, style_result.all_scores):
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.write(f"{LABEL_META[label]['emoji']} {label.title()}")
                        with c2:
                            st.progress(float(score), text=f"{score*100:.1f}%")

                st.markdown("---")
                st.caption(
                    "⚠️ **Disclaimer:** Automated analysis, not a certified fact-check. "
                    "For high-stakes claims, cross-check with trusted fact-checking "
                    "organizations such as BOOM, Alt News, or PolitiFact."
                )

            except Exception as e:
                logger.exception("Analysis failed")
                st.error(f"Something went wrong while analyzing the text: {e}")
                st.caption("Check that your API key is valid, or try again in a moment.")

# ============================================================
# TAB 2: HISTORY
# ============================================================
with tab_history:
    st.markdown("### 🕘 Recent Analyses")
    if not st.session_state.history:
        st.write("No analyses yet in this session.")
    else:
        for record in st.session_state.history:
            meta = LABEL_META[record.style.top_label]
            vmeta = VERDICT_META.get(record.fact.verdict, VERDICT_META["Unverifiable"]) if record.fact else VERDICT_META["Unverifiable"]
            st.markdown(
                f"""
                <div class="history-item">
                    <b>{vmeta['emoji']} {record.fact.verdict if record.fact else 'N/A'}</b>
                    &nbsp;|&nbsp; {meta['emoji']} {record.style.top_label.title()}
                    ({record.style.top_score*100:.1f}%) — <i>{record.timestamp}</i><br>
                    <span style="color:#6b7280;">{record.text[:140]}{"..." if len(record.text) > 140 else ""}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()

# ============================================================
# TAB 3: METHODOLOGY / ABOUT
# ============================================================
with tab_about:
    st.markdown("### 📘 Methodology")
    st.write(
        "**1. Style Analysis (Zero-Shot Classification, local)**\n\n"
        f"Uses `{STYLE_MODEL_NAME}`, a multilingual model covering 100+ languages. "
        "Estimates whether the *phrasing and framing* of text resembles credible "
        "news, misinformation, satire, or propaganda — without checking facts."
    )
    st.write(
        "**2. Fact Verification (Google Gemini API)**\n\n"
        f"Uses Google's `{GEMINI_MODEL_NAME}` model, which has real-world "
        "knowledge, to evaluate the actual truth of the claim and return a "
        "verdict: True, False, Partially True, or Unverifiable — with an "
        "explanation and confidence level."
    )
    st.write(
        "**Why combine both?** Style analysis alone cannot tell true from "
        "false statements written in a similar tone. Fact verification alone "
        "doesn't capture manipulative *framing* (e.g. technically true but "
        "misleading headlines). Together, they give a fuller picture."
    )
    st.markdown("---")
    st.write(
        "**Getting a free Gemini API key:**\n"
        "1. Go to https://aistudio.google.com/apikey\n"
        "2. Sign in with a Google account\n"
        "3. Click **Create API key** (free tier, no billing required for basic use)\n"
        "4. Add it in Streamlit Cloud under **Settings → Secrets** as:\n"
    )
    st.code('GOOGLE_API_KEY = "your-key-here"', language="toml")

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f'<p class="footer-note">{APP_ICON} {APP_TITLE} · Built with 🤗 Transformers + Streamlit + Google Gemini · '
    "API key stored securely via Streamlit secrets</p>",
    unsafe_allow_html=True
)
