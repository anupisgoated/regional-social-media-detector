"""
Misinformation & Fake News Detector for Regional Social Media
================================================================
A Streamlit application combining:
  1. Transformer-based linguistic style analysis (zero-shot classification)
  2. Lightweight fact cross-referencing via Wikipedia (keyless, public API)

No API keys required. All inference runs locally or against public,
unauthenticated endpoints.

Author: (your name)
"""

import re
import logging
import requests
from datetime import datetime
from dataclasses import dataclass, field

import streamlit as st
from transformers import pipeline

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
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
STYLE_LABELS = ["credible news", "misinformation", "satire", "propaganda"]
CONFIDENCE_THRESHOLD = 0.45
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
MAX_HISTORY_ITEMS = 25

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
class FactCheckHit:
    entity: str
    summary: str
    url: str

@dataclass
class AnalysisRecord:
    text: str
    timestamp: str
    style: StyleResult
    facts: list = field(default_factory=list)

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_classifier():
    """
    Loads the multilingual zero-shot classifier from Hugging Face Hub.
    Public model — no authentication or API key required.
    """
    logger.info("Loading zero-shot classification model: %s", MODEL_NAME)
    return pipeline("zero-shot-classification", model=MODEL_NAME)


def analyze_style(classifier, text: str) -> StyleResult:
    """Runs zero-shot style classification on the given text."""
    result = classifier(text, candidate_labels=STYLE_LABELS, multi_label=False)
    return StyleResult(
        top_label=result["labels"][0],
        top_score=result["scores"][0],
        all_labels=result["labels"],
        all_scores=result["scores"]
    )

# ============================================================
# FACT CROSS-REFERENCE MODULE (Wikipedia, keyless, public)
# ============================================================
def extract_candidate_entities(text: str) -> list:
    """
    Extracts likely proper-noun phrases (people, places, organizations, events)
    from the text using simple capitalization heuristics.
    This is a lightweight, dependency-free alternative to full NER.
    """
    # Matches sequences of capitalized words (e.g. "Lionel Messi", "World Cup")
    candidates = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", text)
    # Deduplicate while preserving order, drop very short/common words
    seen = set()
    filtered = []
    for c in candidates:
        c_clean = c.strip()
        if c_clean.lower() not in seen and len(c_clean) > 2:
            seen.add(c_clean.lower())
            filtered.append(c_clean)
    return filtered[:5]  # limit to top 5 to keep requests fast


def fetch_wikipedia_summary(entity: str) -> FactCheckHit | None:
    """
    Queries Wikipedia's public REST API for a summary of the given entity.
    No API key required — this is a fully public, unauthenticated endpoint.
    """
    try:
        response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity.replace(' ', '_')}",
            timeout=6,
            headers={"User-Agent": "MisinfoDetectorApp/1.0"}
        )
        if response.status_code == 200:
            data = response.json()
            if "extract" in data and data.get("type") != "disambiguation":
                return FactCheckHit(
                    entity=entity,
                    summary=data["extract"],
                    url=data.get("content_urls", {}).get("desktop", {}).get("page", "")
                )
    except requests.RequestException as e:
        logger.warning("Wikipedia lookup failed for '%s': %s", entity, e)
    return None


def run_fact_cross_reference(text: str) -> list:
    """
    Extracts entities from text and fetches reference summaries from Wikipedia
    so the user can manually verify claims against a real source.
    """
    entities = extract_candidate_entities(text)
    hits = []
    for entity in entities:
        hit = fetch_wikipedia_summary(entity)
        if hit:
            hits.append(hit)
    return hits

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
        "1. **Style Analysis** — a transformer model estimates whether the "
        "*writing style* resembles credible news, misinformation, satire, "
        "or propaganda.\n"
        "2. **Fact Cross-Reference** — pulls real Wikipedia summaries for "
        "named entities in the text (people, places, events) so you can "
        "manually verify specific claims."
    )
    st.markdown("---")
    st.subheader("⚙️ Technical details")
    st.write(
        f"- Style model: `{MODEL_NAME}`\n"
        "- Fact source: Wikipedia REST API (public, keyless)\n"
        "- 100+ languages supported for style analysis\n"
        "- No data stored server-side beyond your session"
    )
    st.markdown("---")
    st.warning(
        "**Important limitation:** Style Analysis judges *how text is written*, "
        "not whether it's true. A factually correct sentence can score as "
        "'misinformation' if it's phrased unusually, and a well-written lie "
        "can score as 'credible'. Always check the Fact Cross-Reference panel "
        "and independent sources before drawing conclusions.",
        icon="⚠️"
    )

# ============================================================
# HEADER
# ============================================================
st.markdown(f'<p class="main-header">{APP_ICON} {APP_TITLE}</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Style analysis + real fact cross-referencing for regional social media content — multilingual and keyless.</p>',
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
        if not text_input.strip():
            st.warning("Please enter some text before analyzing.")
        elif len(text_input.strip()) < 10:
            st.warning("Text is too short for reliable analysis. Please provide a full sentence or more.")
        else:
            try:
                with st.spinner("Loading model (first run may take a moment)..."):
                    classifier = load_classifier()

                with st.spinner("Running style analysis..."):
                    style_result = analyze_style(classifier, text_input)

                with st.spinner("Cross-referencing facts against Wikipedia..."):
                    fact_hits = run_fact_cross_reference(text_input)

                record = AnalysisRecord(
                    text=text_input,
                    timestamp=datetime.now().strftime("%d %b %Y, %H:%M"),
                    style=style_result,
                    facts=fact_hits
                )
                add_to_history(record)

                # ---------- STYLE ANALYSIS OUTPUT ----------
                st.markdown("### 📊 Style Analysis")
                st.caption("Estimates writing style — does NOT verify factual accuracy.")

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

                with st.expander("Full confidence breakdown"):
                    for label, score in zip(style_result.all_labels, style_result.all_scores):
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.write(f"{LABEL_META[label]['emoji']} {label.title()}")
                        with c2:
                            st.progress(float(score), text=f"{score*100:.1f}%")

                # ---------- FACT CROSS-REFERENCE OUTPUT ----------
                st.markdown("### 📚 Fact Cross-Reference")
                st.caption(
                    "Real Wikipedia summaries for named entities detected in your text. "
                    "Compare these against the claim to check accuracy."
                )

                if fact_hits:
                    for hit in fact_hits:
                        st.markdown(
                            f"""
                            <div class="fact-card">
                                <b>🔎 {hit.entity}</b><br>
                                <span style="color:#374151;">{hit.summary}</span><br>
                                <a href="{hit.url}" target="_blank">Read more on Wikipedia →</a>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.write(
                        "No recognizable named entities were found, or no matching "
                        "Wikipedia articles were available for cross-referencing."
                    )

                st.markdown("---")
                st.caption(
                    "⚠️ **Disclaimer:** This tool provides an automated estimate and reference "
                    "material — it is not a certified fact-check. For verified claims, consult "
                    "trusted fact-checking organizations such as BOOM, Alt News, or PolitiFact."
                )

            except Exception as e:
                logger.exception("Analysis failed")
                st.error(f"Something went wrong while analyzing the text: {e}")
                st.caption("Try refreshing the page or shortening the input text.")

# ============================================================
# TAB 2: HISTORY
# ============================================================
with tab_history:
    st.markdown("### 🕘 Recent Analyses")
    if not st.session_state.history:
        st.write("No analyses yet in this session.")
    else:
        for i, record in enumerate(st.session_state.history):
            meta = LABEL_META[record.style.top_label]
            with st.container():
                st.markdown(
                    f"""
                    <div class="history-item">
                        <b>{meta['emoji']} {record.style.top_label.title()}</b>
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
        "**1. Style Analysis (Zero-Shot Classification)**\n\n"
        f"Uses `{MODEL_NAME}`, a multilingual model trained on natural language "
        "inference across 100+ languages. Given a piece of text and a set of "
        "candidate labels, it estimates how well the text's phrasing and framing "
        "align with each label — without any task-specific fine-tuning."
    )
    st.write(
        "**2. Fact Cross-Reference (Wikipedia)**\n\n"
        "Extracts capitalized multi-word phrases as candidate named entities "
        "(a lightweight heuristic, not full NER), then queries Wikipedia's public "
        "summary API for each. This surfaces real reference information next to "
        "the claim, so the user — not the model — makes the final judgment on accuracy."
    )
    st.write(
        "**Why this matters:** Zero-shot style classifiers cannot verify facts. "
        "They have no access to a knowledge base and cannot distinguish a true "
        "statement from a false one if both are phrased similarly. Combining a "
        "style signal with a real, independent fact source gives a more honest "
        "and useful result than either alone."
    )
    st.markdown("---")
    st.write(
        "**Suggested next step for production use:** fine-tune the style model "
        "on a labeled regional dataset (e.g. IFND, FakeNewsNet, or a custom-labeled "
        "corpus of regional social media posts) to better capture local slang, "
        "code-mixing, and platform-specific patterns."
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f'<p class="footer-note">{APP_ICON} {APP_TITLE} · Built with 🤗 Transformers + Streamlit + Wikipedia API · '
    "No API key required · No data stored server-side</p>",
    unsafe_allow_html=True
                        )
