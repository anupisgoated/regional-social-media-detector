import streamlit as st
import requests
import re
import time
from transformers import pipeline

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ============================================================
# CUSTOM CSS — animated newspaper theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Special+Elite&display=swap');

.stApp {
    background: linear-gradient(135deg, #1a1a1a 0%, #2b2b2b 50%, #1a1a1a 100%);
}

/* Floating newspaper icons */
.floating-emoji {
    position: fixed;
    font-size: 28px;
    opacity: 0.15;
    animation: floatUp 12s linear infinite;
    z-index: 0;
}
@keyframes floatUp {
    0% { transform: translateY(110vh) rotate(0deg); }
    100% { transform: translateY(-10vh) rotate(360deg); }
}

/* Title styling */
.newspaper-title {
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    font-size: 2.6rem;
    text-align: center;
    background: linear-gradient(90deg, #ff4b4b, #ffcc00, #ff4b4b);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 3s linear infinite;
    margin-bottom: 0;
}
@keyframes shine {
    to { background-position: 200% center; }
}

.subtitle-type {
    font-family: 'Special Elite', monospace;
    text-align: center;
    color: #cccccc;
    border-top: 2px dashed #888;
    border-bottom: 2px dashed #888;
    padding: 8px 0;
    margin-bottom: 15px;
}

/* Scrolling breaking news ticker */
.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: #b30000;
    padding: 6px 0;
    margin-bottom: 20px;
    border-radius: 4px;
    box-shadow: 0 0 12px rgba(255,0,0,0.5);
}
.ticker {
    display: inline-block;
    white-space: nowrap;
    animation: scrollLeft 14s linear infinite;
    font-family: 'Special Elite', monospace;
    color: white;
    font-weight: bold;
    letter-spacing: 1px;
}
@keyframes scrollLeft {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}

/* Result cards */
.result-card {
    border-radius: 10px;
    padding: 18px;
    margin: 10px 0;
    animation: fadeInPop 0.6s ease;
    border: 1px solid rgba(255,255,255,0.15);
}
@keyframes fadeInPop {
    0% { opacity: 0; transform: scale(0.9); }
    100% { opacity: 1; transform: scale(1); }
}

.stamp {
    display: inline-block;
    border: 3px solid;
    border-radius: 8px;
    padding: 4px 14px;
    font-family: 'Special Elite', monospace;
    font-weight: bold;
    transform: rotate(-6deg);
    animation: stampIn 0.4s ease;
    font-size: 1.1rem;
}
@keyframes stampIn {
    0% { transform: rotate(-6deg) scale(2); opacity: 0; }
    100% { transform: rotate(-6deg) scale(1); opacity: 1; }
}
</style>

<div class="floating-emoji" style="left:5%; animation-delay:0s;">📰</div>
<div class="floating-emoji" style="left:20%; animation-delay:3s;">🗞️</div>
<div class="floating-emoji" style="left:40%; animation-delay:6s;">📰</div>
<div class="floating-emoji" style="left:65%; animation-delay:2s;">🗞️</div>
<div class="floating-emoji" style="left:85%; animation-delay:5s;">📰</div>

<div class="ticker-wrap">
  <div class="ticker">🚨 BREAKING &nbsp;•&nbsp; AI-POWERED FACT CHECK &nbsp;•&nbsp; MULTILINGUAL DETECTION &nbsp;•&nbsp; NO API KEY REQUIRED &nbsp;•&nbsp; ANALYZE ANY REGIONAL POST NOW &nbsp;•&nbsp; </div>
</div>

<div class="newspaper-title">📰 THE TRUTH GAZETTE</div>
<div class="subtitle-type">Fake News & Misinformation Detector — Multilingual Transformer + Live Fact Retrieval</div>
""", unsafe_allow_html=True)

# ============================================================
# MODEL LOADING (cached)
# ============================================================
@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )

classifier = load_classifier()
STYLE_LABELS = ["credible news", "misinformation", "satire", "propaganda"]

# ============================================================
# WIKIPEDIA FACT-CHECK LAYER (keyless, public API)
# ============================================================
def extract_key_phrases(text):
    """Grab capitalized word sequences and standalone numbers/years as candidate entities."""
    phrases = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', text)
    phrases = [p for p in phrases if len(p) > 2]
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    seen, unique = set(), []
    for p in phrases:
        if p.lower() not in seen:
            seen.add(p.lower())
            unique.append(p)
    return unique[:4]  # limit to avoid too many API calls

def wikipedia_lookup(query):
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": 1
        }
        r = requests.get(search_url, params=params, timeout=6)
        results = r.json().get("query", {}).get("search", [])
        if not results:
            return None
        title = results[0]["title"]
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
        s = requests.get(summary_url, timeout=6)
        if s.status_code == 200:
            data = s.json()
            return {"title": data.get("title"), "extract": data.get("extract", "")}
        return None
    except Exception:
        return None

# ============================================================
# UI
# ============================================================
text_input = st.text_area(
    "📝 Paste a social media post, headline, or news snippet:",
    height=160,
    placeholder="e.g. 'Breaking: Government announces free electricity for all farmers starting tomorrow...'"
)

col1, col2 = st.columns(2)
with col1:
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)
with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if analyze_btn:
    if not text_input.strip():
        st.warning("Please paste some text first.")
    else:
        # ---- Style analysis ----
        with st.spinner("🕵️ Analyzing tone and style..."):
            result = classifier(text_input, candidate_labels=STYLE_LABELS, multi_label=False)

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        st.markdown("## 🗂️ Style Analysis")
        color = "#1e5631" if top_label == "credible news" else "#7a1f1f"
        st.markdown(f"""
        <div class="result-card" style="background:{color};">
            <span class="stamp" style="color:white; border-color:white;">{top_label.upper()}</span>
            <p style="color:#eee; margin-top:10px;">Confidence: {top_score*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        for label, score in zip(result["labels"], result["scores"]):
            st.write(f"**{label.title()}**")
            st.progress(float(score))

        st.caption("⚠️ Style analysis judges *how the text sounds* — it cannot verify facts on its own. See fact-check below.")

        # ---- Fact-check layer ----
        st.markdown("## 🔎 Fact-Check Layer (Wikipedia)")
        phrases = extract_key_phrases(text_input)

        if not phrases:
            st.info("No clear named entities detected to fact-check. Style analysis above is your best signal.")
        else:
            any_found = False
            for phrase in phrases:
                with st.spinner(f"Checking '{phrase}'..."):
                    info = wikipedia_lookup(phrase)
                time.sleep(0.2)
                if info:
                    any_found = True
                    st.markdown(f"""
                    <div class="result-card" style="background:#2b2b3d;">
                        <b>📖 {info['title']}</b>
                        <p style="color:#ddd;">{info['extract'][:400]}{'...' if len(info['extract'])>400 else ''}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card" style="background:#3d2b2b;">
                        <b>❓ No Wikipedia article found for "{phrase}"</b>
                        <p style="color:#ddd;">Could not verify — treat with caution.</p>
                    </div>
                    """, unsafe_allow_html=True)

            if any_found:
                st.info("👆 Compare the claim in your text against these summaries manually — mismatches are a strong signal of misinformation.")

        st.markdown("---")
        st.caption(
            "⚠️ This tool combines style detection + factual retrieval, but is NOT a certified fact-checker. "
            "Always verify with trusted news sources before sharing."
        )

st.markdown("---")
st.caption("🗞️ The Truth Gazette — Built with 🤗 Transformers (mDeBERTa-v3) + Wikipedia API + Streamlit. No API key required.")
