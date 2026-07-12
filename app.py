import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ---------- Load model (cached so it only loads once per session) ----------
@st.cache_resource
def load_classifier():
    # Multilingual model — works across many regional languages (Hindi, Bengali,
    # Tamil, Spanish, Arabic, etc.), no API key required, fully local inference.
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )

classifier = load_classifier()

LABELS = ["credible news", "misinformation", "satire", "propaganda"]

# ---------- UI ----------
st.title("📰 Fake News & Misinformation Detector")
st.caption("Multilingual transformer model — works on regional social media text, news snippets, or posts.")

text_input = st.text_area(
    "Paste a social media post, headline, or news snippet:",
    height=180,
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
        with st.spinner("Analyzing content..."):
            result = classifier(text_input, candidate_labels=LABELS, multi_label=False)

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        st.subheader("Result")
        if top_label == "credible news":
            st.success(f"✅ Likely **{top_label.upper()}** ({top_score*100:.1f}% confidence)")
        else:
            st.error(f"⚠️ Likely **{top_label.upper()}** ({top_score*100:.1f}% confidence)")

        st.markdown("### Confidence Breakdown")
        for label, score in zip(result["labels"], result["scores"]):
            st.write(f"**{label.title()}**")
            st.progress(float(score))

        st.markdown("---")
        st.caption(
            "⚠️ This tool gives a probabilistic estimate based on language patterns, "
            "not a verified fact-check. Always cross-check with trusted news sources "
            "or fact-checking organizations before sharing."
        )

st.markdown("---")
st.caption("Built with 🤗 Transformers (mDeBERTa-v3, multilingual) + Streamlit — no API key required.")
