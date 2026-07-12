import streamlit as st
from transformers import pipeline

# Page setup
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News & Misinformation Detector")
st.write(
    "AI-based detection system using a Transformer NLP model. "
    "Enter a news statement to analyze it."
)

# Load model
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-fake-news-detection"
    )

classifier = load_model()

# Input box
news_text = st.text_area(
    "Enter a news article or social media post:",
    height=150,
    placeholder="Example: Scientists discovered a new planet..."
)

if st.button("Analyze News"):
    if news_text.strip():

        with st.spinner("Analyzing with Transformer model..."):
            result = classifier(news_text)[0]

        label = result["label"]
        confidence = result["score"] * 100

        if "REAL" in label.upper():
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.info(
            "Note: AI predictions are not perfect. "
            "Always verify important information using trusted sources."
        )

    else:
        st.warning("Please enter some text first.")
