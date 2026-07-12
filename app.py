import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰"
)

st.title("📰 Fake News & Misinformation Detector")
st.write("Transformer-based fake news classification system")


@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-fake-news-detection",
        return_all_scores=True
    )


classifier = load_model()


news = st.text_area(
    "Enter news text:",
    height=150
)


if st.button("🔍 Analyze"):

    if news.strip():

        with st.spinner("Analyzing..."):

            output = classifier(news)

            # Remove extra nesting
            scores = output[0]

            best = max(
                scores,
                key=lambda x: x["score"]
            )

            label = best["label"]
            confidence = best["score"] * 100


            if label == "LABEL_1":
                prediction = "Fake News ❌"
            else:
                prediction = "Real News ✅"


        st.subheader("Result")

        if "Fake" in prediction:
            st.error(prediction)
        else:
            st.success(prediction)

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    else:
        st.warning("Please enter text.")
