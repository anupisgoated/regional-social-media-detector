import streamlit as st
from transformers import pipeline


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰"
)


st.title("📰 Fake News Detection using Transformer")
st.write("Enter a news statement to classify it.")


@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-fake-news-detection",
        return_all_scores=True
    )


classifier = load_model()


text = st.text_area(
    "News text:",
    height=150
)


if st.button("Analyze"):

    if text.strip():

        result = classifier(text)[0]

        # Find highest probability class
        best = max(
            result,
            key=lambda x: x["score"]
        )

        label = best["label"]
        confidence = best["score"] * 100


        # Convert labels
        if label in ["LABEL_1", "1"]:
            output = "Fake News ❌"
        elif label in ["LABEL_0", "0"]:
            output = "Real News ✅"
        else:
            output = label


        st.subheader("Prediction")

        if "Fake" in output:
            st.error(output)
        else:
            st.success(output)


        st.write(
            f"Confidence: {confidence:.2f}%"
        )

    else:
        st.warning("Enter some text first.")
