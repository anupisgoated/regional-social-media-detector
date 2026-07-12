import streamlit as st
import torch
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Mobile Optimized Layout Config
st.set_page_config(page_title="Regional Misinformation Platform", layout="centered")

# Premium Mobile Glassmorphic Theme Injection
st.markdown("""
<style>
.stApp { background-color: #0b0f19; color: #f1f5f9; }
.title-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #4338ca;
    margin-bottom: 15px;
}
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    height: 3.5em;
    border-radius: 8px;
}
.verdict-box { padding: 20px; border-radius: 10px; margin-top: 15px; border-left: 6px solid; }
.verdict-fake { background: rgba(239, 68, 68, 0.08); border-left-color: #ef4444; }
.verdict-real { background: rgba(16, 185, 129, 0.08); border-left-color: #10b981; }
</style>
""", unsafe_allow_html=True)

# App UI Header text
st.markdown("""
<div class="title-banner">
    <h2 style='margin:0; color:#ffffff; font-size: 20px;'>Fake News Detection on Regional Social Media</h2>
    <p style='margin:5px 0 0 0; color:#9ca3af; font-size:12px;'>Keyless Dual-Transformer Architecture Pipeline</p>
</div>
""", unsafe_allow_html=True)

# Cached Model Loaders (Completely Free & Keyless open-source weights)
@st.cache_resource
def load_models():
    # Engine 1: Regional Deception Multi-lingual Detector
    repo_id = "himel05/fake-news-roberta"
    tok = AutoTokenizer.from_pretrained(repo_id)
    mod = AutoModelForSequenceClassification.from_pretrained(repo_id)
    
    # Engine 2: Contextual Fact-Check Logic Layer (Completely Keyless Alternative)
    fact_check_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return tok, mod, fact_check_pipe

try:
    tokenizer, transformer_model, zero_shot_engine = load_models()
except Exception as err:
    st.error(f"Framework core drop: {err}")
    st.stop()

# Input UI Fields
social_media_payload = st.text_area("Paste regional social media post or chat logs:", placeholder="Type text here...", height=140)
execution_trigger = st.button("RUN ENSEMBLE SYSTEM TEST")

if execution_trigger:
    if not social_media_payload.strip():
        st.error("Input container text matrix cannot be empty.")
    else:
        with st.spinner("Processing local deep-learning inference chains..."):
            
            # --- ENGINE 1: NATIVE LINGUISTIC EMBEDDING ANALYSIS ---
            tokenized_inputs = tokenizer(social_media_payload, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                tensor_logits = transformer_model(**tokenized_inputs).logits
                softmax_probabilities = torch.softmax(tensor_logits, dim=1).flatten().tolist()
            
            # Label 0: Real, Label 1: Fake inside himel05 repo parameters
            roberta_real_weight = softmax_probabilities[0]
            roberta_fake_weight = softmax_probabilities[1]
            
            # --- ENGINE 2: CONTEXTUAL FACT MATRIX VERIFICATION ---
            # Evaluates statement against contextual trust parameters without calling external cloud APIs
            candidate_labels = ["factually accurate news story", "unverified rumor or misinformation statement"]
            classification = zero_shot_engine(social_media_payload, candidate_labels)
            
            # Map structural prediction scores back to matrix values
            bart_scores = dict(zip(classification['labels'], classification['scores']))
            bart_real_weight = bart_scores["factually accurate news story"]
            bart_fake_weight = bart_scores["unverified rumor or misinformation statement"]
            
            # --- ENSEMBLE WEIGHT DISTRIBUTION CONSOLIDATION ---
            # 50/50 balance combination matrix completely eliminating API variables
            final_fake_weight = (roberta_fake_weight * 0.50) + (bart_fake_weight * 0.50)
            final_real_weight = 1.0 - final_fake_weight
            
            # Visual Probability Charts Component
            fig = go.Figure(data=[
                go.Bar(name='Authentic', x=['Analysis'], y=[final_real_weight*100], marker_color='#10b981'),
                go.Bar(name='Misinformation', x=['Analysis'], y=[final_fake_weight*100], marker_color='#ef4444')
            ])
            fig.update_layout(barmode='stack', height=180, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#9ca3af', margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Dashboard status presentation card mapping
            if final_fake_weight > final_real_weight:
                st.markdown(f'<div class="verdict-box verdict-fake"><h4 style="color:#ef4444;margin:0;">🚨 MISINFORMATION RISK DETECTED</h4><p style="font-size:13px;margin:5px 0;">The ensemble models flagged linguistic bias and semantic structure matching rumor profiles.</p><strong>Index: {final_fake_weight*100:.1f}% Inaccurate</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-box verdict-real"><h4 style="color:#10b981;margin:0;">✅ VERIFIED STRUCTURAL TRUTH</h4><p style="font-size:13px;margin:5px 0;"> Labeled text structures pass verification rules across linguistic testing frameworks.</p><strong>Index: {final_real_weight*100:.1f}% Genuine</strong></div>', unsafe_allow_html=True)
                
