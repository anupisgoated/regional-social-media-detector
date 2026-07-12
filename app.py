import os
import streamlit as st
import torch
import google.generativeai as genai
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    <p style='margin:5px 0 0 0; color:#9ca3af; font-size:12px;'>Transformer Architecture & Factual Verification Pipeline</p>
</div>
""", unsafe_allow_html=True)

# Cached Transformer Pipeline Model Weights Loader
@st.cache_resource
def load_xlm_roberta_pipeline():
    repo_id = "himel05/fake-news-roberta"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    return tokenizer, model

try:
    tokenizer, transformer_model = load_xlm_roberta_pipeline()
except Exception as err:
    st.error(f"Framework core drop: {err}")
    st.stop()

# Input Container text component
social_media_payload = st.text_area("Paste regional social media post or chat logs:", placeholder="Type text here...", height=140)
execution_trigger = st.button("RUN ENSEMBLE SYSTEM TEST")

# Processing pipeline evaluation
if execution_trigger:
    if not social_media_payload.strip():
        st.error("Input container text matrix cannot be empty.")
    else:
        with st.spinner("Analyzing data vectors..."):
            # Step 1: Text Transformer Calculation
            tokenized_inputs = tokenizer(social_media_payload, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                tensor_logits = transformer_model(**tokenized_inputs).logits
                softmax_probabilities = torch.softmax(tensor_logits, dim=1).flatten().tolist()
            
            # FIXED: Label 1 is Fake, Label 0 is Real for himel05/fake-news-roberta
            base_transformer_fake_score = softmax_probabilities[1]
            
            # Step 2: Gemini API Integration Validation Layer
            api_key_check = os.environ.get("GEMINI_API_KEY")
            gemini_fake_score = 0.5
            gemini_reasoning = "External fact check context engine running on local fallback defaults."
            
            if api_key_check:
                try:
                    genai.configure(api_key=api_key_check)
                    query_agent = genai.GenerativeModel("gemini-1.5-flash")
                    prompt = f"Analyze this regional text: \"{social_media_payload}\". Reply ONLY in this format: PROBABILITY_FALSE: <0.0 to 1.0> | REASONING: <one clear sentence>"
                    agent_res = query_agent.generate_content(prompt).text.strip()
                    if "PROBABILITY_FALSE:" in agent_res and " | REASONING:" in agent_res:
                        p_part, r_part = agent_res.split(" | REASONING:", 1)
                        gemini_fake_score = float(p_part.replace("PROBABILITY_FALSE:", "").strip())
                        gemini_reasoning = r_part.strip()
                except:
                    pass
            
            # Step 3: Combined Weighted Average Resolution Formula
            final_fake_weight = (base_transformer_fake_score * 0.40) + (gemini_fake_score * 0.60)
            final_real_weight = 1.0 - final_fake_weight
            
            # Visual Probability Charts Component mapping
            fig = go.Figure(data=[
                go.Bar(name='Authentic', x=['Analysis'], y=[final_real_weight*100], marker_color='#10b981'),
                go.Bar(name='Misinformation', x=['Analysis'], y=[final_fake_weight*100], marker_color='#ef4444')
            ])
            fig.update_layout(barmode='stack', height=180, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#9ca3af', margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Dashboard status text presentation verdict cards block mapping
            if final_fake_weight > final_real_weight:
                st.markdown(f'<div class="verdict-box verdict-fake"><h4 style="color:#ef4444;margin:0;">🚨 MISINFORMATION FLAG</h4><p style="font-size:13px;margin:5px 0;">{gemini_reasoning}</p><strong>Index: {final_fake_weight*100:.1f}% Inaccurate</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-box verdict-real"><h4 style="color:#10b981;margin:0;">✅ VERIFIED ACCURATE</h4><p style="font-size:13px;margin:5px 0;">{gemini_reasoning}</p><strong>Index: {final_real_weight*100:.1f}% Genuine</strong></div>', unsafe_allow_html=True)
                
