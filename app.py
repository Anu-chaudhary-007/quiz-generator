import os
import json
import streamlit as st

from core.prompt import build_quiz_prompt
from providers import hf as hf_provider
from providers import gemini as gemini_provider

st.set_page_config(page_title="🧠 Quiz Generator (Step 1)", page_icon="🧠")

st.title("🧠 Quiz Generator — Step 1: Setup & Test")

provider = st.secrets.get("PROVIDER", os.getenv("PROVIDER", "hf")).lower()

col1, col2, col3 = st.columns(3)
with col1:
    topic = st.text_input("Topic", "Python basics")
with col2:
    difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=0)
with col3:
    n = st.number_input("Number of questions", min_value=1, max_value=20, value=3, step=1)

st.caption(f"Using provider: **{provider.upper()}**")

if st.button("Generate (smoke test)"):
    prompt = build_quiz_prompt(topic, difficulty, n)
    try:
        if provider == "hf":
            token = st.secrets.get("HUGGINGFACE_API_TOKEN", os.getenv("HUGGINGFACE_API_TOKEN"))
            model_id = st.secrets.get("HF_MODEL_ID", os.getenv("HF_MODEL_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1"))
            text = hf_provider.generate(prompt, token=token, model_id=model_id)
        elif provider == "gemini":
            api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
            model_id = st.secrets.get("GEMINI_MODEL_ID", os.getenv("GEMINI_MODEL_ID", "gemini-1.5-flash"))
            text = gemini_provider.generate(prompt, api_key=api_key, model_id=model_id)
        else:
            st.error("Unknown provider. Set PROVIDER to 'hf' or 'gemini' in secrets.")
            st.stop()

        # Try parsing JSON (models may sometimes add junk; we’ll harden later)
        try:
            data = json.loads(text)
            st.success("Received valid JSON ✅")
            st.json(data)
        except json.JSONDecodeError:
            st.warning("Model did not return perfect JSON (expected at this step). Showing raw text:")
            st.code(text)
    except Exception as e:
        st.error(f"Generation failed: {e}")

