"""
Streamlit UI for HealthcareBot with Qdrant backend
"""

import streamlit as st
import os
from typing import Dict, Any
import google.generativeai as genai

# Import Qdrant utilities and bot
from utils_qdrant import load_qdrant_and_model
from run_bot_qdrant import HealthcareBot

# Page config
st.set_page_config(
    page_title="Clinical Trials Search Assistant",
    page_icon="🏥",
    layout="wide"
)

# Title + description (updated list of diseases)
SUPPORTED_DISEASES = [
    "Diabetes", "Obesity", "Hypertension", "Cardiovascular disease",
    "Chronic Kidney Disease", "Alzheimer’s disease", "Parkinson’s disease",
    "Asthma", "COPD", "Breast cancer", "Lung cancer", "Prostate cancer",
    "Rheumatoid Arthritis"
]

st.title("🏥 Clinical Trials Search Assistant")
st.markdown("**Powered by Qdrant + Gemini 2.0 Flash**")
st.markdown(
    f"Search across 260,000+ clinical trials for:\n"
    f"- {', '.join(SUPPORTED_DISEASES)}"
)

# Sidebar for API keys + config
with st.sidebar:
    st.header("⚙️ Configuration")

    # Read from environment (set in Cloud Run)
    gemini_key = os.getenv("GEMINI_API_KEY")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    qdrant_url = os.getenv(
        "QDRANT_URL",
        "https://215ec69e-fa22-4f38-bcf3-941e73901a68.us-east4-0.gcp.cloud.qdrant.io",
    )

    st.divider()
    st.markdown("### 📊 System Status")
    if gemini_key and qdrant_key:
        st.success("✓ Backend API keys detected")
    else:
        st.error("❌ Missing backend API keys – set them in Cloud Run.")





# Initialize state
for key in ["messages", "bot", "qdrant_client", "embed_model"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "bot" else []

# Initialize bot
if gemini_key and qdrant_key and st.session_state.bot is None:
    with st.spinner("Initializing system..."):
        try:
            os.environ["GEMINI_API_KEY"] = gemini_key
            genai.configure(api_key=gemini_key)
            gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

            qdrant_client, embed_model = load_qdrant_and_model(qdrant_url, qdrant_key)

            st.session_state.qdrant_client = qdrant_client
            st.session_state.embed_model = embed_model
            st.session_state.bot = HealthcareBot(
                qdrant_client,
                embed_model,
                gemini_model
            )
            st.success("System ready!")
        except Exception as e:
            st.error(f"Initialization failed: {e}")

# Display chat history
if st.session_state.messages:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and "metadata" in m:
                with st.expander("📊 Details"):
                    col1, col2 = st.columns(2)
                    col1.metric("Trials Found", m["metadata"]["num_trials"])
                    col2.metric("Confidence", f"{m['metadata']['avg_confidence']:.0%}")

# Input bar
if prompt := st.chat_input("Ask about clinical trials..."):
    if st.session_state.bot is None:
        st.error("Backend not initialized – check API keys in Cloud Run and reload the page")
    else:
        # Store and show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process 🔍
        with st.chat_message("assistant"):
            with st.spinner("Searching clinical trials..."):
                result = st.session_state.bot.chat(prompt)

                response = result["response"]
                st.markdown(response)

                metadata = {
                    "num_trials": result["num_trials"],
                    "avg_confidence": result["avg_confidence"]
                }

                with st.expander("📊 Details"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Trials Found", metadata["num_trials"])
                    col2.metric("Confidence", f"{metadata['avg_confidence']:.0%}")
                    col3.metric("Session Hash", result["session_hash"][:8])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "metadata": metadata
                })

# Sidebar Example Queries — fixed to trigger the bot
with st.sidebar:
    st.divider()
    st.markdown("### 💡 Example Queries")

    examples = [
        "Trials for insulin therapy in diabetes",
        "Breast cancer immunotherapy trials",
        "Alzheimer’s disease new medication studies",
        "COPD clinical trials recruiting now",
        "Prostate cancer hormone therapy trials",
        "Rheumatoid arthritis biologic trials",
        "Obesity and GLP-1 weight loss trials",
        "Heart failure new treatment studies",
        "Parkinson’s disease clinical studies"
    ]

    for example in examples:
        if st.button(example):
            if st.session_state.bot:
                # Act like user input was typed
                st.session_state.messages.append({"role": "user", "content": example})
                result = st.session_state.bot.chat(example)

                metadata = {
                    "num_trials": result["num_trials"],
                    "avg_confidence": result["avg_confidence"]
                }

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "metadata": metadata
                })

                st.rerun()

# Footer
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
🔬 Powered by Qdrant Vector Database + Gemini 2.0 Flash<br>
📊 Searching clinical trials across 13 disease areas
</div>
""", unsafe_allow_html=True)
