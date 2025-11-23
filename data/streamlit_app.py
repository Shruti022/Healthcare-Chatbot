import streamlit as st
import json
import os
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer

# --- CRITICAL: Import the modular components ---
# IMPORTANT: In a Streamlit environment, ensure utils.py, agents.py, and 
# orchestrator.py are accessible via standard Python imports.
# Assuming they are in the same directory for local testing.
# In Colab, you must manually run the .py files or ensure they are properly
# imported before Streamlit runs (see notebook instructions).

from utils import load_data_and_index
from orchestrator import HealthcareBot

# --- 1. CONFIGURATION AND INITIALIZATION (Runs once per session) ---

def initialize_chatbot():
    """Initializes all agents, loads the RAG index, and stores the bot in session state."""
    
    # 1. API Key Setup
    # Use st.secrets or environment variable for secure key management
    # For Colab/Local testing, assume key is set externally or hardcoded temporarily
    if "GEMINI_API_KEY" not in os.environ:
        st.error("GEMINI_API_KEY environment variable not set.")
        # Fallback for demonstration, replace with your actual key if needed
        API_KEY = "**************" 
        genai.configure(api_key=API_KEY)
    else:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    # 2. Data Paths (Update these paths if necessary for your deployment)
    CHUNK_MAP_PATH = 'clinical_trials_diabetes_full_chunk_map.json' 
    FAISS_INDEX_PATH = 'clinical_trials_diabetes_full_faiss.index' 

    if not os.path.exists(CHUNK_MAP_PATH) or not os.path.exists(FAISS_INDEX_PATH):
         st.warning("⚠️ RAG data files not found locally. Please ensure 'clinical_trials_diabetes_full_chunk_map.json' and 'clinical_trials_diabetes_full_faiss.index' are accessible.")
         st.stop()
         
    # 3. Load RAG Components
    try:
        embed_model, faiss_index, chunk_map = load_data_and_index(CHUNK_MAP_PATH, FAISS_INDEX_PATH)
    except Exception as e:
        st.error(f"Failed to load RAG index: {e}. Check file paths and content.")
        st.stop()
        
    # 4. Initialize Core Model and Profile
    gemini_model = genai.GenerativeModel('models/gemini-2.0-flash')

    initial_profile = {
        'user_id': 'Alice',
        'age': 55, 
        'conditions': ['Type 2 Diabetes', 'High Cholesterol'],
        'medications': ['Statin']
    }

    # 5. Create the Orchestrator Bot
    bot = HealthcareBot(
        gemini_model=gemini_model,
        embed_model=embed_model,
        faiss_index=faiss_index,
        chunk_map=chunk_map,
        initial_profile=initial_profile
    )
    
    # Store the bot and chat history in Streamlit's session state
    st.session_state.bot = bot
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add initial welcome message from the bot
        st.session_state.messages.append({"role": "assistant", "content": f"Hello {initial_profile['user_id']}! I am your AI Health Assistant. How can I help you today?"})
        
    st.sidebar.success("System Ready. Multi-Agent Pipeline Initialized.")


# --- 2. MAIN STREAMLIT APP LAYOUT ---

st.set_page_config(layout="wide", page_title="Master's Healthcare RAG Chatbot")
st.title("🩺 Personalized Healthcare Assistant (RAG Agent)")

# Initialize the system if it hasn't been done yet
if "bot" not in st.session_state:
    initialize_chatbot()

bot = st.session_state.bot

# --- Sidebar for Profile and Provenance Control ---
with st.sidebar:
    st.header("👤 User Profile & Memory")
    st.json(bot.profile_agent.profile)
    
    st.header("🛠️ Provenance & Debug")
    if bot.history:
        latest_hash = bot.history[-1].get('response_hash', 'N/A')
        st.code(f"Latest Hash: {latest_hash}", language='markdown')

# --- Main Chat Interface ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and bot response
if prompt := st.chat_input("Ask about diabetes, weight loss, or clinical trials..."):
    # 1. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Process the query via the Multi-Agent Pipeline
    with st.spinner('Thinking... Running Multi-Agent Pipeline...'):
        result = bot.process_query(prompt)
    
    # 3. Generate the response for the user
    with st.chat_message("assistant"):
        st.markdown(result['recommendation'])
        
        # 4. Display Provenance/Metadata in an Expander (Crucial for 7000-Level Project)
        with st.expander("🔬 Agent-Level Provenance Chain (Traceability)"):
            st.markdown(f"**Safety Status:** `{result['safety_status']}` | **Cited Trials:** `{result['cited_trials']}`")
            st.markdown(f"**Reproducibility Hash:** `{result['session_hash']}`")
            st.markdown("---")
            
            # Display the Provenance Chain
            for step in result['provenance_chain']:
                with st.container(border=True):
                    st.caption(f"**{step['agent']}** @ {step['timestamp'].split('T')[1].split('.')[0]}", )
                    
                    if step['agent'] == 'RetrievalAgent':
                        st.code(step['output'].get('query', 'N/A'), language='markdown')
                        st.markdown(f"Retrieved {len(step['output'].get('trials', []))} trials. Best Score: `{step['output'].get('trials', [{}])[0].get('retrieval_score', 'N/A'):.2f}`")
                    elif step['agent'] == 'DiagnosisAdvisor':
                        veto = step['output'].get('veto', False)
                        st.markdown(f"**Veto:** `{veto}`")
                    
                    # Log the full JSON output of the step (for full debugging)
                    st.json(step)

    # 5. Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result['recommendation']})