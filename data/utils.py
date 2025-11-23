import json
import numpy as np
import faiss
from datetime import datetime
import hashlib
from sentence_transformers import SentenceTransformer

# --- Core Utility Functions ---

def calculate_confidence_score(distance, normalization_factor=1.0):
    """Calculates an inverse L2 distance score (closer to 1.0 is better)."""
    # Simple inverse distance score: 1 / (1 + distance)
    return normalization_factor / (normalization_factor + distance)

def load_data_and_index(chunk_map_path, faiss_path):
    """Loads pre-built chunks and FAISS index for quick startup."""
    print("⏳ Loading pre-built RAG index...")
    
    # Load chunk map (the human-readable documents/metadata)
    # Ensure this file is loaded correctly from your Drive path
    with open(chunk_map_path, 'r') as f:
        chunk_map = json.load(f)

    # Load FAISS index
    index = faiss.read_index(faiss_path)
    
    # Load embedding model (must match the model used for indexing)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2') 

    print(f"✅ RAG Index Ready: {index.ntotal} vectors loaded.")
    return embed_model, index, chunk_map

# --- Novelty: Provenance Logging ---

def log_provenance_step(agent_name, input_data, output_data, detail=None):
    """
    Creates a detailed log entry for a single agent step.
    This fulfills the 'Agent-level provenance chaining' novelty point.
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'agent': agent_name,
        'input': input_data,
        'output': output_data,
        'detail': detail or {},
        'model_version': 'gemini-2.0-flash' # Default model, should be dynamically retrieved
    }
    return log_entry

def generate_reproducibility_hash(conversation_history, corpus_version='v1.0'):
    """
    Generates a deterministic session hash based on the entire conversation history.
    This fulfills the 'Reproducibility hash' novelty point.
    """
    # Include corpus version for global reproducibility
    raw = f"{corpus_version}|{'|'.join([t['query'] for t in conversation_history])}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()