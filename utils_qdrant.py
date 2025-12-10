import json
import hashlib
from datetime import datetime
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "clinical_trials"

SUPPORTED_DISEASES = [
    "diabetes", "obesity", "hypertension", "cardiovascular", "ckd",
    "alzheimer", "parkinson", "asthma", "copd",
    "breast_cancer", "lung_cancer", "prostate_cancer",
    "rheumatoid_arthritis"
]

# --- Confidence score from cosine similarity ---
# score returned by Qdrant: higher is better, max = 1.0+

def convert_similarity_to_confidence(score: float) -> float:
    """Map cosine similarity into readable clinical confidence (0-1 scale)."""
    if score is None:
        return 0.0
    # Clip values to avoid >1.0 float drift
    return max(0.0, min(1.0, score))


# --- Load Qdrant client + embedding model (updated) ---

def load_qdrant_and_model(qdrant_url: str, qdrant_api_key: str):
    """Loads Qdrant client + embedding model and verifies dataset."""

    print("⏳ Connecting to Qdrant...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    try:
        info = client.get_collection(COLLECTION_NAME)
        print(f"🔗 Qdrant Connected → {info.points_count:,} trials indexed")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to access Qdrant collection: {e}")

    # Verify disease field exists in payload
    try:
        test_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=[0.0] * 384,
            with_payload=True,
            limit=1
        )
        sample_payload = test_results.points[0].payload

        if "disease" not in sample_payload:
            print("⚠️ WARNING: Disease field missing in payload — filtering may degrade")
        else:
            print("🧠 Disease-based retrieval enabled")

    except:
        print("⚠️ Could not verify payload fields")

    # Load matching embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("🧬 Embedding model initialized")

    print(f"📌 Supported diseases: {SUPPORTED_DISEASES}")

    return client, embed_model


# --- Provenance logging (unchanged) ---

def log_provenance_step(agent_name: str, input_data, output_data, detail=None):
    """Creates a structured log entry for multi-agent tracking"""
    return {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "input": input_data,
        "output": output_data,
        "detail": detail or {},
        "model_version": "gemini-2.0-flash",  # Keep consistent with production runtime
    }


# --- Reproducibility hash (unchanged) ---

def generate_reproducibility_hash(conversation_history, corpus_version: str = "v1.0"):
    """Stable, deterministic ID for rerunning full system evaluations"""
    queries = [turn.get("query", "") for turn in conversation_history]
    raw = f"{corpus_version}|{'|'.join(queries)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()
