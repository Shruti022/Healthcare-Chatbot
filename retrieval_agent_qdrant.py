from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient

from utils_qdrant import (
    convert_similarity_to_confidence,
    log_provenance_step,
)

# ============================================================
# Disease Config for Label + Synonyms
# ============================================================

CONDITION_CONFIG: Dict[str, str] = {
    "diabetes": "Diabetes",
    "obesity": "Obesity",
    "hypertension": "Hypertension",
    "cardiovascular": "Cardiovascular Disease",
    "ckd": "Chronic Kidney Disease",
    "alzheimer": "Alzheimer’s Disease",
    "parkinson": "Parkinson’s Disease",
    "asthma": "Asthma",
    "copd": "Chronic Obstructive Pulmonary Disease",
    "breast_cancer": "Breast Cancer",
    "lung_cancer": "Lung Cancer",
    "prostate_cancer": "Prostate Cancer",
    "stroke": "Stroke",
    "rheumatoid_arthritis": "Rheumatoid Arthritis",
}

DISEASE_SYNONYMS: Dict[str, List[str]] = {
    "diabetes": ["diabetes", "t2d", "t1d", "blood sugar", "insulin", "glucose"],
    "obesity": ["obesity", "overweight", "weight loss"],
    "hypertension": ["hypertension", "high blood pressure"],
    "cardiovascular": ["cardiovascular", "heart failure", "coronary", "myocardial", "angina", "cad"],
    "ckd": ["kidney disease", "renal", "ckd"],
    "alzheimer": ["alzheimer", "dementia", "memory loss"],
    "parkinson": ["parkinson"],
    "asthma": ["asthma", "wheezing", "inhaler"],
    "copd": ["copd", "chronic obstructive", "emphysema"],
    "breast_cancer": ["breast cancer", "her2"],
    "lung_cancer": ["lung cancer", "nsclc"],
    "prostate_cancer": ["prostate cancer"],
    "stroke": ["stroke", "cva", "cerebrovascular"],
    "rheumatoid_arthritis": ["rheumatoid arthritis", "ra"],
}


# ============================================================
# Qdrant Retrieval Agent
# ============================================================

class QdrantRetrievalAgent:
    """Vector similarity + disease-aware scoring against Qdrant database"""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embed_model,
        collection_name: str = "clinical_trials",
    ):
        self.client = qdrant_client
        self.embed_model = embed_model
        self.collection_name = collection_name

    # --------------------------------------------------------
    # Detect disease from the query → soft filter
    # --------------------------------------------------------
    def detect_disease(self, query: str) -> Optional[Dict[str, str]]:
        q = query.lower()
        for key, syns in DISEASE_SYNONYMS.items():
            if any(s in q for s in syns):
                return {"key": key, "label": CONDITION_CONFIG.get(key, key)}
        return None

    # --------------------------------------------------------
    # Score disease match from payload ("disease" + "conditions")
    # --------------------------------------------------------
    def disease_match_score(self, disease_hint, payload):
        if not disease_hint:
            return 0.0

        trial_disease = (payload.get("disease") or "").lower()
        trial_conds = (payload.get("conditions") or "").lower()

        key = disease_hint["key"]
        label = disease_hint["label"].lower()

        # Strong match: exact disease field match
        if label in trial_disease:
            return 1.0

        # Secondary match: synonyms appear
        for s in DISEASE_SYNONYMS.get(key, []):
            if s in trial_conds:
                return 0.6

        return 0.0

    # --------------------------------------------------------
    # Main retrieve function
    # --------------------------------------------------------
    def retrieve(self, parsed: Dict[str, Any], top_k=5, candidate_k=30):
        query = parsed.get("query") or parsed.get("user_question") or ""
        query = query.strip()

        if not query:
            empty = {"query": "", "trials": [], "avg_confidence": 0.0}
            return empty, log_provenance_step("Qdrant", parsed, empty)

        disease_hint = self.detect_disease(query)
        q_vec = self.embed_model.encode([query])[0]

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec.tolist(),
            limit=candidate_k,
            with_payload=True
        )

        trials = []
        confidences = []

        for point in result.points:
            pl = point.payload or {}

            sim = float(point.score)
            confidence = convert_similarity_to_confidence(sim)
            confidences.append(confidence)

            # Disease scoring
            disease_w = self.disease_match_score(disease_hint, pl)

            # Simple status weighting
            status = str(pl.get("status", "Unknown")).title()
            status_w = 1.0 if status == "Completed" else 0.8

            # Final relevance scoring
            relevance = (
                0.65 * sim +      # Semantic similarity
                0.25 * disease_w +  # Correct disease
                0.10 * status_w   # Study phase quality indicator
            )

            trials.append({
                "nct_id": pl.get("nct_id"),
                "title": pl.get("title"),
                "text": pl.get("text"),
                "status": status,
                "confidence": confidence,
                "relevance": relevance,
            })

        # Sort + rank
        trials.sort(key=lambda x: x["relevance"], reverse=True)
        top = trials[:top_k]

        for i, t in enumerate(top, 1):
            t["rank"] = i

        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

        out = {
            "query": query,
            "disease_hint": disease_hint,
            "trials": top,
            "avg_confidence": avg_conf,
        }

        log = log_provenance_step("QdrantRetrievalAgent", parsed, out)
        return out, log
