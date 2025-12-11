"""
Updated HealthcareBot using Qdrant instead of FAISS
- Uses QdrantRetrievalAgent for vector search
- Shows rich greeting (G2) ONLY when no clear supported disease is detected
"""

import json
import re
from typing import List, Dict, Any
import numpy as np
import requests
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Import utilities
from utils_qdrant import (
    load_qdrant_and_model,
    log_provenance_step,
    generate_reproducibility_hash,
)

# Qdrant-based retrieval agent
from retrieval_agent_qdrant import QdrantRetrievalAgent

# Optional (not used directly here, kept for future reranking)
CrossEncoder = None
try:
    from sentence_transformers import CrossEncoder  # noqa: F401
except Exception:
    pass


# ============================================================
# PARSER
# ============================================================
class SymptomParser:
    def __init__(self, model):
        self.model = model

    def parse(self, text: str):
        """
        Enhanced parser for clinical trial search queries.
        Decides:
        - Are they searching for trials or just asking a question?
        - Which disease is implied (if any)?
        """
        prompt = (
            "You are a clinical trial search classifier for medical research.\n"
            "You support conditions including: diabetes, obesity, hypertension, "
            "cardiovascular disease, chronic kidney disease (CKD), Alzheimer's disease, "
            "Parkinson's disease, asthma, COPD, breast cancer, lung cancer, "
            "prostate cancer, stroke, and rheumatoid arthritis.\n\n"
            f"User Input: \"{text}\"\n\n"
            "Your tasks:\n"
            "1) Decide if the user is searching for clinical trials or just asking a general question.\n"
            "2) Detect which disease(s) they are talking about.\n"
            "3) Detect if the query is not about health or clinical trials (off_topic).\n\n"
            "Classification Rules:\n"
            "- If the query mentions or implies trials, studies, research, clinical experiments, etc. → intent='trial_search'\n"
            "- If the user is mainly describing themselves (age, diagnosis, comorbidities, meds) → intent='profile_info'\n"
            "- If they ask 'what is X', 'how does Y work', etc. without asking about trials → intent='general_question'\n"
            "- Simple greetings (hi, hello, hey) → intent='greeting'\n"
            "- If clearly not about health or clinical research → intent='off_topic', is_disease_related=false\n\n"
            "You must detect disease_focus whenever possible.\n\n"
            "Return ONLY valid JSON with this exact format:\n"
            "{\n"
            "  \"intent\": \"trial_search\" | \"profile_info\" | \"general_question\" | \"greeting\" | \"off_topic\",\n"
            "  \"query_type\": \"trial_query\" | \"profile_statement\" | \"knowledge_seeking\" | \"greeting\",\n"
            "  \"search_keywords\": [\"keyword1\", \"keyword2\"],\n"
            "  \"is_disease_related\": true or false,\n"
            "  \"disease_focus\": [\"diabetes\", \"obesity\", \"hypertension\", \"cardiovascular\", \"ckd\",\n"
            "                     \"alzheimer\", \"parkinson\", \"asthma\", \"copd\",\n"
            "                     \"breast_cancer\", \"lung_cancer\", \"prostate_cancer\",\n"
            "                     \"stroke\", \"rheumatoid_arthritis\"],\n"
            "  \"user_question\": \"the question in plain English\",\n"
            "  \"trial_interest\": \"what type of trial they want (diet, medication, technology, surgery, etc.)\"\n"
            "}\n"
        )

        try:
            res = self.model.generate_content(prompt)
            raw = (res.text or "").strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                parsed = json.loads(raw)
        except Exception:
            # Fallback: simple heuristic if model fails
            text_lower = text.lower()
            disease_focus = []

            def has_any(words):
                return any(w in text_lower for w in words)

            if has_any(["diabetes", "insulin", "glucose", "hba1c", "metformin", "glp-1", "sglt2"]):
                disease_focus.append("diabetes")
            if has_any(["obesity", "overweight", "weight loss"]):
                disease_focus.append("obesity")
            if has_any(["hypertension", "high blood pressure"]):
                disease_focus.append("hypertension")
            if has_any(["heart failure", "cardiovascular", "angina", "coronary", "myocardial", "stroke"]):
                disease_focus.append("cardiovascular")
            if has_any(["chronic kidney", "ckd", "renal failure", "kidney disease"]):
                disease_focus.append("ckd")
            if has_any(["alzheimer", "alzheimers", "dementia", "memory loss", "cognitive decline"]):
                disease_focus.append("alzheimer")
            if has_any(["parkinson", "parkinson's"]):
                disease_focus.append("parkinson")
            if has_any(["asthma", "wheezing"]):
                disease_focus.append("asthma")
            if has_any(["copd", "chronic obstructive", "emphysema", "chronic bronchitis"]):
                disease_focus.append("copd")
            if has_any(["breast cancer"]):
                disease_focus.append("breast_cancer")
            if has_any(["lung cancer", "nsclc", "small cell lung cancer"]):
                disease_focus.append("lung_cancer")
            if has_any(["prostate cancer"]):
                disease_focus.append("prostate_cancer")
            if has_any(["stroke", "cerebrovascular accident", "cva"]):
                disease_focus.append("stroke")
            if has_any(["rheumatoid arthritis", "ra", "inflammatory arthritis"]):
                disease_focus.append("rheumatoid_arthritis")

            if has_any(["trial", "study", "studies", "research", "clinical"]):
                intent = "trial_search"
                query_type = "trial_query"
            elif has_any(["hi", "hello", "hey"]):
                intent = "greeting"
                query_type = "greeting"
            else:
                intent = "general_question"
                query_type = "knowledge_seeking"

            parsed = {
                "intent": intent,
                "query_type": query_type,
                "search_keywords": [text] if intent == "trial_search" else [],
                "is_disease_related": bool(disease_focus),
                "disease_focus": disease_focus,
                "user_question": text,
                "trial_interest": "general",
            }

        # --- Heuristic correction layer on top of model output ---
        text_lower = text.lower()
        diseases = set(parsed.get("disease_focus") or [])

        def maybe_add(words, label):
            if any(w in text_lower for w in words):
                diseases.add(label)

        maybe_add(["diabetes", "insulin", "glucose", "hba1c", "metformin", "glp-1", "sglt2"], "diabetes")
        maybe_add(["obesity", "overweight", "weight loss"], "obesity")
        maybe_add(["hypertension", "high blood pressure"], "hypertension")
        maybe_add(["heart failure", "cardiovascular", "angina", "coronary", "myocardial", "stroke"], "cardiovascular")
        maybe_add(["chronic kidney", "ckd", "renal failure", "kidney disease"], "ckd")
        maybe_add(["alzheimer", "alzheimers", "dementia", "memory loss", "cognitive decline"], "alzheimer")
        maybe_add(["parkinson", "parkinson's"], "parkinson")
        maybe_add(["asthma", "wheezing", "inhaler"], "asthma")
        maybe_add(["copd", "chronic obstructive", "emphysema", "chronic bronchitis"], "copd")
        maybe_add(["breast cancer"], "breast_cancer")
        maybe_add(["lung cancer", "nsclc", "small cell lung cancer"], "lung_cancer")
        maybe_add(["prostate cancer"], "prostate_cancer")
        maybe_add(["stroke", "cerebrovascular accident", "cva"], "stroke")
        maybe_add(["rheumatoid arthritis", "inflammatory arthritis", "ra"], "rheumatoid_arthritis")

        parsed["disease_focus"] = list(diseases)

        # Force trial_search if obvious trial keywords
        trial_keywords = [
            "trial", "study", "studies", "research",
            "clinical", "show me", "are there", "what trials",
        ]
        if any(kw in text_lower for kw in trial_keywords):
            parsed["intent"] = "trial_search"
            parsed["query_type"] = "trial_query"

        # If we detected diseases, ensure is_disease_related = True
        if diseases and parsed.get("intent") != "off_topic":
            parsed["is_disease_related"] = True
        elif "is_disease_related" not in parsed:
            parsed["is_disease_related"] = bool(diseases)

        log = log_provenance_step("SymptomParser", text, parsed)
        return parsed, log


# ============================================================
# PROFILE AGENT
# ============================================================
class ProfileAgent:
    def __init__(self, initial_profile: Dict[str, Any] = None):
        if initial_profile is None:
            initial_profile = {
                "user_id": "Patient",
                "conditions": [],
                "extracted_conditions": [],
                "history": [],
            }
        self.profile = initial_profile

    def update_profile(self, turn_data: Dict[str, Any]):
        self.profile.setdefault("history", []).append(turn_data)
        self.profile.setdefault("extracted_conditions", [])

        parsed = turn_data.get("parsed", {})
        diseases = parsed.get("disease_focus") or []
        if diseases:
            current = set(self.profile["extracted_conditions"])
            for d in diseases:
                current.add(d)
            self.profile["extracted_conditions"] = list(current)

        snapshot = {
            "user_id": self.profile.get("user_id", "Patient"),
            "known_conditions": self.profile.get("extracted_conditions", []),
            "num_turns": len(self.profile["history"]),
        }
        log = log_provenance_step("ProfileAgent", turn_data, {"profile_snapshot": snapshot})
        return log


# ============================================================
# EVIDENCE-WEIGHTED SCORER (still used downstream)
# ============================================================
class EvidenceWeightedScorer:
    def __init__(self):
        self.status_weights = {
            "Completed": 1.0,
            "Active, Not Recruiting": 0.9,
            "Recruiting": 0.85,
            "Enrolling By Invitation": 0.8,
            "Not Yet Recruiting": 0.6,
            "Terminated": 0.4,
            "Withdrawn": 0.3,
            "Suspended": 0.3,
            "Unknown Status": 0.5,
        }

        self.design_keywords = {
            "randomized controlled": 1.0,
            "double-blind": 0.95,
            "randomized": 0.9,
            "controlled": 0.85,
            "interventional": 0.8,
            "prospective": 0.75,
            "observational": 0.6,
            "retrospective": 0.5,
        }

    def calculate_weighted_score(
        self,
        trial: Dict[str, Any],
        base_confidence: float,
        query: str,
    ) -> Dict[str, Any]:
        match_score = base_confidence * 0.35

        status = str(trial.get("status", "Unknown Status")).strip().title()
        status_score = self.status_weights.get(status, 0.5) * 0.25

        design_score = self._calculate_design_quality(trial) * 0.20
        keyword_score = self._calculate_keyword_density(trial, query) * 0.10
        completeness_score = self._calculate_completeness(trial) * 0.10

        weighted_score = (
            match_score +
            status_score +
            design_score +
            keyword_score +
            completeness_score
        )

        breakdown = {
            "base_confidence": base_confidence,
            "weighted_score": weighted_score,
            "factors": {
                "semantic_match": match_score,
                "trial_status": status_score,
                "study_design": design_score,
                "keyword_density": keyword_score,
                "completeness": completeness_score,
            },
        }

        return {
            "weighted_score": min(weighted_score, 1.0),
            "breakdown": breakdown,
        }

    def _calculate_design_quality(self, trial: Dict[str, Any]) -> float:
        text = f"{trial.get('title', '')} {trial.get('text', '')}".lower()
        max_score = 0.0
        for keyword, weight in self.design_keywords.items():
            if keyword in text:
                max_score = max(max_score, weight)
        return max_score if max_score > 0 else 0.6

    def _calculate_keyword_density(self, trial: Dict[str, Any], query: str) -> float:
        text = f"{trial.get('title', '')} {trial.get('text', '')}".lower()
        stopwords = {
            "the", "a", "an", "and", "or", "for", "with", "in", "on", "at", "to",
            "of", "is", "are", "what", "trials", "trial", "study", "studies", "clinical",
        }
        query_terms = [
            term for term in query.lower().split()
            if term not in stopwords and len(term) > 2
        ]
        if not query_terms:
            return 0.5
        matches = sum(1 for term in query_terms if term in text)
        density = matches / len(query_terms)
        return min(density, 1.0)

    def _calculate_completeness(self, trial: Dict[str, Any]) -> float:
        text = trial.get("text", "") or ""
        title = trial.get("title", "") or ""
        score = 0.0
        if len(title) > 10:
            score += 0.3
        if len(text) > 200:
            score += 0.7
        return min(score, 1.0)


# ============================================================
# PubMed Helper (NCT → PubMed abstract)
# ============================================================
def fetch_pubmed_abstract_for_nct(nct_id: str):
    try:
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": f"{nct_id}[si]",
            "retmode": "json",
            "retmax": 1,
        }
        r = requests.get(esearch_url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        idlist = data.get("esearchresult", {}).get("idlist", [])
        if not idlist:
            return None

        pmid = idlist[0]

        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "text",
        }
        r2 = requests.get(efetch_url, params=params, timeout=10)
        r2.raise_for_status()
        abstract_text = r2.text.strip()
        if not abstract_text:
            return None

        return {"pmid": pmid, "abstract": abstract_text}
    except Exception:
        return None


# ============================================================
# DIAGNOSIS / ADVISOR
# ============================================================
class DiagnosisAdvisor:
    def __init__(self, model):
        self.model = model

    def _handle_general_question(self, parsed: Dict[str, Any], retrieved: Dict[str, Any]):
        trials = retrieved.get("trials", [])
        user_question = parsed.get("user_question") or " ".join(parsed.get("symptoms", []))

        evidence_parts = []
        for t in trials[:3]:
            evidence_parts.append(f"Trial {t['nct_id']}: {t['text'][:400]}")
        evidence = "\n\n".join(evidence_parts) if evidence_parts else "No specific trials available."

        prompt = (
            "You are a medical research educator. Answer the user's question clearly using reliable medical knowledge.\n"
            "The clinical trial evidence below provides real-world context - mention it if helpful.\n\n"
            f"USER'S QUESTION: {user_question}\n\n"
            "CLINICAL TRIAL CONTEXT (for reference only):\n"
            f"{evidence}\n\n"
            "Instructions:\n"
            "- Answer the question directly in 3–5 sentences.\n"
            "- Be specific and educational.\n"
            "- Do NOT give diagnoses or treatment instructions.\n"
            "- End with: 'For personalized advice, please consult your healthcare provider.'\n"
        )

        try:
            res = self.model.generate_content(prompt)
            text = (res.text or "").strip()
            if not text or len(text) < 50:
                text = (
                    "I don't have enough information to answer this question accurately. "
                    "For personalized guidance, please consult your healthcare provider."
                )
            return text
        except Exception:
            return (
                "I'm unable to generate a detailed answer right now. "
                "For personalized guidance, please consult your healthcare provider."
            )

    def _handle_symptom_query(
        self,
        parsed: Dict[str, Any],
        retrieved: Dict[str, Any],
        profile: Dict[str, Any],
    ):
        trials = retrieved.get("trials", [])
        if not trials:
            return "No relevant trials were found. Please try refining your query."

        formatted_trials = []
        for t in trials[:5]:
            title = t.get("title", "") or t["text"].split("\n")[0].replace("Title: ", "")
            status = t.get("status", "Unknown")
            weighted_score = t.get("weighted_score", t.get("relevance", 0.0))

            raw_text = t.get("text", "")
            brief_summary = raw_text.split("Summary:", 1)[-1].strip() if "Summary:" in raw_text else raw_text

            if brief_summary:
                prompt = (
                    "Rewrite the following clinical trial description as a short, clear paragraph "
                    "about what the study is testing:\n\n"
                    f"{brief_summary}\n\n"
                    "Guidelines:\n"
                    "- Use 2–4 sentences.\n"
                    "- Plain English, minimal jargon.\n"
                    "- Include the purpose and the main type of participant.\n"
                )
                try:
                    res = self.model.generate_content(prompt)
                    brief_summary = res.text.strip() if res.text else brief_summary
                except Exception:
                    if len(brief_summary) > 600:
                        brief_summary = brief_summary[:600] + "..."
            else:
                brief_summary = "No summary available."

            pubmed_block = ""
            pub = fetch_pubmed_abstract_for_nct(t["nct_id"])
            if pub:
                abs_text = pub["abstract"]
                max_len = 2000
                if len(abs_text) > max_len:
                    abs_text = abs_text[:max_len] + "..."
                pubmed_block = (
                    f"  PubMed abstract (PMID {pub['pmid']}):\n"
                    f"  {abs_text}\n\n"
                    f"  PubMed link: https://pubmed.ncbi.nlm.nih.gov/{pub['pmid']}/\n\n"
                )

            formatted_trials.append(
                f"**{t['nct_id']}** (Relevance: {weighted_score:.0%})\n"
                f"• {title}\n"
                f"  Status: {status}\n\n"
                f"  {brief_summary}\n\n"
                f"{pubmed_block}"
            )

        trials_text = "\n\n".join(formatted_trials)
        num_trials = len(formatted_trials)

        response = (
            f"I found {num_trials} clinical trial{'s' if num_trials != 1 else ''} relevant to your request:\n\n"
            f"{trials_text}\n\n"
            "Summary: These trials explore potential treatments or management strategies for the condition you asked about. "
            "More details are available using the listed NCT IDs.\n\n"
            "To learn more or consider participation, visit clinicaltrials.gov and search by NCT ID. "
            "Always discuss clinical trial options with your healthcare provider."
        )

        return response

    def advise(self, parsed: Dict[str, Any], retrieved: Dict[str, Any], profile: Dict[str, Any]):
        trials = retrieved.get("trials", [])
        avg_conf = retrieved.get("avg_confidence", 0.0)
        query_type = parsed.get("query_type", "trial_query")
        is_disease_related = parsed.get("is_disease_related", True)

        draft = {
            "recommendation": "",
            "avg_confidence": avg_conf,
            "query_type": query_type,
        }

        if not is_disease_related:
            draft["recommendation"] = (
                "I’m specialized in clinical trials for medical conditions such as:\n"
                "- Diabetes\n"
                "- Obesity\n"
                "- Hypertension & cardiovascular disease\n"
                "- Chronic kidney disease (CKD)\n"
                "- Alzheimer’s & Parkinson’s disease\n"
                "- Asthma & COPD\n"
                "- Breast, lung, and prostate cancer\n"
                "- Stroke\n"
                "- Rheumatoid arthritis\n\n"
                "Your question does not appear to be about a health condition or clinical research. "
                "If you’d like, you can ask me about trials for one of these conditions."
            )
            draft["confidence_veto"] = True
            log = log_provenance_step(
                "DiagnosisAdvisor",
                parsed,
                draft,
                {"veto": True, "reason": "off_topic"},
            )
            return draft, log

        if not trials or avg_conf < 0.05:
            draft["recommendation"] = (
                "Based on the trials I retrieved, I don’t have strong enough evidence to answer this question directly. "
                "Please consult your healthcare provider for personalized advice."
            )
            draft["confidence_veto"] = True
            log = log_provenance_step(
                "DiagnosisAdvisor",
                parsed,
                draft,
                {"veto": True, "reason": "low_confidence"},
            )
            return draft, log

        if query_type == "knowledge_seeking":
            draft["recommendation"] = self._handle_general_question(parsed, retrieved)
        else:
            draft["recommendation"] = self._handle_symptom_query(parsed, retrieved, profile)

        draft["confidence_veto"] = False
        log = log_provenance_step("DiagnosisAdvisor", parsed, draft)
        return draft, log



# ============================================================
# SAFETY FILTER
# ============================================================

class ActiveSafetyFilter:
    def __init__(self, model):
        self.model = model
        self.safety_cfg = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def _looks_like_medical_advice(self, text: str) -> bool:
        t = text.lower()
        risky_phrases = [
            "you should stop",
            "stop taking",
            "change your dose",
            "change your medication",
            "start taking",
            "take this drug",
            "take this medicine",
            "this is the best treatment for you",
            "i recommend you",
            "i recommend that you",
            "instead of your current medicine",
            "instead of your current medication",
            "it is safe to stop",
            "you can discontinue",
            "you can safely stop",
            "your diagnosis is",
            "the diagnosis is",
            "you have ",
        ]
        return any(p in t for p in risky_phrases)

    def verify(self, advice_text: str, trials: List[Dict[str, Any]]):
        # 0) Always allow pure trial listings
        if any(
            marker in advice_text
            for marker in ["NCT", "clinical trial", "clinicaltrials.gov"]
        ):
            log = log_provenance_step(
                "ActiveSafetyFilter",
                {"advice": advice_text},
                {"final_text": advice_text, "status": "Pass (Trial Listing)"},
            )
            return advice_text, "Pass (Trial Listing)", log

        # 1) Hard keyword veto for obvious medical advice
        if self._looks_like_medical_advice(advice_text):
            safer = (
                "⚠️ I cannot provide specific medical instructions such as starting, "
                "stopping, or changing medications or doses. Please discuss treatment "
                "decisions with your healthcare provider. I can help you explore "
                "relevant clinical trials and explain their general purpose instead."
            )
            log = log_provenance_step(
                "ActiveSafetyFilter",
                {"advice": advice_text},
                {"final_text": safer, "status": "Revised (Keyword Safety Gate)"},
            )
            return safer, "Revised (Keyword Safety Gate)", log

        # 2) Use LLM audit as a secondary check
        evidence_text = "\n".join(t["text"][:500] for t in trials[:3])

        audit_prompt = (
            "You are a Medical Safety Officer reviewing AI-generated advice.\n\n"
            "ADVICE:\n"
            f"{advice_text}\n\n"
            "EVIDENCE FROM CLINICAL TRIALS (for context):\n"
            f"{evidence_text}\n\n"
            "Mark advice as UNSAFE if it does ANY of the following:\n"
            "- Suggests starting, stopping, or changing any medication or dose.\n"
            "- Provides a diagnosis (for example, 'you have X', 'this means you have Y').\n"
            "- Recommends a specific treatment, drug, or trial as best for this user.\n"
            "- Discourages consulting a healthcare professional.\n"
            "- Makes strong clinical claims that are not clearly supported by evidence.\n\n"
            "If the advice is acceptable, respond with exactly: SAFE\n"
            "If it is not acceptable, respond starting with: CORRECTED: <safer version>\n"
        )

        try:
            res = self.model.generate_content(
                audit_prompt, safety_settings=self.safety_cfg
            )
            txt = (res.text or "").strip()
            if txt.startswith("SAFE") or "SAFE" == txt:
                final_text = advice_text
                status = "Pass"
            else:
                final_text = f"⚠️ SAFETY REVISION:\n{txt}"
                status = "Revised"
        except Exception:
            if "NCT" in advice_text or "clinical trial" in advice_text.lower():
                final_text = advice_text
                status = "Pass (API Fallback)"
            else:
                final_text = (
                    "⚠️ Safety filter triggered. Please consult your healthcare provider "
                    "for personalized medical advice."
                )
                status = "Revised (API Error)"

        log = log_provenance_step(
            "ActiveSafetyFilter",
            {"advice": advice_text},
            {"final_text": final_text, "status": status},
        )
        return final_text, status, log


# ============================================================
# HEALTHCAREBOT - Updated to use Qdrant + G2 / B3 behavior
# ============================================================

class HealthcareBot:
    def __init__(self, qdrant_client, embed_model, gemini_model, initial_profile=None):
        self.parser = SymptomParser(gemini_model)
        self.profile_agent = ProfileAgent(initial_profile)
        self.evidence_scorer = EvidenceWeightedScorer()

        # Qdrant-based retrieval agent
        self.retrieval = QdrantRetrievalAgent(
            qdrant_client=qdrant_client,
            embed_model=embed_model,
            collection_name="clinical_trials",
        )

        self.advisor = DiagnosisAdvisor(gemini_model)
        self.safety_filter = ActiveSafetyFilter(gemini_model)
        self.conversation_history: List[Dict[str, Any]] = []
        self.provenance_log: List[Dict[str, Any]] = []

    # -------------- Greeting builder (G2 style) --------------
    def _build_greeting(self) -> str:
        return (
            "👋 Hi, I’m your clinical trial assistant.\n\n"
            "I can search **real clinical trials** for:\n"
            "- Diabetes (type 1 & type 2)\n"
            "- Obesity & weight management\n"
            "- Hypertension & cardiovascular disease\n"
            "- Chronic kidney disease (CKD)\n"
            "- Alzheimer’s disease\n"
            "- Parkinson’s disease\n"
            "- Asthma\n"
            "- COPD (chronic obstructive pulmonary disease)\n"
            "- Breast, lung, and prostate cancer\n"
            "- Stroke\n"
            "- Rheumatoid arthritis\n\n"
            "**Try asking:**\n"
            "• \"GLP-1 agonist trials for type 2 diabetes\"\n"
            "• \"Weight loss studies for obesity\"\n"
            "• \"Breast cancer immunotherapy clinical trials\"\n"
            "• \"New asthma biologic trials for adults\"\n"
            "• \"Recent rheumatoid arthritis treatment studies\"\n\n"
            "Tell me your condition and what kind of trial you’re interested in (medication, diet, devices, etc.), "
            "and I’ll surface the most relevant studies."
        )

    def chat(self, user_input: str) -> Dict[str, Any]:
        """Process user input through the pipeline."""

        # 1) Parse intent
        parsed, parse_log = self.parser.parse(user_input)
        self.provenance_log.append(parse_log)

        intent = parsed.get("intent", "")
        disease_focus = parsed.get("disease_focus") or []
        is_disease_related = parsed.get("is_disease_related", True)

        # 2) G2 + B3 behavior:
        #    If clear disease → go straight to retrieval.
        #    If greeting / off-topic / no disease → show greeting instead.
        if intent == "greeting" or not disease_focus or not is_disease_related:
            greeting = self._build_greeting()

            full_turn = {
                "query": user_input,
                "parsed": parsed,
                "retrieved": {"query": "", "trials": [], "avg_confidence": 0.0},
                "response": greeting,
                "timestamp": parse_log["timestamp"],
            }
            self.conversation_history.append(full_turn)

            return {
                "response": greeting,
                "avg_confidence": 0.0,
                "num_trials": 0,
                "provenance": self.provenance_log[-5:],
                "session_hash": generate_reproducibility_hash(self.conversation_history),
            }

        # 3) Update profile for disease-related / trial queries
        turn_data = {"query": user_input, "parsed": parsed}
        profile_log = self.profile_agent.update_profile(turn_data)
        self.provenance_log.append(profile_log)

        # 4) Retrieve trials from Qdrant
        retrieved, retrieval_log = self.retrieval.retrieve(parsed, top_k=5)
        self.provenance_log.append(retrieval_log)

        # 5) Generate advisory response
        profile_snapshot = {
            "user_id": self.profile_agent.profile.get("user_id", "Patient"),
            "known_conditions": self.profile_agent.profile.get("extracted_conditions", []),
        }

        draft, advisor_log = self.advisor.advise(parsed, retrieved, profile_snapshot)
        self.provenance_log.append(advisor_log)

        # 6) Safety filter
        advice_text = draft.get("recommendation", "") if isinstance(draft, dict) else str(draft)
        trials = retrieved.get("trials", [])

        final_response, safety_status, safety_log = self.safety_filter.verify(advice_text, trials)
        self.provenance_log.append(safety_log)

        # 7) Save turn
        full_turn = {
            "query": user_input,
            "parsed": parsed,
            "retrieved": retrieved,
            "response": final_response,
            "timestamp": parse_log["timestamp"],
            "safety_status": safety_status,
        }
        self.conversation_history.append(full_turn)

        return {
            "response": final_response,
            "avg_confidence": retrieved.get("avg_confidence", 0.0),
            "num_trials": len(retrieved.get("trials", [])),
            "provenance": self.provenance_log[-5:],
            "session_hash": generate_reproducibility_hash(self.conversation_history),
            "safety_status": safety_status,
        }


def run_bot(user_input: str, qdrant_client, embed_model, gemini_model) -> Dict[str, Any]:
    """Convenience wrapper for single queries."""
    bot = HealthcareBot(qdrant_client, embed_model, gemini_model)
    return bot.chat(user_input)
