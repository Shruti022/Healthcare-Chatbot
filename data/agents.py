import json
import re
import faiss
from datetime import datetime
from google.generativeai.types import HarmCategory, HarmBlockThreshold
# NOTE: The import below relies on utils.py being run first via %run
from utils import calculate_confidence_score, log_provenance_step

# --- 1. Symptom Parser ---
class SymptomParser:
    """Extracts structured medical entities from free text."""
    def __init__(self, gemini_model):
        self.model = gemini_model

    def parse(self, user_input):
        prompt = f"""Extract medical entities to JSON.
        Input: "{user_input}"
        Output format: {{"symptoms": ["list"], "duration": "text", "context": "text"}}
        If the input is a simple greeting (e.g., "Hi", "Hello"), return {{"symptoms": [], "duration": "greeting", "context": "greeting"}}.
        """
        
        # Check for simple greeting first (Performance improvement for conversational flow)
        if user_input.strip().lower() in ['hi', 'hello', 'hey']:
             parsed = {"symptoms": [], "duration": "greeting", "context": "greeting"}
        else:
            try:
                response = self.model.generate_content(prompt)
                text = response.text.strip()
                match = re.search(r'\{.*\}', text, re.DOTALL)
                parsed = json.loads(match.group(0)) if match else json.loads(text)
            except Exception as e:
                print(f"Parser Error: {e}")
                parsed = {"symptoms": [user_input], "duration": "unknown", "context": "error"}

        return parsed, log_provenance_step('SymptomParser', user_input, parsed)


# --- 2. Profile Agent ---
class ProfileAgent:
    """Stores user context (long-term profile) and conversation history."""
    def __init__(self, initial_profile=None):
        self.profile = {
            'user_id': 'master_ds_student_default',
            'age': 45,                  # Filters pediatric trials
            'conditions': ['Type 2 Diabetes'], # Primary clinical focus
            'medications': ['Metformin'],
            'history': []               # Short-term conversation history
        }
        if initial_profile:
            self.profile.update(initial_profile)

    def get_profile_context(self):
        """Returns structured context string for query conditioning."""
        context = (f"User Profile: Age {self.profile['age']} "
                   f"with Conditions: {', '.join(self.profile['conditions'])}. "
                   f"Current medications: {', '.join(self.profile['medications'])}.")
        return context

    def update_profile(self, new_turn_data):
        """Updates short-term memory (history)."""
        self.profile['history'].append(new_turn_data)
        self.profile['history'] = self.profile['history'][-5:] # Keep last 5 turns
        return log_provenance_step('ProfileAgent', new_turn_data, self.profile['history'])


# --- 3. Retrieval Agent ---
class RetrievalAgent:
    """Implements RAG with Profile-Conditioned filtering and Evidence Scoring."""
    def __init__(self, embed_model, faiss_index, chunk_map, profile_agent):
        self.embed_model = embed_model
        self.index = faiss_index
        self.chunk_map = chunk_map
        self.profile_agent = profile_agent
        self.embedding_dimension = embed_model.get_sentence_embedding_dimension()

    def retrieve(self, parsed_symptoms, top_k=10):
        
        if not parsed_symptoms.get('symptoms'):
            return {'trials': [], 'query': None, 'max_distance': float('inf')}, log_provenance_step('RetrievalAgent', parsed_symptoms, 'No symptoms, skipping retrieval.')
            
        # --- Novelty: Profile-Conditioned Query Enhancement ---
        profile_context = self.profile_agent.get_profile_context()
        base_query = f"{' '.join(parsed_symptoms.get('symptoms', []))} {parsed_symptoms.get('context', '')}"
        
        # Conditioned Query: Adds profile context for better semantic retrieval
        query_text = f"Clinical trial search for: {base_query}. Considering user profile: {profile_context}"
        query_embedding = self.embed_model.encode([query_text]).astype('float32')

        # Retrieve a buffer of candidates (top_k * 2) for filtering and sorting
        D, I = self.index.search(query_embedding, top_k * 2)
        
        retrieved = []
        seen_nct_ids = set()
        
        for i, idx in enumerate(I[0]):
            if idx == -1: continue 
            
            item = self.chunk_map[idx]
            nct_id = item['nct_id']
            
            # --- Novelty: Profile-Conditioned Filter (Age-based Exclusion) ---
            is_pediatric = re.search(r'\b(child|pediatric|adolescent)\b', item['text'], re.IGNORECASE)
            
            if self.profile_agent.profile['age'] >= 18 and is_pediatric:
                continue # Skip this trial due to age mismatch

            if nct_id not in seen_nct_ids:
                # --- Novelty: Evidence Scoring ---
                confidence_score = calculate_confidence_score(D[0][i])
                
                retrieved.append({
                    'nct_id': nct_id,
                    'title': item['title'],
                    'text': item['text'],
                    'status': item['status'],
                    'retrieval_score': confidence_score,
                    'retrieval_distance': D[0][i]
                })
                seen_nct_ids.add(nct_id)
                
                if len(retrieved) >= top_k: break 

        # Final sort is generally by score/distance
        retrieved.sort(key=lambda x: x['retrieval_score'], reverse=True)
        
        max_distance = D[0][0] if D.size > 0 else float('inf')
        
        retrieval_output = {'trials': retrieved, 'query': query_text, 'max_distance': max_distance}
        return retrieval_output, log_provenance_step('RetrievalAgent', parsed_symptoms, retrieval_output)


# --- 4. Diagnosis Advisor ---
class DiagnosisAdvisor:
    """Generates evidence-backed recommendations with confidence-based veto."""
    def __init__(self, gemini_model, confidence_threshold=0.85): 
        self.model = gemini_model
        self.confidence_threshold = confidence_threshold

    def advise(self, parsed_symptoms, retrieved_data):
        
        # --- Novelty: Retrieval Confidence Veto (Part of Multi-agent consensus + veto) ---
        best_score = retrieved_data['trials'][0]['retrieval_score'] if retrieved_data['trials'] else 0.0
        
        if best_score < self.confidence_threshold:
             veto_message = f"⚠️ EVIDENCE ALERT: The search did not find strong, specific clinical evidence (best score: {best_score:.2f}) related to your query. Please consult a medical professional for guidance."
             return {'recommendation': veto_message, 'evidence_used': [], 'confidence_veto': True}, log_provenance_step('DiagnosisAdvisor', retrieved_data, {'veto': True})

        # --- Novelty: Evidence-weighted Prompting ---
        evidence_lines = []
        for t in retrieved_data['trials']:
            # Inject the score into the context for the LLM to weight its own decision
            evidence_lines.append(f"TRIAL {t['nct_id']} (Relevance Score: {t['retrieval_score']:.2f}): {t['text']}")
            
        evidence = "\n".join(evidence_lines)
        
        # Construct the detailed prompt
        prompt = f"""Role: Evidence-Based Medical Assistant. Use the provided clinical trial evidence to generate a transparent and fact-grounded response.

        PATIENT QUERY/SYMPTOMS: {parsed_symptoms}

        AUTHORIZED EVIDENCE (Ranked by Relevance Score):
        {evidence}

        TASK:
        1. Answer the patient's specific question using ONLY the evidence provided, prioritizing findings from trials with the HIGHEST Relevance Score.
        2. Always cite the specific Trial ID (TRIAL NCT_ID) for every claim made.
        3. If the evidence does not explicitly answer the specific question, or if all high-score evidence is ambiguous, YOU MUST STATE that the evidence is insufficient.
        4. Include a general medical disclaimer at the end of your response.
        5. DO NOT GUESS, HALLUCINATE, or provide recommendations not supported by the evidence.
        """
        try:
            res = self.model.generate_content(prompt)
            advisor_output = {'recommendation': res.text, 'evidence_used': retrieved_data['trials'], 'confidence_veto': False}
            return advisor_output, log_provenance_step('DiagnosisAdvisor', retrieved_data, advisor_output)
        except Exception as e:
            error_output = {'recommendation': f"Error generating advice: {e}", 'evidence_used': [], 'confidence_veto': False}
            return error_output, log_provenance_step('DiagnosisAdvisor', retrieved_data, error_output, {'error': str(e)})


# --- 5. Active Safety Filter ---
class ActiveSafetyFilter:
    """Final check layer for dangerous content, ensuring medical disclaimers."""
    def __init__(self, gemini_model):
        self.model = gemini_model
        # Configure model to allow reading trap questions for critique
        self.safety_config = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.system_disclaimer = "\n\n***Medical Disclaimer: This information is for educational purposes only and is NOT a substitute for professional medical advice. Always consult a qualified healthcare provider.***"


    def verify(self, advice_text, evidence_list):
        
        # Skip LLM audit if the recommendation is already a Veto message
        if "EVIDENCE ALERT" in advice_text:
            return advice_text, "Vetoed (Low Confidence)", log_provenance_step('SafetyFilter', {'input': advice_text, 'evidence': evidence_list}, {'status': 'Skipped Audit'})

        evidence_str = "\n".join([t['text'] for t in evidence_list])

        audit_prompt = f"""
        AUDIT TASK: You are a Medical Safety Officer. Review the AI's draft advice.

        EVIDENCE: {evidence_str[:5000]} ...

        ADVICE TO CHECK: {advice_text}

        CHECKLIST:
        1. Does the advice suggest changing medication or stopping treatment without a doctor? -> FAIL
        2. Does the advice claim a cure or make unsupported definitive statements? -> FAIL
        3. Does the advice ignore a dangerous user intent (e.g., overdose question)? -> FAIL
        
        If SAFE, reply "SAFE".
        If UNSAFE, reply with a CORRECTED, SAFER, and highly cautionary version.
        """
        
        try:
            res = self.model.generate_content(
                audit_prompt,
                safety_settings=self.safety_config
            )

            audit_result = res.text
            final_status = "Pass"
            
            if "SAFE" in audit_result and len(audit_result) < 20:
                final_text = advice_text # Use original text if safe
            else:
                final_text = f"⚠️ SAFETY REVISION: {audit_result}"
                final_status = "Revised"
            
            # Always append the hardcoded disclaimer
            final_text += self.system_disclaimer

            audit_log = log_provenance_step('SafetyFilter', advice_text, {'status': final_status, 'audit_llm_output': audit_result})
            return final_text, final_status, audit_log

        except Exception as e:
            # Fail-safe mechanism
            fallback_msg = f"⚠️ SEVERE SAFETY ALERT: The system detected potentially dangerous content and failed audit. Please consult a doctor."
            final_status = "Revised (API Triggered)"
            final_text = fallback_msg + self.system_disclaimer
            audit_log = log_provenance_step('SafetyFilter', advice_text, {'status': final_status, 'error': str(e)})
            return final_text, final_status, audit_log