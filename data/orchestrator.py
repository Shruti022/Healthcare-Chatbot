import json
import hashlib
# NOTE: The import below relies on agents.py and utils.py being run first via %run
from agents import SymptomParser, RetrievalAgent, DiagnosisAdvisor, ActiveSafetyFilter, ProfileAgent
from utils import generate_reproducibility_hash, log_provenance_step

class HealthcareBot:
    def __init__(self, gemini_model, embed_model, faiss_index, chunk_map, initial_profile=None):
        self.gemini_model = gemini_model
        
        # --- Initialize Agents ---
        self.parser = SymptomParser(gemini_model)
        self.profile_agent = ProfileAgent(initial_profile)
        # Note: Retriever depends on the ProfileAgent for filtering
        self.retriever = RetrievalAgent(embed_model, faiss_index, chunk_map, self.profile_agent)
        self.advisor = DiagnosisAdvisor(gemini_model)
        self.safety = ActiveSafetyFilter(gemini_model)
        
        # --- Internal State for Conversation Continuity ---
        self.history = [] # Short-term memory of the conversation
        self.provenance_chain = [] # Stores logs from every agent call

    def _handle_simple_greeting(self, user_input):
        """Handles simple non-RAG conversational turns (e.g., "Hi")."""
        # Uses the profile to personalize the greeting
        greeting_response = f"Hello {self.profile_agent.profile['user_id']}! I am your AI Health Assistant. I can look up information from clinical trials and medical guidelines regarding your conditions ({', '.join(self.profile_agent.profile['conditions'])}). Please tell me your symptoms or query."
        
        # Log this non-RAG step
        log_entry = log_provenance_step('GreetingAgent', user_input, greeting_response, {'type': 'Greeting/Non-RAG'})
        self.provenance_chain.append(log_entry)
        
        # Generate hash based on the input and lack of citations
        session_hash = generate_reproducibility_hash(self.history + [{'query': user_input}])
        
        return {
            'recommendation': greeting_response,
            'cited_trials': [],
            'safety_status': "Non-RAG Pass",
            'session_hash': session_hash
        }

    def process_query(self, user_input):
        self.provenance_chain = [] # Reset chain for new turn
        
        # 1. Parse Input
        parsed, parse_log = self.parser.parse(user_input)
        self.provenance_chain.append(parse_log)
        
        # --- Multi-Agent Consensus: Route based on Parse Output (Addressing 'Hi' queries) ---
        if parsed.get('context') == 'greeting' or not parsed.get('symptoms'):
            return self._handle_simple_greeting(user_input)

        # 2. Retrieve Data (Profile-Conditioned)
        retrieved_data, retrieve_log = self.retriever.retrieve(parsed)
        self.provenance_chain.append(retrieve_log)

        # 3. Draft Advice (Evidence-Weighted)
        draft_advice, advise_log = self.advisor.advise(parsed, retrieved_data)
        self.provenance_chain.append(advise_log)

        # --- Check for Veto (Retrieval Confidence Veto) ---
        if draft_advice.get('confidence_veto', False):
            final_text = draft_advice['recommendation']
            safety_status = "Vetoed (Low Confidence)"
            evidence_list = []
        else:
            # 4. Safety Audit (Multi-Agent Veto/Correction)
            final_text, safety_status, safety_log = self.safety.verify(
                draft_advice['recommendation'],
                retrieved_data['trials']
            )
            self.provenance_chain.append(safety_log)
            evidence_list = retrieved_data['trials']

        # 5. Provenance & Hashing
        nct_ids = [t['nct_id'] for t in evidence_list]
        
        # Update history for the hash calculation
        current_turn_for_hash = {'query': user_input, 'response_hash': None} # Hash is generated *after* adding
        session_hash = generate_reproducibility_hash(self.history + [current_turn_for_hash])
        
        # 6. Update Profile/History (Memory)
        new_turn_data = {'query': user_input, 'response_hash': session_hash}
        profile_log = self.profile_agent.update_profile(new_turn_data)
        self.history.append(new_turn_data) # Update local history only after hash generated
        self.provenance_chain.append(profile_log)


        return {
            'recommendation': final_text,
            'cited_trials': nct_ids,
            'safety_status': safety_status,
            'session_hash': session_hash,
            'provenance_chain': self.provenance_chain # <-- Full log for the paper
        }