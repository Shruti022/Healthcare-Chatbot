## Overview 
Clinical Trials Search Assistant is a Streamlit web app that helps clinicians, clinical researchers, and informed patients discover relevant clinical trials across ~135k studies from ClinicalTrials.gov.​
The system combines SentenceTransformer (all‑MiniLM‑L6‑v2) embeddings, a Qdrant vector database, and Gemini 2.0 Flash to support natural‑language trial search for 13–14 diseases (diabetes, obesity, hypertension, cardiovascular disease, CKD, Alzheimer’s, Parkinson’s, asthma, COPD, breast cancer, lung cancer, prostate cancer, stroke, and rheumatoid arthritis).
A multi‑agent pipeline (parser → disease‑aware retriever → advisor → safety filter) surfaces the top‑5 trials for a query, explains them in plain language, and enforces safety constraints to avoid direct treatment recommendations


## Repository structure
The core files in this repo (deployment folder) are:
- app.py – Streamlit UI for the Clinical Trials Search Assistant.
- run_bot_qdrant.py – defines HealthcareBot, wiring together the parser, profile agent, Qdrant retriever, diagnosis advisor, and safety filter.
- utils_qdrant.py – utilities to connect to Qdrant, load the MiniLM embedding model, log provenance, and compute reproducibility hashes.
- retrieval_agent_qdrant.py – implements the QdrantRetrievalAgent with disease‑aware scoring over the clinical_trials collection.
- requirements – Python dependencies used by the app and Docker image (Streamlit, qdrant-client, google-generativeai, sentence-transformers, torch, transformers, pandas, numpy, etc.).
- Dockerfile, .dockerignore – container image definition for deployment on Google Cloud Run.
- update_qdrant_auto.py – optional script to rebuild or extend the Qdrant index from CSV exports (deduplicates nct_id, filters bad statuses, embeds with MiniLM, uploads to Qdrant).​
(This is present in Colab notebook)


### Quickstart: run locally

**Prerequisites**  
- Python **3.10+** (tested with your Colab / local environment; 3.10 or 3.11 is safest for the torch + transformers versions in `requirements`).[1]
- A Qdrant Cloud cluster with an existing `clinical_trials` collection and API key.
- A Gemini API key with access to **Gemini 2.0 Flash**.   
- `git` and `pip` installed.

**Setup**  
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements
```

The `requirements` file includes:
- `streamlit==1.31.0`  
- `pandas==2.1.4`, `numpy==1.26.3`  
- `qdrant-client==1.16.1`  
- `google-generativeai==0.3.2`  
- `torch==2.3.1`, `transformers==4.40.2`, `sentence-transformers==2.7.0`  
- `requests==2.31.0` 

**Configure environment variables**  

Set the following environment variables in your shell (or via a `.env` file and a loader if you prefer):
```bash
export GEMINI_API_KEY="your_gemini_key_here"
export QDRANT_API_KEY="your_qdrant_key_here"
export QDRANT_URL="https://<your-cluster>.qdrant.io"
```

On Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="your_gemini_key_here"
$env:QDRANT_API_KEY="your_qdrant_key_here"
$env:QDRANT_URL="https://<your-cluster>.qdrant.io"
```

In the Streamlit app, you can also enter keys in the **sidebar**, but for production deployments (Docker / Cloud Run) use environment variables.

**Run the app locally**
```bash
streamlit run app.py
```

Then open the URL printed by Streamlit (usually `http://localhost:8501`) in your browser. You should see:
  - Title “Clinical Trials Search Assistant”
  - A sidebar with fields for Gemini API Key, Qdrant API Key, and Qdrant URL
  - A chat input box “Ask about clinical trials...”



### Configuration & environment variables

The app needs three secrets/config values:
- `GEMINI_API_KEY` – Google Gemini API key with access to **Gemini 2.0 Flash** (used by SymptomParser, DiagnosisAdvisor, and ActiveSafetyFilter).   
- `QDRANT_API_KEY` – API key for your Qdrant Cloud cluster.   
- `QDRANT_URL` – HTTPS URL of your Qdrant cluster (for example: `https://215ec69e-fa22-4f38-bcf3-941e73901a68.us-east4-0.gcp.cloud.qdrant.io`).   

> **Local development:** you can either export these variables before running `streamlit run app.py`, or enter them in the Streamlit sidebar under “⚙️ Configuration”.   
> **Docker / Cloud Run:** set these as environment variables on the service (recommended) so keys are **not** hard-coded in code or UI.


### Using the app

⁠1. Open the app (local ⁠ http://localhost:8501 ⁠ or your Cloud Run URL).   
⁠2. Configure keys in the sidebar if they are not already set. Once keys are valid, the sidebar shows “✓ Keys configured” and the bot is initialized.   
⁠3. Type a question in the chat input, such as:
	- Baseline-style examples:  
		- ⁠ "GLP-1 agonist trials for type 2 diabetes" ⁠  
		- ⁠ "breast cancer immunotherapy trials"
	- Robust, patient-style examples:  
		- ⁠ "RA meds stopped working what studies?" 
		- ⁠ "asthma with obesity study?" ⁠   
⁠4. The assistant will:
	- Parse the query to detect disease and intent.  
	- Retrieve trials from Qdrant and compute a hybrid score.   
	- Return *up to 5 trials* with NCT IDs, titles, plain-English summaries, and (when available) PubMed abstracts and links.   
	- Show metrics in the “📊 Details” expander (Trials Found, Confidence, Session Hash).   
	- Include a safety disclaimer that it does not give diagnoses or treatment recommendations. 

