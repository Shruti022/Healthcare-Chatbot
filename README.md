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
- Python **3.10+** (tested with your Colab / local environment; 3.10 or 3.11 is safest for the torch + transformers versions in `requirements`).
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
	- ⁠ "asthma with obesity study?" 
4. The assistant will:
- Parse the query to detect disease and intent.  
- Retrieve trials from Qdrant and compute a hybrid score.   
- Return *up to 5 trials* with NCT IDs, titles, plain-English summaries, and (when available) PubMed abstracts and links.   
- Show metrics in the “📊 Details” expander (Trials Found, Confidence, Session Hash).   
- Include a safety disclaimer that it does not give diagnoses or treatment recommendations. 


### Optional: Run in Colab with public URL
You can also launch the app from a Colab notebook and expose it via Cloudflare Tunnel (as in your evaluation notebook).   
```bash
# Install dependencies
!pip install -r requirements

# (Optional) download cloudflared
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

# Start Streamlit in the background
!streamlit run app.py &>/dev/null &

# Expose the app on a public URL
!./cloudflared tunnel --url http://localhost:8501 --no-autoupdate
```

> Set ⁠ GEMINI_API_KEY ⁠, ⁠ QDRANT_API_KEY ⁠, and ⁠ QDRANT_URL ⁠ as environment variables in the notebook or via the Streamlit sidebar.
> The last command prints a public HTTPS URL you can share for demos.



### Optional: Deploying to Google Cloud Run

The repo includes a ⁠ Dockerfile ⁠ so you can deploy the Streamlit app as a container on Cloud Run.

⁠*High-level steps:*
1. Build and push the image:
```bash
gcloud builds submit --tag gcr.io/<PROJECT_ID>/clinical-trials-app
```
⁠
2. Deploy to Cloud Run:
```bash
gcloud run deploy clinical-trials-app \
	--image gcr.io/<PROJECT_ID>/clinical-trials-app \
	--platform managed \
	--region us-central1 \
	--allow-unauthenticated \
	--set-env-vars GEMINI_API_KEY=...,QDRANT_API_KEY=...,QDRANT_URL=...
```
⁠
⁠3. Cloud Run returns a URL like  
> https://clinical-trials-app-XXXXXXXXXX.us-central1.run.app/ ⁠ – this is the stable URL we’re using now.   
⁠In Cloud Run, environment variables are the recommended way to inject secrets; they are not visible in the source code or UI.


### Advanced: Updating the Qdrant corpus
⁠The app assumes a Qdrant collection ⁠ clinical_trials ⁠ already exists. To rebuild or extend it from CSV exports, use ⁠ update_qdrant_auto.py.

*What the script does:*
- Finds all ⁠ .csv ⁠ files in a specified Drive folder.
- Loads and concatenates them, *drops duplicate trials by ⁠ nct_id ⁠*, and filters out bad statuses (Terminated, Withdrawn, Suspended, etc.).
- Builds chunks of the form ⁠ "Title: ...\nSummary: ..." ⁠ using ⁠ brief_title ⁠ and ⁠ brief_summary.
- Generates 384‑dim embeddings with ⁠ SentenceTransformer("all-MiniLM-L6-v2").
- Uploads embeddings + payloads into the ⁠ clinical_trials ⁠ collection in Qdrant, either in ⁠ "refresh" ⁠ mode (delete + recreate) or ⁠ "add" ⁠ mode.

*Usage (inside Colab or locally):*
```python
from update_qdrant_auto import QdrantAutoPipeline

DRIVE_FOLDER = "/content/drive/MyDrive/LLM_Based_GenAI_Sem1/data"
QDRANT_URL  = "https://<your-cluster>.qdrant.io"
 
pipeline = QdrantAutoPipeline(QDRANT_URL, qdrant_key)
pipeline.run_auto_pipeline(DRIVE_FOLDER, mode="refresh")  # or "add"
```
 ⁠
⁠This step is *optional*; most users can simply reuse the existing Qdrant index without running this pipeline.



### Troubleshooting

⁠*Environment / versions*
- Use Python *3.10 or 3.11*; older versions may conflict with ⁠ torch==2.3.1 ⁠ and ⁠ transformers==4.40.2.
- If installation fails, upgrade pip: ⁠ python -m pip install --upgrade pip .


⁠*App does not start / Streamlit error*
- Ensure ⁠ pip install -r requirements ⁠ completed without errors.
- Confirm you run ⁠ streamlit run app.py ⁠ from the repo root (where ⁠ app.py ⁠ lives). 

*“Please enter API keys first!” in the UI*
- Either set ⁠ GEMINI_API_KEY ⁠, ⁠ QDRANT_API_KEY ⁠, and ⁠ QDRANT_URL ⁠ as environment variables *before* launching, or fill them in the sidebar “⚙️ Configuration” section. 


⁠*Qdrant connection errors*
- Check that ⁠ QDRANT_URL ⁠ matches your cluster URL exactly (including region and protocol).
- Verify ⁠ QDRANT_API_KEY ⁠ is correct and the collection ⁠ clinical_trials ⁠ exists and has points (you can confirm with ⁠ client.get_collection("clinical_trials") ⁠).


⁠*Gemini errors / rate limit issues*
- Make sure ⁠ GEMINI_API_KEY ⁠ is valid and has access to Gemini 2.0 Flash.
- For large evaluation runs, respect rate limits; your notebooks use ⁠ time.sleep(0.75) ⁠ between calls to avoid hitting quotas. 


⁠*No trials returned / empty results*
- Verify Qdrant collection ⁠ clinical_trials ⁠ contains vectors (⁠ points_count > 0 ⁠).
- Check that your query is about one of the 13–14 supported diseases; off-topic or unsupported conditions may trigger the greeting or a low-confidence message instead of trials. 


⁠*UI shows but is slow*
- First call may be slower because it loads the embedding model and Qdrant client.
- On Cloud Run, ensure you allocate enough memory/CPU for the container; cold starts are expected.
