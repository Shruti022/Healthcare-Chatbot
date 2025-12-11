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


## Quickstart: run locally
Prerequisites
- Python 3.10+ (tested with your Colab / local environment; 3.10 or 3.11 is safest for the torch + transformers versions in requirements).​
- A Qdrant Cloud cluster with an existing clinical_trials collection and API key.​
- A Gemini API key with access to Gemini 2.0 Flash.
- git and pip installed

Setup
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements

```


