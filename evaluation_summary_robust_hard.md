# Robustness Evaluation – Challenging Queries

## Overall Performance

- Top-1 accuracy (hard queries): **71.4%**
- Hit@5 (at least one correct disease in top-5): **88.6%**
- NDCG@5: **203.0%**
- MAP@5: **77.0%**

## Per-Disease Robustness

- **asthma** – Top-1: 100.0%, Hit@5: 100.0%, NDCG@5: 248.1%, MAP@5: 93.9%
- **obesity** – Top-1: 100.0%, Hit@5: 100.0%, NDCG@5: 243.3%, MAP@5: 89.5%
- **parkinson** – Top-1: 100.0%, Hit@5: 100.0%, NDCG@5: 263.6%, MAP@5: 93.4%
- **prostate_cancer** – Top-1: 100.0%, Hit@5: 100.0%, NDCG@5: 242.1%, MAP@5: 97.8%
- **diabetes** – Top-1: 80.0%, Hit@5: 100.0%, NDCG@5: 221.3%, MAP@5: 88.7%
- **breast_cancer** – Top-1: 80.0%, Hit@5: 100.0%, NDCG@5: 242.1%, MAP@5: 91.8%
- **rheumatoid_arthritis** – Top-1: 80.0%, Hit@5: 80.0%, NDCG@5: 235.9%, MAP@5: 80.0%
- **hypertension** – Top-1: 80.0%, Hit@5: 100.0%, NDCG@5: 222.4%, MAP@5: 79.1%
- **cardiovascular** – Top-1: 60.0%, Hit@5: 80.0%, NDCG@5: 122.5%, MAP@5: 67.8%
- **alzheimer** – Top-1: 60.0%, Hit@5: 100.0%, NDCG@5: 238.5%, MAP@5: 85.2%
- **copd** – Top-1: 60.0%, Hit@5: 100.0%, NDCG@5: 226.2%, MAP@5: 79.7%
- **ckd** – Top-1: 60.0%, Hit@5: 80.0%, NDCG@5: 199.5%, MAP@5: 73.6%
- **lung_cancer** – Top-1: 40.0%, Hit@5: 100.0%, NDCG@5: 136.6%, MAP@5: 57.9%
- **stroke** – Top-1: 0.0%, Hit@5: 0.0%, NDCG@5: 0.0%, MAP@5: 0.0%

## Error Patterns & Confusions

- The confusion matrix (saved as `eval_confusion_matrix_hard.csv`) shows where the system retrieves the wrong disease under difficult, ambiguous conditions (e.g., cardiometabolic overlap between diabetes, obesity, and cardiovascular disease, or asthma vs COPD vs lung cancer).
- These hard queries intentionally mix comorbidities, vague language, and non-standard phrasing, so we **expect performance to drop** compared to the clean baseline. This provides evidence of limitations and future work.