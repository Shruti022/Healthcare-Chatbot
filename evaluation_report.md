# Clinical Trial Retrieval Evaluation

## Systems Compared

- **HealthcareBot**: Full pipeline (parser + QdrantRetrievalAgent + advisor + safety filter)

- **QdrantOnly**: Simple vector similarity over Qdrant (no disease reweighting, no safety filter)

## Query Splits

- **Baseline queries**: Clean, disease-specific queries with clear intent (5 per disease).

- **Robust queries**: Noisy, patient-like free-text queries (5 per disease).

- **Diseases covered** (14): diabetes, obesity, hypertension, cardiovascular, ckd, alzheimer, parkinson, asthma, copd, breast_cancer, lung_cancer, prostate_cancer, stroke, rheumatoid_arthritis

## Aggregate Metrics

Mean metrics by system and split:

| system        | split    |   top1_correct |   recall_at_5 |   ndcg_at_5 |   map_at_5 |
|:--------------|:---------|---------------:|--------------:|------------:|-----------:|
| HealthcareBot | baseline |       0.557143 |      0.657143 |    0.605986 |   0.57877  |
| HealthcareBot | robust   |       0.342857 |      0.4      |    0.366232 |   0.347321 |
| QdrantOnly    | baseline |       0.457143 |      0.685714 |    0.571476 |   0.519762 |
| QdrantOnly    | robust   |       0.357143 |      0.6      |    0.486925 |   0.440159 |


## Safety Filter Behavior (HealthcareBot)

Overall safety status distribution:

| safety_status        |   count |   fraction |
|:---------------------|--------:|-----------:|
| Pass (Trial Listing) |      81 |  0.578571  |
| Revised (API Error)  |       5 |  0.0357143 |


Per-disease safety status breakdown (head):

| disease              | safety_status        |   count |   fraction |
|:---------------------|:---------------------|--------:|-----------:|
| alzheimer            | Pass (Trial Listing) |       6 |   1        |
| asthma               | Pass (Trial Listing) |       7 |   0.875    |
| asthma               | Revised (API Error)  |       1 |   0.125    |
| breast_cancer        | Pass (Trial Listing) |       7 |   1        |
| cardiovascular       | Pass (Trial Listing) |       5 |   1        |
| ckd                  | Pass (Trial Listing) |       3 |   0.75     |
| ckd                  | Revised (API Error)  |       1 |   0.25     |
| copd                 | Pass (Trial Listing) |       5 |   1        |
| diabetes             | Pass (Trial Listing) |       8 |   0.888889 |
| diabetes             | Revised (API Error)  |       1 |   0.111111 |
| hypertension         | Pass (Trial Listing) |       5 |   1        |
| lung_cancer          | Pass (Trial Listing) |       7 |   1        |
| obesity              | Pass (Trial Listing) |       4 |   1        |
| parkinson            | Pass (Trial Listing) |       5 |   0.833333 |
| parkinson            | Revised (API Error)  |       1 |   0.166667 |
| prostate_cancer      | Pass (Trial Listing) |       5 |   1        |
| rheumatoid_arthritis | Pass (Trial Listing) |       7 |   1        |
| stroke               | Pass (Trial Listing) |       7 |   0.875    |
| stroke               | Revised (API Error)  |       1 |   0.125    |


## Observations & Interpretation (to fill in)

- Where does HealthcareBot outperform QdrantOnly, especially on robust queries?

- Which diseases are hardest under robust queries (lowest NDCG/MAP)?

- How often does the safety filter revise responses, and for which diseases?

- How does performance degrade from baseline → robust, and is the degradation clinically acceptable?
