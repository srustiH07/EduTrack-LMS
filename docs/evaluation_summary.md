# EduTrack LMS - Evaluation Summary

## 1. Evaluation Objective

The recommendation system was evaluated using NDCG@10 to measure the quality of the ranked recommendation list.

The client acceptance target is:

**NDCG@10 >= 0.55**

---

## 2. SVD Hyperparameter Experiments

Eight SVD configurations were evaluated using the same leave-one-out evaluation strategy.

| Experiment                   | NDCG@10 |
| ---------------------------- | ------: |
| Baseline                     |  0.4773 |
| Factors 20                   |  0.4845 |
| Factors 100                  |  0.4577 |
| Epochs 60                    |  0.4807 |
| Higher learning rate         |  0.4840 |
| Lower regularization         |  0.4773 |
| Higher regularization        |  0.4755 |
| Combined tuned configuration |  0.4565 |

### Best configuration

* Factors: 20
* Epochs: 30
* Learning rate: 0.005
* Regularization: 0.02
* Random state: 42
* NDCG@10: **0.4845**

---

## 3. Target Gap

Target NDCG@10:

**0.55**

Best observed NDCG@10:

**0.4845**

Remaining gap:

**0.0655**

Therefore, the current offline model does **not yet meet the client's NDCG@10 target**.

This result is reported transparently. No artificial score adjustment or fabricated evaluation result is used.

---

## 4. Retraining Evaluation

The retraining pipeline was executed using:

`src/retrain_model.py`

The candidate model achieved:

**NDCG@10: 0.4669**

The current production model achieved:

**NDCG@10: 0.4845**

Because the candidate performed worse than the production model, the candidate was rejected.

Result:

**Current production model remains unchanged.**

This demonstrates the model-promotion safeguard in the retraining pipeline.

---

## 5. Recommendation Tests

### Existing Student

Student:

`STU-00007`

Recommendation type:

**Hybrid**

Top recommendations include:

1. Web Development
2. Finance
3. Python Programming
4. Mathematics
5. Data Science

Observed API response time:

**221 ms**

---

### Cold-Start Student

Student:

`NEW-STUDENT-001`

Recommendation type:

**Content-Based Fallback**

Top recommendations include:

1. Web Development
2. Finance
3. Physics
4. Mathematics
5. Chemistry

Observed API response time:

**148 ms**

---

## 6. Dashboard Performance

Client requirement:

**Recommendation rendering under 1 second**

Observed existing-student response:

**221 ms**

Observed cold-start response:

**148 ms**

Both observed response times are below the required 1-second threshold.

---

## 7. Acceptance Criteria Summary

| Acceptance Criterion              | Result                   |
| --------------------------------- | ------------------------ |
| NDCG@10 >= 0.55                   | Not yet achieved         |
| Streamlit response under 1 second | Passed in observed tests |
| Cold-start fallback               | Implemented and tested   |
| A/B test design documented        | Completed                |
| Model retrain pipeline            | Implemented and tested   |

---

## 8. Current Project Status

The project has an end-to-end working prototype consisting of:

```text
Data Pipeline
     ↓
SVD Matrix Factorization
     ↓
Hybrid Recommendation Engine
     ↓
Cold-Start Fallback
     ↓
FastAPI REST API
     ↓
Streamlit Dashboard
```

The remaining primary ML improvement is to increase ranking quality from the current best NDCG@10 of **0.4845** to the target of **0.55 or higher**.

Possible future improvements include richer course-level data, additional implicit-feedback models, better content features, temporal signals, and learning-to-rank approaches.
