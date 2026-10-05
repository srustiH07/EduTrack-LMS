# EduTrack LMS - Adaptive Learning Path Recommendation System

## 1. Project Overview

EduTrack LMS is an adaptive learning path recommendation system designed for an EdTech platform.

The system recommends the next learning subjects for students using collaborative filtering with a content-based fallback for students who do not have sufficient interaction history.

The project provides:

* Data processing pipeline
* Matrix-factorization recommendation model
* Hybrid recommendation engine
* Cold-start content-based fallback
* FastAPI REST API
* Streamlit dashboard
* Offline NDCG@10 evaluation
* Hyperparameter experimentation
* Model retraining and promotion pipeline
* A/B test design
* Model card and deployment documentation

---

## 2. Problem Statement

Students on an online learning platform have different learning interests, engagement patterns, and subject histories.

A static learning sequence may not provide personalized recommendations.

EduTrack LMS addresses this problem by using historical student-content interactions to identify relevant learning subjects. Existing students receive hybrid recommendations combining collaborative filtering and content-based signals, while new students receive content-based recommendations using available student performance information.

---

## 3. System Architecture

```text
                    EduTrack LMS
                         |
                         v
              Student-Course Interactions
                         |
                         v
                Data Processing Pipeline
                         |
                         v
              Student-Subject Interaction
                         |
                         v
              Matrix Factorization (SVD)
                         |
                         v
               Recommendation Engine
                    /             \
                   /               \
          Existing Student      New Student
                |                    |
                v                    v
        Collaborative +        Content-Based
        Content-Based Signal       Fallback
                \                    /
                 \                  /
                  v                v
                 Top Recommendations
                         |
              +----------+----------+
              |                     |
              v                     v
          FastAPI API        Streamlit Dashboard
```

---

## 4. Recommendation Strategy

### Existing Students

For students with interaction history, the system uses a hybrid approach:

```text
Hybrid Score =
0.70 × Collaborative Filtering Score
+
0.30 × Content-Based Score
```

The collaborative filtering component uses matrix factorization through Surprise SVD.

The content-based component uses student performance characteristics to estimate subject suitability.

Previously seen subjects are filtered from the final recommendation list.

### Cold-Start Students

For a new student without interaction history, collaborative filtering cannot provide reliable personalized recommendations.

Therefore, the system automatically switches to a content-based fallback.

The fallback uses student performance characteristics and subject profiles to rank suitable learning subjects.

This prevents the system from returning empty recommendations for new students.

---

## 5. Dataset

The project uses the following provided data packs:

### Student Performance

`data/aiml_student_performance.csv`

Contains student performance and learning-progress information including:

* Student ID
* Grade
* Subject
* Pre-test score
* Post-test score
* Score improvement
* Lessons completed
* Completion percentage
* Quiz performance
* Platform sessions
* Device
* At-risk label

### Content Engagement

`data/aiml_content_engagement.csv`

Contains learning-content interaction information including:

* Student ID
* Content type
* Subject
* Time spent
* Completion
* Replay activity
* Likes
* Notes taken
* Difficulty rating
* Engagement score

### EdTech Tests

`data/aiml_edtech_tests.json`

Contains the provided EdTech test data used for project validation.

---

## 6. Data Processing

The data pipeline is implemented in:

`src/data_pipeline.py`

The pipeline:

1. Loads the content engagement dataset.
2. Validates the required columns.
3. Cleans numeric interaction fields.
4. Normalizes time and difficulty signals.
5. Creates a weighted interaction score.
6. Aggregates interactions by student and subject.
7. Saves the processed dataset.

Output:

`data/processed/interactions.csv`

Current processed dataset:

* Raw engagement records: 600
* Processed student-subject interactions: 588
* Unique students: 459
* Subjects: 10

---

## 7. Model Training

The primary collaborative filtering model is:

**Surprise SVD Matrix Factorization**

Implemented in:

`src/train_model.py`

The evaluation uses a leave-one-out strategy for students with at least two interactions.

The production SVD configuration selected during experimentation is:

* Factors: 20
* Epochs: 30
* Learning rate: 0.005
* Regularization: 0.02
* Random state: 42

Production model:

`models/svd_recommender.pkl`

---

## 8. Model Experimentation

Hyperparameter experiments are implemented in:

`src/tune_model.py`

Results are stored in:

`results/svd_experiments.csv`

The best tested SVD configuration achieved:

**NDCG@10: 0.4845**

Target:

**NDCG@10 >= 0.55**

The current SVD model does not yet meet the target. Hyperparameter tuning alone was insufficient, so the project uses a hybrid recommendation architecture and documents this limitation for further model improvement.

---

## 9. Recommendation Engine

The recommendation engine is implemented in:

`src/recommender.py`

It provides:

* Model loading
* Dataset loading
* Student profiling
* Collaborative filtering
* Content-based scoring
* Hybrid recommendation
* Cold-start fallback
* Confidence scoring
* Seen-subject filtering

The model and datasets are cached in memory so the API and dashboard do not repeatedly load the model for every request.

---

## 10. FastAPI REST API

The API is implemented in:

`src/api.py`

Start the API from the project root:

```text
python -m uvicorn src.api:app --reload
```

The API runs by default at:

`http://127.0.0.1:8000`

### Available Endpoints

#### Root

```text
GET /
```

Checks whether the service is running.

#### Health

```text
GET /health
```

Verifies that the recommendation model and datasets can be loaded.

#### Recommendations

```text
GET /recommendations/{student_id}
```

Returns personalized recommendations.

Example:

```text
GET /recommendations/STU-00007?top_n=5
```

#### Alternative Student Endpoint

```text
GET /students/{student_id}/recommendations
```

Swagger documentation is available at:

`http://127.0.0.1:8000/docs`

---

## 11. Streamlit Dashboard

The dashboard is implemented in:

`dashboard.py`

Start the dashboard from a second PowerShell window:

```text
streamlit run dashboard.py
```

The dashboard allows the user to:

1. Enter a student ID.
2. Select the number of recommendations.
3. Request recommendations from the FastAPI service.
4. View recommendation type.
5. View recommendation scores.
6. View confidence percentages.
7. View API response time.

The dashboard uses the already-trained model through the API instead of retraining the model during each request.

---

## 12. Dashboard Performance

The acceptance requirement is:

**Dashboard recommendation rendering should be under 1 second.**

Observed results after model and dataset caching:

| Scenario           | Student         | Type                   | Response Time |
| ------------------ | --------------- | ---------------------- | ------------: |
| Existing student   | STU-00007       | Hybrid                 |        221 ms |
| Cold-start student | NEW-STUDENT-001 | Content-Based Fallback |        148 ms |

Both observed response times are below the 1-second requirement.

---

## 13. Model Retraining Pipeline

The retraining pipeline is implemented in:

`src/retrain_model.py`

The pipeline:

1. Regenerates the processed interaction dataset.
2. Creates the evaluation split.
3. Trains a candidate SVD model.
4. Calculates NDCG@10.
5. Compares the candidate with the current best model.
6. Promotes the candidate only when it performs better.
7. Creates a backup of the current production model before promotion.
8. Records retraining history.

Retraining history:

`results/retraining_history.csv`

The latest tested candidate achieved:

**NDCG@10: 0.4669**

Current production score:

**NDCG@10: 0.4845**

Because the candidate performed worse, it was rejected and the production model remained unchanged.

Candidate model:

`models/svd_recommender_candidate.pkl`

---

## 14. A/B Test Design

The A/B test design is documented in:

`docs/ab_test_design.md`

The proposed experiment uses:

* Control: current production recommendation system
* Treatment: candidate recommendation system
* Traffic allocation: 50/50
* Fixed student assignment
* Primary metrics:

  * Recommendation click-through rate
  * Course/subject completion rate
* Secondary metrics:

  * Learning time
  * Lessons completed
  * Platform sessions
  * Quiz performance
  * Retention
  * Recommendation acceptance
  * Confidence

The treatment should only be promoted when it demonstrates meaningful improvement over the control while maintaining acceptable latency and reliability.

---

## 15. Model Card

The model card is available at:

`docs/model_card.md`

It documents:

* Model purpose
* Intended use
* Data
* Model architecture
* Evaluation
* Cold-start strategy
* Confidence scoring
* Limitations
* Ethical considerations
* Retraining
* A/B testing
* Future improvements

---

## 16. Project Structure

```text
EduTrack-LMS/
│
├── data/
│   ├── aiml_content_engagement.csv
│   ├── aiml_edtech_tests.json
│   ├── aiml_student_performance.csv
│   └── processed/
│       └── interactions.csv
│
├── docs/
│   ├── ab_test_design.md
│   └── model_card.md
│
├── models/
│   ├── svd_recommender.pkl
│   └── svd_recommender_candidate.pkl
│
├── results/
│   ├── retraining_history.csv
│   └── svd_experiments.csv
│
├── src/
│   ├── api.py
│   ├── data_pipeline.py
│   ├── recommender.py
│   ├── retrain_model.py
│   ├── train_model.py
│   └── tune_model.py
│
├── analyze_engagement.py
├── analyze_relationships.py
├── dashboard.py
├── inspect_data.py
├── requirem

