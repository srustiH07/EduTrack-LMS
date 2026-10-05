# EduTrack LMS Model Card

## 1. Model Overview

Model name: EduTrack LMS Adaptive Recommendation System

Model type: Hybrid recommendation system

Primary collaborative filtering model: Matrix Factorization using Surprise SVD

Recommendation modes:
- Hybrid recommendation for students with interaction history
- Content-based fallback for cold-start students

The system recommends the next learning subjects for students based on historical learning interactions and student performance characteristics.

## 2. Intended Use

The model is intended to support adaptive learning in an EdTech Learning Management System.

Primary use cases:
- Recommend the next learning subjects for a student
- Personalize learning paths
- Provide ranked recommendations through a REST API
- Display recommendations through a Streamlit dashboard

The model is designed as a decision-support system and should complement, not replace, educational judgment.

## 3. Data Used

The project uses the following provided datasets:

- aiml_student_performance.csv
- aiml_content_engagement.csv
- aiml_edtech_tests.json

The content engagement dataset contains student-subject interaction information and is the primary dataset used for collaborative filtering.

The student performance dataset provides additional profile information for the content-based component.

The provided test JSON contains educational risk/test information and is not used as direct recommendation interaction data.

## 4. Data Processing

The content engagement records are cleaned and aggregated by student and subject.

The interaction score combines:

- Engagement score
- Completion
- Likes
- Replays
- Notes taken
- Normalized time spent
- Normalized difficulty

The resulting processed dataset contains:

- 600 raw engagement records
- 588 processed student-subject interactions
- 459 unique students
- 10 subjects

## 5. Collaborative Filtering Model

The collaborative filtering component uses Singular Value Decomposition (SVD).

The baseline configuration was:

- Factors: 50
- Epochs: 30
- Learning rate: 0.005
- Regularization: 0.02
- Random state: 42

Hyperparameter experimentation tested different factor counts, training epochs, learning rates, and regularization values.

The best tested SVD configuration used:

- Factors: 20
- Epochs: 30
- Learning rate: 0.005
- Regularization: 0.02

## 6. Evaluation

The evaluation uses leave-one-out testing for students with at least two interactions.

Evaluation results:

- Eligible students: 110
- Training interactions: 478
- Test interactions: 110
- Best SVD NDCG@10: 0.4845

Target:

NDCG@10 >= 0.55

The current offline SVD result does not yet meet the target. This limitation is documented rather than hidden.

## 7. Hybrid Recommendation Strategy

For known students, the recommendation engine combines:

- Collaborative filtering score: 70%
- Content-based score: 30%

Previously seen subjects are filtered from the recommendation candidates.

For students without sufficient interaction history, the system uses a content-based fallback.

This allows the system to handle cold-start users.

## 8. Confidence Scores

The dashboard displays normalized confidence scores for ranked recommendations.

Confidence is intended as a relative ranking confidence indicator rather than a calibrated probability.

## 9. API Deployment

The recommendation engine is exposed through FastAPI.

Main endpoints:

- GET /
- GET /health
- GET /recommendations/{student_id}
- GET /students/{student_id}/recommendations

The API loads the trained model and cached datasets rather than retraining during each request.

## 10. Dashboard

A Streamlit dashboard provides:

- Student ID input
- Recommendation count selection
- Recommendation type
- API response time
- Recommended subjects
- Recommendation scores
- Confidence scores

Observed response times during testing were below the 1-second dashboard requirement.

## 11. Cold-Start Testing

A new student without historical interactions can receive recommendations through the content-based fallback.

Example test:

Student ID:
NEW-STUDENT-001

Recommendation type:
Content-Based Fallback

Observed API response time:
148 ms

## 12. Retraining Pipeline

The project includes an automated retraining script.

The pipeline:

1. Reprocesses the latest interaction data.
2. Creates a training/test split.
3. Trains a candidate SVD model.
4. Evaluates NDCG@10.
5. Compares the candidate with the current best score.
6. Promotes the candidate only when it performs better.
7. Keeps the current production model when the candidate is worse.
8. Records retraining results in retraining_history.csv.

A tested candidate achieved NDCG@10 of 0.4669 and was rejected because the current best score was 0.4845.

## 13. A/B Testing

The project includes an A/B test design.

Control:
Current production hybrid recommendation system.

Treatment:
New candidate recommendation model.

Traffic allocation:
50% control and 50% treatment.

Primary business metrics:
- Recommendation click-through rate
- Course/subject completion rate

Secondary metrics:
- Learning time
- Platform sessions
- Quiz performance
- Retention
- Recommendation acceptance

The experiment should use fixed student assignment to prevent users from switching groups during the experiment.

## 14. Limitations

The current datasets contain subject-level interactions rather than individual course IDs.

Therefore, the current dashboard recommends learning subjects instead of specific courses.

The available interaction data is relatively sparse.

The current best offline SVD NDCG@10 of 0.4845 is below the project target of 0.55.

Confidence scores are ranking indicators and are not calibrated probabilities.

The content-based fallback depends on the quality and availability of student profile information.

## 15. Ethical and Responsible Use

Recommendations should not unfairly restrict a student's learning options.

The system should be monitored for biased recommendation patterns across student groups.

Student data should be handled securely and only used for legitimate educational purposes.

Human educators should be able to review and override recommendations when necessary.

## 16. Future Improvements

Potential improvements include:

- More detailed course-level interaction data
- Item/content embeddings
- Neural collaborative filtering
- Learning-to-rank models
- More advanced hybrid weighting
- Better cold-start features
- Calibrated confidence estimation
- Online A/B testing
- Automated model monitoring
- Drift detection
- Scheduled production retraining

## 17. Production Status

Current status:

- Data pipeline: Complete
- SVD model: Complete
- Hyperparameter experimentation: Complete
- Hybrid recommendation engine: Complete
- Cold-start fallback: Complete
- FastAPI integration: Complete
- Streamlit dashboard: Complete
- Retraining pipeline: Complete
- A/B test design: Complete
- Model card: Complete
- NDCG@10 target: Not yet achieved