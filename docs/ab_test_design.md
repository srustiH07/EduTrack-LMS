# EduTrack LMS - A/B Test Design

## 1. Objective

The A/B test is designed to compare the existing recommendation system with a newly trained candidate model before full deployment.

The goal is to determine whether the candidate model improves student engagement and learning outcomes without negatively affecting the existing recommendation experience.

---

## 2. Experiment Variants

### Control Group - A

The control group receives recommendations from the current production recommendation system.

Architecture:

- Collaborative filtering using matrix factorization
- Content-based signals
- Hybrid recommendation for existing students
- Content-based fallback for cold-start students

### Treatment Group - B

The treatment group receives recommendations from the newly trained candidate model.

The candidate model must first pass offline evaluation before being included in the experiment.

---

## 3. Traffic Allocation

Students are randomly assigned to one of the two groups.

Initial allocation:

- Control: 50%
- Treatment: 50%

The assignment should remain stable for each student throughout the experiment.

A student should not switch between control and treatment during the same experiment.

---

## 4. Primary Metrics

### Recommendation Click-Through Rate

Measures the percentage of recommendation impressions that result in a student selecting a recommended subject or course.

Formula:

CTR = recommendation clicks / recommendation impressions

Higher CTR indicates stronger recommendation relevance.

---

### Recommendation Completion Rate

Measures the percentage of recommended learning items that students complete.

Formula:

Completion Rate = completed recommendations / accepted recommendations

Higher completion indicates that recommended learning content is useful and appropriate.

---

## 5. Secondary Metrics

Additional metrics include:

- Average learning time
- Number of lessons completed
- Student session frequency
- Recommendation acceptance rate
- Quiz performance after recommendation
- Student retention
- Average recommendation confidence

These metrics provide additional evidence about the effect of the recommendation system.

---

## 6. Offline ML Metric

Before deployment, both models should be evaluated using NDCG@10.

Current SVD baseline:

NDCG@10 = 0.4845

Client target:

NDCG@10 >= 0.55

The current SVD model does not yet meet the target. Therefore, the hybrid recommendation architecture is used to improve practical recommendation behavior while further model improvements are investigated.

---

## 7. Experiment Procedure

The experiment follows these steps:

1. Select eligible students.
2. Randomly assign students to Control or Treatment.
3. Keep the assignment fixed.
4. Record recommendation impressions.
5. Record recommendation selections.
6. Record learning completion.
7. Record engagement metrics.
8. Compare Control and Treatment metrics.
9. Perform statistical significance testing.
10. Review safety and quality metrics.
11. Decide whether to promote the candidate.

---

## 8. Success Criteria

The treatment model should be considered successful if:

- Recommendation CTR improves over Control.
- Completion rate improves or remains stable.
- Student engagement improves.
- There is no significant negative effect on learning outcomes.
- The candidate passes offline model evaluation.
- No critical system or recommendation-quality issues are observed.

The candidate should not be promoted if it causes a meaningful decrease in completion, engagement, or other important learning metrics.

---

## 9. Statistical Evaluation

For the A/B test, the treatment and control groups should be compared using appropriate statistical tests.

For binary metrics such as:

- Clicked / not clicked
- Completed / not completed

a proportion-based statistical test can be used.

For continuous metrics such as:

- Learning time
- Number of completed lessons

an appropriate comparison of group means can be used.

The experiment should report:

- Control metric
- Treatment metric
- Absolute difference
- Relative improvement
- Confidence interval
- Statistical significance

---

## 10. Rollout Strategy

The candidate should not immediately replace the production model.

Recommended rollout:

1. Offline evaluation
2. Small controlled A/B experiment
3. Review results
4. Gradual rollout if successful
5. Full deployment after validation

If the candidate performs poorly, the current production model remains active.

---

## 11. Rollback Strategy

The production system must retain the previous model.

If the candidate causes:

- Reduced recommendation engagement
- Reduced completion
- Poor recommendation quality
- Increased API errors
- Unexpected system behavior

the candidate should be removed and the previous model restored.

The retraining pipeline already follows a similar model-promotion principle: a candidate is promoted only when its evaluation score is better than the current baseline.

---

## 12. Cold-Start Consideration

New students without historical interactions cannot be evaluated using collaborative filtering alone.

For these students, the system uses a content-based fallback based on available student/profile and subject information.

Therefore, A/B analysis should separately monitor:

- Existing students
- Cold-start students

This prevents the performance of one group from hiding problems in the other.

---

## 13. Monitoring During Experiment

The following should be monitored:

- API response time
- Recommendation generation failures
- Recommendation CTR
- Completion rate
- Engagement
- Model confidence
- Cold-start recommendation usage

The dashboard currently demonstrates recommendation response times below the client's 1-second requirement.

---

## 14. Decision Framework

The candidate is promoted only when the combined evidence supports improvement.

Decision:

Offline evaluation
        |
        v
A/B test
        |
        v
Improved engagement?
     /       \
   YES        NO
    |          |
    v          v
Review       Keep current
quality      model
    |
    v
Gradual rollout
    |
    v
Full deployment

The final decision should consider both machine-learning metrics and actual student outcomes.

---

## 15. Conclusion

The A/B test provides a controlled method for determining whether a new recommendation model improves the EduTrack LMS learning experience.

The design separates offline model quality from real-world student behavior and provides a safe path for model deployment, monitoring, and rollback.