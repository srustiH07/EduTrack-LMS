# EduTrack LMS - Adaptive Learning Path Recommendation System

## 1. Project Overview

EduTrack LMS is an adaptive learning path recommendation system designed for an EdTech platform.

The system recommends the next learning subjects for students using collaborative filtering with a content-based fallback for students who do not have sufficient interaction history.

The project provides:

- Data processing pipeline
- Matrix-factorization recommendation model
- Hybrid recommendation engine
- Cold-start content-based fallback
- FastAPI REST API
- Streamlit dashboard
- Offline NDCG@10 evaluation
- Hyperparameter experimentation
- Model retraining and promotion pipeline
- A/B test design

---

## 2. Problem Statement

Students on an online learning platform may have different learning interests, engagement levels, and subject histories.

A static learning sequence may not provide personalized recommendations.

EduTrack LMS addresses this problem by using student interaction data to recommend relevant subjects while providing a fallback strategy for new students.

---

## 3. System Architecture

```text
                    EduTrack LMS
                         |
                         v
              Student-Course Interactions
                         |
                         v
                 Data Processing
                         |
                         v
              Matrix Factorization
                  SVD Recommender
                         |
                         v
              Recommendation Engine
                    /          \
                   /            \
          Existing Student    New Student
                |                  |
                v                  v
          Collaborative       Content-Based
              + Content          Fallback
                \                  /
                 \                /
                  v              v
                 Top 5 Recommendations
                         |
              +----------+----------+
              |                     |
              v                     v
          FastAPI API        Streamlit Dashboard