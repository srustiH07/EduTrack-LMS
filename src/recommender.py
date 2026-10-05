import pickle
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "svd_recommender.pkl"

INTERACTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "interactions.csv"
)

PERFORMANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "aiml_student_performance.csv"
)


# ---------------------------------------------------------
# Cached resources
# ---------------------------------------------------------

_MODEL = None
_INTERACTIONS = None
_PERFORMANCE = None
_SUBJECT_PROFILES = None


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

def load_model():
    """
    Load the trained Surprise SVD model once.

    The saved pickle contains a dictionary, so this function
    extracts the actual model object from that dictionary.
    """

    global _MODEL

    if _MODEL is None:

        with open(MODEL_PATH, "rb") as file:
            saved_object = pickle.load(file)

        # The training script saves a dictionary containing
        # the trained model.
        if isinstance(saved_object, dict):

            if "model" in saved_object:
                _MODEL = saved_object["model"]

            elif "svd_model" in saved_object:
                _MODEL = saved_object["svd_model"]

            else:
                raise ValueError(
                    "The saved model dictionary does not contain "
                    "'model' or 'svd_model'."
                )

        else:
            # Fallback if the pickle contains the model directly.
            _MODEL = saved_object

    return _MODEL


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

def load_datasets():
    """
    Load datasets once and cache them in memory.
    """

    global _INTERACTIONS
    global _PERFORMANCE
    global _SUBJECT_PROFILES

    if _INTERACTIONS is None:

        _INTERACTIONS = pd.read_csv(
            INTERACTIONS_PATH
        )

    if _PERFORMANCE is None:

        _PERFORMANCE = pd.read_csv(
            PERFORMANCE_PATH
        )

    if _SUBJECT_PROFILES is None:

        _SUBJECT_PROFILES = build_subject_profiles(
            _PERFORMANCE
        )

    return (
        _INTERACTIONS,
        _PERFORMANCE
    )


# ---------------------------------------------------------
# Normalize values
# ---------------------------------------------------------

def normalize_series(series):
    """
    Min-max normalize a pandas Series.
    """

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            1.0,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


# ---------------------------------------------------------
# Build content-based subject profiles
# ---------------------------------------------------------

def build_subject_profiles(performance):
    """
    Build subject-level content profiles using
    student performance information.
    """

    profile = (
        performance
        .groupby("subject")
        .agg(
            avg_post_test=(
                "post_test_score",
                "mean"
            ),
            avg_improvement=(
                "score_improvement",
                "mean"
            ),
            avg_completion=(
                "completion_pct",
                "mean"
            ),
            avg_quiz=(
                "best_quiz_score",
                "mean"
            ),
            avg_sessions=(
                "platform_sessions",
                "mean"
            ),
        )
        .reset_index()
    )

    numeric_columns = [
        "avg_post_test",
        "avg_improvement",
        "avg_completion",
        "avg_quiz",
        "avg_sessions",
    ]

    for column in numeric_columns:

        profile[column] = normalize_series(
            profile[column]
        )

    profile["content_score"] = (
        0.30 * profile["avg_post_test"]
        + 0.20 * profile["avg_improvement"]
        + 0.20 * profile["avg_completion"]
        + 0.20 * profile["avg_quiz"]
        + 0.10 * profile["avg_sessions"]
    )

    return profile


# ---------------------------------------------------------
# Get student profile
# ---------------------------------------------------------

def get_student_profile(
    student_id,
    performance
):
    """
    Return performance information for a student.
    """

    student_data = performance[
        performance["student_id"].astype(str)
        == str(student_id)
    ]

    if student_data.empty:

        return None

    return student_data.iloc[0]


# ---------------------------------------------------------
# Collaborative filtering scores
# ---------------------------------------------------------

def collaborative_scores(
    model,
    student_id,
    candidate_subjects
):
    """
    Generate collaborative-filtering predictions
    for candidate subjects.
    """

    scores = []

    for subject in candidate_subjects:

        prediction = model.predict(
            str(student_id),
            str(subject)
        )

        scores.append(
            (
                subject,
                float(prediction.est)
            )
        )

    return scores


# ---------------------------------------------------------
# Content-based / cold-start scores
# ---------------------------------------------------------

def cold_start_scores(
    student_id,
    performance,
    subject_profiles
):
    """
    Generate content-based recommendations.

    For a completely new student, overall subject profiles
    are used.
    """

    profile = get_student_profile(
        student_id,
        performance
    )

    # -----------------------------------------------------
    # Completely new student
    # -----------------------------------------------------

    if profile is None:

        ranked = subject_profiles[
            [
                "subject",
                "content_score"
            ]
        ].copy()

        ranked = ranked.sort_values(
            "content_score",
            ascending=False
        )

        return [
            (
                row["subject"],
                float(row["content_score"])
            )
            for _, row in ranked.iterrows()
        ]

    # -----------------------------------------------------
    # Existing student fallback
    # -----------------------------------------------------

    subject_profiles = subject_profiles.copy()

    subject_profiles[
        "student_match_score"
    ] = (
        0.30 * subject_profiles[
            "avg_post_test"
        ]
        + 0.20 * subject_profiles[
            "avg_improvement"
        ]
        + 0.20 * subject_profiles[
            "avg_completion"
        ]
        + 0.20 * subject_profiles[
            "avg_quiz"
        ]
        + 0.10 * subject_profiles[
            "avg_sessions"
        ]
    )

    ranked = subject_profiles.sort_values(
        "student_match_score",
        ascending=False
    )

    return [
        (
            row["subject"],
            float(
                row["student_match_score"]
            )
        )
        for _, row in ranked.iterrows()
    ]


# ---------------------------------------------------------
# Convert scores to confidence
# ---------------------------------------------------------

def scores_to_confidence(
    ranked_items
):
    """
    Convert recommendation scores into relative
    confidence values between 0 and 1.
    """

    if not ranked_items:

        return []

    scores = [
        score
        for _, score in ranked_items
    ]

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:

        return [
            (
                subject,
                score,
                1.0
            )
            for subject, score in ranked_items
        ]

    confidences = []

    for subject, score in ranked_items:

        confidence = (
            0.5
            + (
                0.5
                * (score - minimum)
                / (maximum - minimum)
            )
        )

        confidences.append(
            (
                subject,
                score,
                confidence
            )
        )

    return confidences


# ---------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------

def recommend(
    student_id,
    top_n=5
):
    """
    Generate personalized recommendations.

    Existing student:
        70% collaborative filtering
        30% content-based scoring

    New student:
        Content-based cold-start fallback
    """

    # Load cached resources
    model = load_model()

    (
        interactions,
        performance
    ) = load_datasets()

    subject_profiles = _SUBJECT_PROFILES

    student_id = str(student_id)

    # -----------------------------------------------------
    # Available subjects
    # -----------------------------------------------------

    all_subjects = sorted(
        interactions["subject"]
        .astype(str)
        .unique()
        .tolist()
    )

    # -----------------------------------------------------
    # Subjects already seen by student
    # -----------------------------------------------------

    seen_subjects = set(
        interactions[
            interactions[
                "student_id"
            ].astype(str)
            == student_id
        ]["subject"]
        .astype(str)
        .tolist()
    )

    candidate_subjects = [
        subject
        for subject in all_subjects
        if subject not in seen_subjects
    ]

    # -----------------------------------------------------
    # Determine whether student is known
    # -----------------------------------------------------

    known_students = set(
        interactions[
            "student_id"
        ].astype(str)
    )

    is_known_student = (
        student_id in known_students
    )

    # -----------------------------------------------------
    # Existing student: hybrid model
    # -----------------------------------------------------

    if is_known_student:

        cf_scores = collaborative_scores(
            model,
            student_id,
            candidate_subjects
        )

        content_scores = dict(
            cold_start_scores(
                student_id,
                performance,
                subject_profiles
            )
        )

        hybrid_scores = []

        for subject, cf_score in cf_scores:

            content_score = content_scores.get(
                subject,
                0.0
            )

            final_score = (
                0.70 * cf_score
                + 0.30 * content_score
            )

            hybrid_scores.append(
                (
                    subject,
                    final_score
                )
            )

        ranked_items = sorted(
            hybrid_scores,
            key=lambda item: item[1],
            reverse=True
        )

        recommendation_type = "hybrid"

    # -----------------------------------------------------
    # New student: content-based fallback
    # -----------------------------------------------------

    else:

        fallback_scores = cold_start_scores(
            student_id,
            performance,
            subject_profiles
        )

        ranked_items = [
            item
            for item in fallback_scores
            if item[0] in candidate_subjects
        ]

        recommendation_type = (
            "content-based fallback"
        )

    # -----------------------------------------------------
    # Select top N
    # -----------------------------------------------------

    ranked_items = ranked_items[:top_n]

    ranked_with_confidence = (
        scores_to_confidence(
            ranked_items
        )
    )

    # -----------------------------------------------------
    # Build API response
    # -----------------------------------------------------

    results = []

    for (
        subject,
        score,
        confidence
    ) in ranked_with_confidence:

        results.append(
            {
                "subject": subject,
                "score": round(
                    score,
                    4
                ),
                "confidence": round(
                    confidence,
                    4
                ),
            }
        )

    return {
        "student_id": student_id,
        "recommendation_type": (
            recommendation_type
        ),
        "recommendations": results,
    }