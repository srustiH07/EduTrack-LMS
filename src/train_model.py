from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from surprise import Dataset, Reader, SVD


# ============================================================
# EduTrack LMS - Collaborative Filtering Model
# ============================================================
#
# Purpose:
# Train and evaluate a matrix factorization recommendation
# model using student-subject interaction scores.
#
# Algorithm:
# Surprise SVD
#
# Evaluation:
# Leave-one-out evaluation for students with at least
# two subject interactions.
#
# Metric:
# NDCG@10
#
# Output:
# models/svd_recommender.pkl
#
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "interactions.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "svd_recommender.pkl"


# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

N_FACTORS = 50
N_EPOCHS = 30
LEARNING_RATE = 0.005
REG_ALL = 0.02
RANDOM_STATE = 42


# ------------------------------------------------------------
# Load processed interaction data
# ------------------------------------------------------------

def load_data():

    print("\nLoading processed interaction data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "student_id",
        "subject",
        "interaction_score",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print(f"Loaded interactions: {len(df)}")
    print(f"Students: {df['student_id'].nunique()}")
    print(f"Subjects: {df['subject'].nunique()}")

    return df


# ------------------------------------------------------------
# Create leave-one-out train/test split
# ------------------------------------------------------------

def create_train_test_split(df):

    print("\nCreating evaluation split...")

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    train_rows = []
    test_rows = []

    eligible_students = 0

    for student_id, group in df.groupby(
        "student_id"
    ):

        # Students with only one interaction cannot provide
        # a valid train/test collaborative-filtering example.
        if len(group) < 2:

            train_rows.extend(
                group.to_dict("records")
            )

            continue

        eligible_students += 1

        # Randomly hold one subject interaction out
        # for evaluation.
        test_index = rng.choice(
            group.index
        )

        for index, row in group.iterrows():

            if index == test_index:

                test_rows.append(
                    row.to_dict()
                )

            else:

                train_rows.append(
                    row.to_dict()
                )

    train_df = pd.DataFrame(
        train_rows
    )

    test_df = pd.DataFrame(
        test_rows
    )

    print(
        f"Students eligible for CF evaluation: "
        f"{eligible_students}"
    )

    print(
        f"Training interactions: "
        f"{len(train_df)}"
    )

    print(
        f"Test interactions: "
        f"{len(test_df)}"
    )

    if test_df.empty:

        raise ValueError(
            "No valid test interactions were created."
        )

    return train_df, test_df


# ------------------------------------------------------------
# Train SVD model
# ------------------------------------------------------------

def train_model(train_df):

    print(
        "\nTraining SVD matrix factorization model..."
    )

    reader = Reader(
        rating_scale=(0, 1)
    )

    dataset = Dataset.load_from_df(
        train_df[
            [
                "student_id",
                "subject",
                "interaction_score",
            ]
        ],
        reader
    )

    trainset = dataset.build_full_trainset()

    model = SVD(
        n_factors=N_FACTORS,
        n_epochs=N_EPOCHS,
        lr_all=LEARNING_RATE,
        reg_all=REG_ALL,
        random_state=RANDOM_STATE,
    )

    model.fit(trainset)

    print(
        "SVD model training completed."
    )

    return model


# ------------------------------------------------------------
# Calculate Discounted Cumulative Gain
# ------------------------------------------------------------

def dcg_at_k(
    relevance_scores,
    k=10
):

    relevance_scores = (
        relevance_scores[:k]
    )

    if len(relevance_scores) == 0:

        return 0.0

    dcg = 0.0

    for position, relevance in enumerate(
        relevance_scores,
        start=1
    ):

        dcg += (
            (2 ** relevance - 1)
            / np.log2(position + 1)
        )

    return dcg


# ------------------------------------------------------------
# Calculate NDCG@10
# ------------------------------------------------------------

def calculate_ndcg(
    actual_subject,
    ranked_subjects,
    k=10
):

    relevance_scores = [
        1 if subject == actual_subject
        else 0
        for subject in ranked_subjects
    ]

    dcg = dcg_at_k(
        relevance_scores,
        k
    )

    ideal_relevance = [1]

    idcg = dcg_at_k(
        ideal_relevance,
        k
    )

    if idcg == 0:

        return 0.0

    return dcg / idcg


# ------------------------------------------------------------
# Evaluate model using NDCG@10
# ------------------------------------------------------------

def evaluate_model(
    model,
    train_df,
    test_df
):

    print(
        "\nEvaluating model with NDCG@10..."
    )

    all_subjects = sorted(
        train_df["subject"]
        .unique()
        .tolist()
    )

    ndcg_scores = []

    ranking_records = []

    for _, test_row in test_df.iterrows():

        student_id = test_row[
            "student_id"
        ]

        actual_subject = test_row[
            "subject"
        ]

        # Subjects already seen by this student
        # during training.
        seen_subjects = set(
            train_df.loc[
                train_df["student_id"]
                == student_id,
                "subject"
            ]
        )

        # Recommend only subjects that the student
        # has not already interacted with.
        candidate_subjects = [
            subject
            for subject in all_subjects
            if subject not in seen_subjects
        ]

        # The held-out subject must be available
        # as a candidate.
        if actual_subject not in candidate_subjects:

            continue

        predictions = []

        for subject in candidate_subjects:

            prediction = model.predict(
                student_id,
                subject
            )

            predictions.append(
                (
                    subject,
                    prediction.est
                )
            )

        # Highest predicted score first.
        predictions.sort(
            key=lambda item: item[1],
            reverse=True
        )

        ranked_subjects = [
            subject
            for subject, score
            in predictions
        ]

        score = calculate_ndcg(
            actual_subject,
            ranked_subjects,
            k=10
        )

        ndcg_scores.append(
            score
        )

        ranking_records.append(
            {
                "student_id": student_id,
                "actual_subject": actual_subject,
                "top_recommendation": (
                    ranked_subjects[0]
                    if ranked_subjects
                    else ""
                ),
                "ndcg_at_10": round(
                    score,
                    4
                ),
            }
        )

    if not ndcg_scores:

        raise ValueError(
            "No valid test cases were available "
            "for NDCG@10 evaluation."
        )

    mean_ndcg = float(
        np.mean(ndcg_scores)
    )

    results_df = pd.DataFrame(
        ranking_records
    )

    print(
        f"Evaluated students: "
        f"{len(ndcg_scores)}"
    )

    print(
        f"NDCG@10: {mean_ndcg:.4f}"
    )

    print(
        "\nSample evaluation results:"
    )

    print(
        results_df
        .head(10)
        .to_string(index=False)
    )

    return mean_ndcg, results_df


# ------------------------------------------------------------
# Train final model using all available interactions
# ------------------------------------------------------------

def train_final_model(df):

    print(
        "\nTraining final SVD model "
        "using all available interactions..."
    )

    reader = Reader(
        rating_scale=(0, 1)
    )

    dataset = Dataset.load_from_df(
        df[
            [
                "student_id",
                "subject",
                "interaction_score",
            ]
        ],
        reader
    )

    trainset = dataset.build_full_trainset()

    model = SVD(
        n_factors=N_FACTORS,
        n_epochs=N_EPOCHS,
        lr_all=LEARNING_RATE,
        reg_all=REG_ALL,
        random_state=RANDOM_STATE,
    )

    model.fit(trainset)

    print(
        "Final model training completed."
    )

    return model


# ------------------------------------------------------------
# Save final model
# ------------------------------------------------------------

def save_model(
    model,
    ndcg_score,
    subjects
):

    print(
        "\nSaving final model..."
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_data = {
        "model": model,
        "ndcg_at_10": ndcg_score,
        "n_factors": N_FACTORS,
        "n_epochs": N_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "reg_all": REG_ALL,
        "random_state": RANDOM_STATE,
        "subjects": subjects,
    }

    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            model_data,
            file
        )

    print(
        f"Model saved to: {MODEL_PATH}"
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print(
        "EDUTRACK LMS - MATRIX FACTORIZATION"
    )
    print("=" * 70)

    # 1. Load processed data.
    df = load_data()

    # 2. Create evaluation split.
    train_df, test_df = (
        create_train_test_split(df)
    )

    # 3. Train evaluation model.
    evaluation_model = train_model(
        train_df
    )

    # 4. Evaluate using NDCG@10.
    ndcg_score, results_df = (
        evaluate_model(
            evaluation_model,
            train_df,
            test_df
        )
    )

    # 5. Train final model using all data.
    final_model = train_final_model(
        df
    )

    # 6. Store the complete subject vocabulary.
    subjects = sorted(
        df["subject"]
        .unique()
        .tolist()
    )

    # 7. Save final model.
    save_model(
        final_model,
        ndcg_score,
        subjects
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("MODEL SUMMARY")
    print("-" * 70)

    print(
        "Algorithm: Surprise SVD"
    )

    print(
        f"Factors: {N_FACTORS}"
    )

    print(
        f"Epochs: {N_EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Regularization: {REG_ALL}"
    )

    print(
        f"NDCG@10: {ndcg_score:.4f}"
    )

    if ndcg_score >= 0.55:

        print(
            "\nAcceptance target: PASSED "
            "(NDCG@10 >= 0.55)"
        )

    else:

        print(
            "\nAcceptance target: NOT YET MET "
            "(NDCG@10 < 0.55)"
        )

    print(
        "\nModel file:"
    )

    print(
        MODEL_PATH
    )

    print("\n" + "=" * 70)
    print(
        "MATRIX FACTORIZATION COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()