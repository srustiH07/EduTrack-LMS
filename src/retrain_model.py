from pathlib import Path
import pickle
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD


# -------------------------------------------------------------------
# Project path setup
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

INTERACTIONS_FILE = PROCESSED_DIR / "interactions.csv"
CURRENT_MODEL_FILE = MODELS_DIR / "svd_recommender.pkl"
CANDIDATE_MODEL_FILE = MODELS_DIR / "svd_recommender_candidate.pkl"
HISTORY_FILE = RESULTS_DIR / "retraining_history.csv"


# -------------------------------------------------------------------
# Step 1: Run existing data pipeline
# -------------------------------------------------------------------

def run_data_pipeline():
    print("\n[1/6] Running data pipeline...")

    from src.data_pipeline import main as data_pipeline_main

    data_pipeline_main()

    if not INTERACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Processed interaction file was not created: "
            f"{INTERACTIONS_FILE}"
        )

    interactions = pd.read_csv(INTERACTIONS_FILE)

    print(f"Processed interactions: {len(interactions)}")

    return interactions


# -------------------------------------------------------------------
# Step 2: Create leave-one-out evaluation split
# -------------------------------------------------------------------

def create_leave_one_out_split(interactions):
    print("\n[2/6] Creating evaluation split...")

    rng = np.random.RandomState(42)

    train_rows = []
    test_rows = []

    for student_id, group in interactions.groupby("student_id"):

        if len(group) >= 2:

            test_index = rng.choice(group.index)

            for index, row in group.iterrows():

                if index == test_index:
                    test_rows.append(row.to_dict())
                else:
                    train_rows.append(row.to_dict())

        else:
            train_rows.extend(group.to_dict("records"))

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    print(f"Training interactions: {len(train_df)}")
    print(f"Test interactions: {len(test_df)}")
    print(f"Eligible students: {len(test_df)}")

    return train_df, test_df


# -------------------------------------------------------------------
# Step 3: Train candidate model
# -------------------------------------------------------------------

def train_candidate(train_df):
    print("\n[3/6] Training candidate SVD model...")

    reader = Reader(rating_scale=(0, 1))

    train_data = Dataset.load_from_df(
        train_df[
            ["student_id", "subject", "interaction_score"]
        ],
        reader
    )

    trainset = train_data.build_full_trainset()

    model = SVD(
        n_factors=20,
        n_epochs=30,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )

    model.fit(trainset)

    print("Candidate model trained.")

    return model


# -------------------------------------------------------------------
# Step 4: Calculate NDCG@10
# -------------------------------------------------------------------

def calculate_ndcg_at_10(
    model,
    train_df,
    test_df,
    all_subjects
):
    print("\n[4/6] Evaluating candidate model...")

    ndcg_scores = []

    train_by_student = {
        student_id: set(group["subject"])
        for student_id, group in train_df.groupby("student_id")
    }

    for _, row in test_df.iterrows():

        student_id = row["student_id"]
        actual_subject = row["subject"]

        seen_subjects = train_by_student.get(
            student_id,
            set()
        )

        candidate_subjects = [
            subject
            for subject in all_subjects
            if subject not in seen_subjects
        ]

        predictions = []

        for subject in candidate_subjects:

            prediction = model.predict(
                student_id,
                subject
            )

            predictions.append(
                (
                    subject,
                    float(prediction.est)
                )
            )

        predictions.sort(
            key=lambda item: item[1],
            reverse=True
        )

        top_10 = [
            subject
            for subject, _ in predictions[:10]
        ]

        if actual_subject in top_10:

            rank = top_10.index(actual_subject) + 1

            ndcg = 1.0 / np.log2(rank + 1)

        else:
            ndcg = 0.0

        ndcg_scores.append(ndcg)

    score = (
        float(np.mean(ndcg_scores))
        if ndcg_scores
        else 0.0
    )

    print(f"Candidate NDCG@10: {score:.4f}")

    return score


# -------------------------------------------------------------------
# Get current model score
# -------------------------------------------------------------------

def get_current_model_score():

    experiments_file = (
        RESULTS_DIR / "svd_experiments.csv"
    )

    if experiments_file.exists():

        experiments = pd.read_csv(
            experiments_file
        )

        if "ndcg_at_10" in experiments.columns:

            return float(
                experiments["ndcg_at_10"].max()
            )

    if HISTORY_FILE.exists():

        history = pd.read_csv(
            HISTORY_FILE
        )

        if (
            "candidate_ndcg_at_10" in history.columns
            and len(history) > 0
        ):

            promoted = history[
                history["status"]
                .astype(str)
                .str.upper()
                == "PROMOTED"
            ]

            if not promoted.empty:

                return float(
                    promoted.iloc[-1][
                        "candidate_ndcg_at_10"
                    ]
                )

    return 0.0


# -------------------------------------------------------------------
# Save candidate model
# -------------------------------------------------------------------

def save_candidate_model(
    model,
    interactions
):

    print("\n[5/6] Saving candidate model...")

    model_package = {
        "model": model,
        "subjects": sorted(
            interactions["subject"]
            .unique()
            .tolist()
        ),
        "trained_at": datetime.now().isoformat(),
        "model_type": "Surprise SVD",
        "n_factors": 20,
        "n_epochs": 30,
        "lr_all": 0.005,
        "reg_all": 0.02
    }

    with open(
        CANDIDATE_MODEL_FILE,
        "wb"
    ) as file:

        pickle.dump(
            model_package,
            file
        )

    print(
        f"Candidate saved to: "
        f"{CANDIDATE_MODEL_FILE}"
    )


# -------------------------------------------------------------------
# Promote candidate model
# -------------------------------------------------------------------

def promote_candidate(
    model,
    interactions
):

    package = {
        "model": model,
        "subjects": sorted(
            interactions["subject"]
            .unique()
            .tolist()
        ),
        "trained_at": datetime.now().isoformat(),
        "model_type": "Surprise SVD",
        "n_factors": 20,
        "n_epochs": 30,
        "lr_all": 0.005,
        "reg_all": 0.02
    }

    backup_file = (
        MODELS_DIR /
        "svd_recommender_backup.pkl"
    )

    if CURRENT_MODEL_FILE.exists():

        shutil.copy2(
            CURRENT_MODEL_FILE,
            backup_file
        )

        print(
            f"Previous model backed up to: "
            f"{backup_file}"
        )

    with open(
        CURRENT_MODEL_FILE,
        "wb"
    ) as file:

        pickle.dump(
            package,
            file
        )

    print(
        "Candidate promoted to current model."
    )


# -------------------------------------------------------------------
# Record retraining history
# -------------------------------------------------------------------

def record_history(
    candidate_score,
    current_score,
    status
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    row = pd.DataFrame(
        [{
            "timestamp":
                datetime.now().isoformat(),

            "current_ndcg_at_10":
                current_score,

            "candidate_ndcg_at_10":
                candidate_score,

            "target_ndcg_at_10":
                0.55,

            "status":
                status
        }]
    )

    if HISTORY_FILE.exists():

        row.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        row.to_csv(
            HISTORY_FILE,
            index=False
        )

    print(
        f"Retraining history updated: "
        f"{HISTORY_FILE}"
    )


# -------------------------------------------------------------------
# Main retraining workflow
# -------------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "EduTrack LMS - Model Retraining Pipeline"
    )
    print("=" * 70)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Step 1
    interactions = run_data_pipeline()

    # Step 2
    train_df, test_df = (
        create_leave_one_out_split(
            interactions
        )
    )

    # Step 3
    model = train_candidate(
        train_df
    )

    all_subjects = sorted(
        interactions["subject"]
        .unique()
        .tolist()
    )

    # Step 4
    candidate_score = (
        calculate_ndcg_at_10(
            model,
            train_df,
            test_df,
            all_subjects
        )
    )

    current_score = (
        get_current_model_score()
    )

    # Save candidate separately
    save_candidate_model(
        model,
        interactions
    )

    # Step 5/6
    print(
        "\n[6/6] Comparing candidate "
        "with current model..."
    )

    print(
        f"Current model NDCG@10:   "
        f"{current_score:.4f}"
    )

    print(
        f"Candidate model NDCG@10: "
        f"{candidate_score:.4f}"
    )

    if candidate_score > current_score:

        promote_candidate(
            model,
            interactions
        )

        status = "PROMOTED"

        print(
            "\nCandidate is better."
        )

        print(
            "Status: PROMOTED"
        )

    else:

        status = "REJECTED"

        print(
            "\nCandidate did not improve "
            "the current model."
        )

        print(
            "Current production model "
            "remains unchanged."
        )

        print(
            "Status: REJECTED"
        )

    record_history(
        candidate_score,
        current_score,
        status
    )

    print("\n" + "=" * 70)

    print(
        "Retraining pipeline completed."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()