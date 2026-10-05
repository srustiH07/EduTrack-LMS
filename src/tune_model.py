from pathlib import Path

import numpy as np
import pandas as pd

from surprise import Dataset, Reader, SVD


# ============================================================
# EduTrack LMS - SVD Model Tuning
# ============================================================
#
# Purpose:
# Experiment with multiple matrix factorization
# configurations and select the model with the
# highest NDCG@10 score.
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

RESULTS_DIR = BASE_DIR / "results"

RESULTS_PATH = (
    RESULTS_DIR
    / "svd_experiments.csv"
)


# ------------------------------------------------------------
# Experiment configurations
# ------------------------------------------------------------

EXPERIMENTS = [
    {
        "name": "baseline",
        "n_factors": 50,
        "n_epochs": 30,
        "lr_all": 0.005,
        "reg_all": 0.02,
    },
    {
        "name": "factors_20",
        "n_factors": 20,
        "n_epochs": 30,
        "lr_all": 0.005,
        "reg_all": 0.02,
    },
    {
        "name": "factors_100",
        "n_factors": 100,
        "n_epochs": 30,
        "lr_all": 0.005,
        "reg_all": 0.02,
    },
    {
        "name": "epochs_60",
        "n_factors": 50,
        "n_epochs": 60,
        "lr_all": 0.005,
        "reg_all": 0.02,
    },
    {
        "name": "higher_learning_rate",
        "n_factors": 50,
        "n_epochs": 30,
        "lr_all": 0.01,
        "reg_all": 0.02,
    },
    {
        "name": "lower_regularization",
        "n_factors": 50,
        "n_epochs": 30,
        "lr_all": 0.005,
        "reg_all": 0.01,
    },
    {
        "name": "higher_regularization",
        "n_factors": 50,
        "n_epochs": 30,
        "lr_all": 0.005,
        "reg_all": 0.05,
    },
    {
        "name": "combined_tuned",
        "n_factors": 100,
        "n_epochs": 60,
        "lr_all": 0.01,
        "reg_all": 0.01,
    },
]


RANDOM_STATE = 42


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_data():

    print("\nLoading processed interaction data...")

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Loaded interactions: {len(df)}"
    )

    return df


# ------------------------------------------------------------
# Create evaluation split
# ------------------------------------------------------------

def create_train_test_split(df):

    print(
        "\nCreating consistent evaluation split..."
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    train_rows = []
    test_rows = []

    for student_id, group in df.groupby(
        "student_id"
    ):

        # Students with only one interaction remain
        # in training data because they cannot be
        # evaluated with leave-one-out testing.
        if len(group) < 2:

            train_rows.extend(
                group.to_dict("records")
            )

            continue

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
        f"Training interactions: {len(train_df)}"
    )

    print(
        f"Test interactions: {len(test_df)}"
    )

    return train_df, test_df


# ------------------------------------------------------------
# Train one SVD configuration
# ------------------------------------------------------------

def train_svd(
    train_df,
    configuration
):

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
        n_factors=configuration[
            "n_factors"
        ],
        n_epochs=configuration[
            "n_epochs"
        ],
        lr_all=configuration[
            "lr_all"
        ],
        reg_all=configuration[
            "reg_all"
        ],
        random_state=RANDOM_STATE,
    )

    model.fit(
        trainset
    )

    return model


# ------------------------------------------------------------
# DCG
# ------------------------------------------------------------

def dcg_at_k(
    relevance_scores,
    k=10
):

    relevance_scores = (
        relevance_scores[:k]
    )

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
# NDCG@10
# ------------------------------------------------------------

def calculate_ndcg(
    actual_subject,
    ranked_subjects
):

    relevance_scores = [
        1 if subject == actual_subject
        else 0
        for subject in ranked_subjects
    ]

    dcg = dcg_at_k(
        relevance_scores,
        10
    )

    idcg = dcg_at_k(
        [1],
        10
    )

    if idcg == 0:

        return 0.0

    return dcg / idcg


# ------------------------------------------------------------
# Evaluate model
# ------------------------------------------------------------

def evaluate_model(
    model,
    train_df,
    test_df
):

    all_subjects = sorted(
        train_df["subject"]
        .unique()
        .tolist()
    )

    scores = []

    for _, test_row in test_df.iterrows():

        student_id = test_row[
            "student_id"
        ]

        actual_subject = test_row[
            "subject"
        ]

        seen_subjects = set(
            train_df.loc[
                train_df["student_id"]
                == student_id,
                "subject"
            ]
        )

        candidate_subjects = [
            subject
            for subject in all_subjects
            if subject not in seen_subjects
        ]

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
            ranked_subjects
        )

        scores.append(
            score
        )

    if not scores:

        return 0.0

    return float(
        np.mean(scores)
    )


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print(
        "EDUTRACK LMS - SVD MODEL EXPERIMENTATION"
    )
    print("=" * 70)

    df = load_data()

    train_df, test_df = (
        create_train_test_split(df)
    )

    experiment_results = []

    print(
        "\nRunning SVD experiments..."
    )

    print("-" * 70)

    for number, configuration in enumerate(
        EXPERIMENTS,
        start=1
    ):

        print(
            f"\nExperiment {number}/"
            f"{len(EXPERIMENTS)}: "
            f"{configuration['name']}"
        )

        print(
            f"Factors={configuration['n_factors']}, "
            f"Epochs={configuration['n_epochs']}, "
            f"LR={configuration['lr_all']}, "
            f"Reg={configuration['reg_all']}"
        )

        model = train_svd(
            train_df,
            configuration
        )

        ndcg = evaluate_model(
            model,
            train_df,
            test_df
        )

        print(
            f"NDCG@10 = {ndcg:.4f}"
        )

        experiment_results.append(
            {
                "experiment": configuration[
                    "name"
                ],
                "n_factors": configuration[
                    "n_factors"
                ],
                "n_epochs": configuration[
                    "n_epochs"
                ],
                "learning_rate": configuration[
                    "lr_all"
                ],
                "regularization": configuration[
                    "reg_all"
                ],
                "ndcg_at_10": round(
                    ndcg,
                    4
                ),
            }
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        experiment_results
    )

    results_df = results_df.sort_values(
        "ndcg_at_10",
        ascending=False
    ).reset_index(
        drop=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    best = results_df.iloc[0]

    print("\n" + "-" * 70)
    print("BEST CONFIGURATION")
    print("-" * 70)

    print(
        f"Experiment: {best['experiment']}"
    )

    print(
        f"NDCG@10: {best['ndcg_at_10']:.4f}"
    )

    print(
        f"Factors: {int(best['n_factors'])}"
    )

    print(
        f"Epochs: {int(best['n_epochs'])}"
    )

    print(
        f"Learning rate: {best['learning_rate']}"
    )

    print(
        f"Regularization: {best['regularization']}"
    )

    print(
        f"\nExperiment results saved to:"
    )

    print(
        RESULTS_PATH
    )

    if best["ndcg_at_10"] >= 0.55:

        print(
            "\nAcceptance target: PASSED"
        )

    else:

        print(
            "\nAcceptance target: NOT YET MET"
        )

    print("\n" + "=" * 70)
    print(
        "SVD EXPERIMENTATION COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()