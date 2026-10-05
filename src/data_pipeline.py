from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# EduTrack LMS - Data Pipeline
# ============================================================
#
# Purpose:
# Convert raw learning engagement data into a clean
# student-subject interaction dataset for recommendation.
#
# Output:
# data/processed/interactions.csv
#
# Output columns:
# student_id
# subject
# interaction_score
#
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    BASE_DIR / "data" / "aiml_content_engagement.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "interactions.csv"


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

REQUIRED_COLUMNS = [
    "student_id",
    "subject",
    "time_spent_mins",
    "completed",
    "replayed",
    "liked",
    "notes_taken",
    "difficulty_rating",
    "engagement_score",
]


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_data():
    print("\nLoading engagement dataset...")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Loaded {len(df)} raw engagement records.")

    return df


# ------------------------------------------------------------
# Validate data
# ------------------------------------------------------------

def validate_data(df):
    print("\nValidating dataset...")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["student_id"].isnull().any():
        raise ValueError("student_id contains missing values.")

    if df["subject"].isnull().any():
        raise ValueError("subject contains missing values.")

    print("Validation successful.")


# ------------------------------------------------------------
# Clean data
# ------------------------------------------------------------

def clean_data(df):
    print("\nCleaning dataset...")

    df = df.copy()

    # Remove duplicate records.
    df = df.drop_duplicates()

    # Remove rows without a student or subject.
    df = df.dropna(
        subset=[
            "student_id",
            "subject",
        ]
    )

    # Convert numeric columns safely.
    numeric_columns = [
        "time_spent_mins",
        "completed",
        "replayed",
        "liked",
        "notes_taken",
        "difficulty_rating",
        "engagement_score",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows where numeric data could not be converted.
    df = df.dropna(subset=numeric_columns)

    # Keep values within sensible ranges.
    df["engagement_score"] = df[
        "engagement_score"
    ].clip(0, 1)

    df["completed"] = df[
        "completed"
    ].clip(0, 1)

    df["replayed"] = df[
        "replayed"
    ].clip(0, 1)

    df["liked"] = df[
        "liked"
    ].clip(0, 1)

    df["notes_taken"] = df[
        "notes_taken"
    ].clip(0, 1)

    df["difficulty_rating"] = df[
        "difficulty_rating"
    ].clip(1, 5)

    df["time_spent_mins"] = df[
        "time_spent_mins"
    ].clip(lower=0)

    print(f"Records after cleaning: {len(df)}")

    return df


# ------------------------------------------------------------
# Create normalized features
# ------------------------------------------------------------

def create_interaction_score(df):
    print("\nCreating interaction features...")

    df = df.copy()

    scaler = MinMaxScaler()

    # Normalize time spent because its scale is different
    # from the binary engagement features.
    df["time_spent_normalized"] = scaler.fit_transform(
        df[["time_spent_mins"]]
    ).ravel()

    # Normalize difficulty so it can contribute modestly
    # without dominating the interaction score.
    df["difficulty_normalized"] = (
        (df["difficulty_rating"] - 1) / 4
    )

    # --------------------------------------------------------
    # Interaction score
    # --------------------------------------------------------
    #
    # Engagement score is the strongest signal.
    # Completion, likes, replay and notes indicate active
    # learning behaviour.
    # Time spent provides an additional engagement signal.
    #
    # Difficulty is intentionally given a small weight.
    #
    # The final score is bounded approximately between 0 and 1.
    # --------------------------------------------------------

    df["interaction_score"] = (
        0.40 * df["engagement_score"]
        + 0.20 * df["completed"]
        + 0.10 * df["liked"]
        + 0.10 * df["replayed"]
        + 0.05 * df["notes_taken"]
        + 0.10 * df["time_spent_normalized"]
        + 0.05 * df["difficulty_normalized"]
    )

    df["interaction_score"] = (
        df["interaction_score"]
        .clip(0, 1)
        .round(4)
    )

    return df


# ------------------------------------------------------------
# Aggregate student-subject interactions
# ------------------------------------------------------------

def aggregate_interactions(df):
    print("\nAggregating student-subject interactions...")

    interactions = (
        df.groupby(
            [
                "student_id",
                "subject",
            ],
            as_index=False
        )["interaction_score"]
        .mean()
    )

    interactions = interactions.sort_values(
        [
            "student_id",
            "interaction_score",
        ],
        ascending=[
            True,
            False,
        ]
    )

    print(
        f"Unique student-subject interactions: "
        f"{len(interactions)}"
    )

    return interactions


# ------------------------------------------------------------
# Save processed data
# ------------------------------------------------------------

def save_data(interactions):
    print("\nSaving processed interaction data...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    interactions.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Saved to: {OUTPUT_PATH}")


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("EDUTRACK LMS - DATA PIPELINE")
    print("=" * 70)

    df = load_data()

    validate_data(df)

    df = clean_data(df)

    df = create_interaction_score(df)

    interactions = aggregate_interactions(df)

    save_data(interactions)

    print("\n" + "-" * 70)
    print("PIPELINE SUMMARY")
    print("-" * 70)

    print(f"Raw records: {len(pd.read_csv(RAW_DATA_PATH))}")
    print(f"Processed interactions: {len(interactions)}")
    print(
        f"Students: "
        f"{interactions['student_id'].nunique()}"
    )
    print(
        f"Subjects: "
        f"{interactions['subject'].nunique()}"
    )

    print("\nFirst 10 processed interactions:")
    print(
        interactions
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("DATA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()