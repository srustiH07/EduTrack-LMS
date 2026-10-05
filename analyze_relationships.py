import pandas as pd


performance = pd.read_csv("data/aiml_student_performance.csv")
engagement = pd.read_csv("data/aiml_content_engagement.csv")


print("\n" + "=" * 70)
print("EDUTRACK LMS - DATA RELATIONSHIP ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------
# 1. Student coverage
# ---------------------------------------------------------

performance_students = set(performance["student_id"])
engagement_students = set(engagement["student_id"])

print("\nSTUDENT COVERAGE")
print("-" * 70)

print(f"Performance students: {len(performance_students)}")
print(f"Engagement students: {len(engagement_students)}")
print(
    f"Students appearing in both datasets: "
    f"{len(performance_students & engagement_students)}"
)


# ---------------------------------------------------------
# 2. Performance records per student
# ---------------------------------------------------------

records_per_student = performance.groupby("student_id").size()

print("\nPERFORMANCE RECORDS PER STUDENT")
print("-" * 70)

print(f"Minimum subjects per student: {records_per_student.min()}")
print(f"Maximum subjects per student: {records_per_student.max()}")
print(f"Average subjects per student: {records_per_student.mean():.2f}")

print("\nDistribution:")
print(records_per_student.value_counts().sort_index().to_string())


# ---------------------------------------------------------
# 3. Student-subject combinations
# ---------------------------------------------------------

student_subject_pairs = performance[
    ["student_id", "subject"]
].drop_duplicates()

print("\nSTUDENT-SUBJECT INTERACTIONS")
print("-" * 70)

print(f"Unique student-subject pairs: {len(student_subject_pairs)}")
print(f"Possible combinations: {1000 * performance['subject'].nunique()}")
print(
    f"Matrix density: "
    f"{len(student_subject_pairs) / (1000 * performance['subject'].nunique()):.4f}"
)


# ---------------------------------------------------------
# 4. Subject statistics
# ---------------------------------------------------------

subject_stats = performance.groupby("subject").agg(
    students=("student_id", "nunique"),
    avg_post_score=("post_test_score", "mean"),
    avg_improvement=("score_improvement", "mean"),
    avg_completion=("completion_pct", "mean"),
)

print("\nSUBJECT STATISTICS")
print("-" * 70)

print(subject_stats.round(2).sort_values(
    "students",
    ascending=False
).to_string())


# ---------------------------------------------------------
# 5. Engagement subject coverage
# ---------------------------------------------------------

engagement_subject_stats = engagement.groupby("subject").agg(
    interactions=("engagement_id", "count"),
    students=("student_id", "nunique"),
    avg_engagement=("engagement_score", "mean"),
    avg_time=("time_spent_mins", "mean"),
    completion_rate=("completed", "mean"),
)

print("\nENGAGEMENT SUBJECT STATISTICS")
print("-" * 70)

print(
    engagement_subject_stats.round(2).sort_values(
        "interactions",
        ascending=False
    ).to_string()
)


# ---------------------------------------------------------
# 6. Performance + engagement overlap
# ---------------------------------------------------------

performance_pairs = set(
    zip(
        performance["student_id"],
        performance["subject"]
    )
)

engagement_pairs = set(
    zip(
        engagement["student_id"],
        engagement["subject"]
    )
)

overlap_pairs = performance_pairs & engagement_pairs

print("\nSTUDENT-SUBJECT OVERLAP")
print("-" * 70)

print(f"Performance pairs: {len(performance_pairs)}")
print(f"Engagement pairs: {len(engagement_pairs)}")
print(f"Pairs appearing in both: {len(overlap_pairs)}")


# ---------------------------------------------------------
# 7. Example students
# ---------------------------------------------------------

print("\nEXAMPLE STUDENT RECORDS")
print("-" * 70)

example_students = performance["student_id"].head(5).tolist()

for student_id in example_students:
    print(f"\nStudent: {student_id}")

    student_rows = performance[
        performance["student_id"] == student_id
    ][
        [
            "subject",
            "post_test_score",
            "score_improvement",
            "completion_pct",
            "best_quiz_score",
        ]
    ]

    print(student_rows.to_string(index=False))


print("\n" + "=" * 70)
print("RELATIONSHIP ANALYSIS COMPLETE")
print("=" * 70)