import pandas as pd


engagement = pd.read_csv("data/aiml_content_engagement.csv")


print("\n" + "=" * 70)
print("EDUTRACK LMS - ENGAGEMENT INTERACTION ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------
# 1. Number of subjects per student
# ---------------------------------------------------------

subjects_per_student = (
    engagement.groupby("student_id")["subject"]
    .nunique()
)


print("\nSUBJECTS PER ENGAGEMENT STUDENT")
print("-" * 70)

print(f"Students: {len(subjects_per_student)}")
print(f"Minimum subjects: {subjects_per_student.min()}")
print(f"Maximum subjects: {subjects_per_student.max()}")
print(f"Average subjects: {subjects_per_student.mean():.2f}")

print("\nDistribution:")
print(
    subjects_per_student
    .value_counts()
    .sort_index()
    .to_string()
)


# ---------------------------------------------------------
# 2. Number of interactions per student
# ---------------------------------------------------------

interactions_per_student = (
    engagement.groupby("student_id")
    .size()
)


print("\nINTERACTIONS PER STUDENT")
print("-" * 70)

print(f"Minimum interactions: {interactions_per_student.min()}")
print(f"Maximum interactions: {interactions_per_student.max()}")
print(f"Average interactions: {interactions_per_student.mean():.2f}")

print("\nDistribution:")
print(
    interactions_per_student
    .value_counts()
    .sort_index()
    .to_string()
)


# ---------------------------------------------------------
# 3. Students with multiple subjects
# ---------------------------------------------------------

multi_subject_students = subjects_per_student[
    subjects_per_student > 1
]


print("\nMULTI-SUBJECT STUDENTS")
print("-" * 70)

print(
    f"Students with more than one subject: "
    f"{len(multi_subject_students)}"
)

print(
    f"Percentage: "
    f"{len(multi_subject_students) / len(subjects_per_student) * 100:.2f}%"
)


# ---------------------------------------------------------
# 4. Student-subject interaction matrix
# ---------------------------------------------------------

matrix = engagement.pivot_table(
    index="student_id",
    columns="subject",
    values="engagement_score",
    aggfunc="mean"
)


print("\nENGAGEMENT MATRIX")
print("-" * 70)

print(f"Rows (students): {matrix.shape[0]}")
print(f"Columns (subjects): {matrix.shape[1]}")
print(
    f"Filled cells: "
    f"{matrix.notna().sum().sum()}"
)

print(
    f"Density: "
    f"{matrix.notna().sum().sum() / matrix.size:.4f}"
)


# ---------------------------------------------------------
# 5. Example multi-subject students
# ---------------------------------------------------------

print("\nEXAMPLE MULTI-SUBJECT STUDENTS")
print("-" * 70)

for student_id in multi_subject_students.head(10).index:
    rows = engagement[
        engagement["student_id"] == student_id
    ][
        [
            "subject",
            "content_type",
            "engagement_score",
            "completed",
            "difficulty_rating"
        ]
    ]

    print(f"\nStudent: {student_id}")
    print(rows.to_string(index=False))


print("\n" + "=" * 70)
print("ENGAGEMENT ANALYSIS COMPLETE")
print("=" * 70)