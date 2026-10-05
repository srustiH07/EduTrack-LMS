import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


def inspect_csv(file_name):
    file_path = DATA_DIR / file_name

    print("\n" + "=" * 70)
    print(f"FILE: {file_name}")
    print("=" * 70)

    df = pd.read_csv(file_path)

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nUnique values:")
    for column in df.columns:
        if df[column].nunique() <= 20:
            print(f"\n{column}:")
            print(df[column].value_counts(dropna=False).to_string())


def inspect_json(file_name):
    file_path = DATA_DIR / file_name

    print("\n" + "=" * 70)
    print(f"FILE: {file_name}")
    print("=" * 70)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"\nJSON type: {type(data).__name__}")

    if isinstance(data, list):
        print(f"Number of records: {len(data)}")

        if data:
            print("\nFirst record:")
            print(json.dumps(data[0], indent=2))

            if isinstance(data[0], dict):
                print("\nKeys:")
                for key in data[0].keys():
                    print(f"  - {key}")

    elif isinstance(data, dict):
        print("\nTop-level keys:")
        for key in data.keys():
            print(f"  - {key}")

        print("\nJSON structure:")
        print(json.dumps(data, indent=2)[:5000])


print("\nEDUTRACK LMS - DATASET INSPECTION")
print("=" * 70)

inspect_csv("aiml_student_performance.csv")
inspect_csv("aiml_content_engagement.csv")
inspect_json("aiml_edtech_tests.json")

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETE")
print("=" * 70)