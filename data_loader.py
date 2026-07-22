from pathlib import Path

import pandas as pd


DATASET_PATH = Path(__file__).resolve().parent / "netflix_titles.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "cleaned_netflix_titles.csv"


def load_and_clean_data(dataset_path: Path = DATASET_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)

    columns_to_clean = ["director", "cast", "country", "rating"]
    for column in columns_to_clean:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown")
        else:
            raise KeyError(f"Column '{column}' not found in dataset")

    df.to_csv(output_path, index=False)

    print(f"Loaded dataset from: {dataset_path}")
    print(f"Cleaned data saved to: {output_path}")
    print(f"Shape after cleaning: {df.shape}")
    print("Missing values after cleaning:")
    print(df[columns_to_clean].isna().sum())
    return df


if __name__ == "__main__":
    load_and_clean_data()
