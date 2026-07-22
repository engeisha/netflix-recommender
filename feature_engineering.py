from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DATASET_PATH = Path(__file__).resolve().parent / "cleaned_netflix_titles.csv"


def engineer_features(dataset_path: Path = DATASET_PATH):
    df = pd.read_csv(dataset_path)

    df["combined_features"] = (
        df["title"].fillna("")
        + " "
        + df["director"].fillna("")
        + " "
        + df["cast"].fillna("")
        + " "
        + df["listed_in"].fillna("")
        + " "
        + df["description"].fillna("")
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

    print(f"Loaded dataset from: {dataset_path}")
    print(f"Combined features shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    return df, tfidf_matrix


if __name__ == "__main__":
    engineer_features()
