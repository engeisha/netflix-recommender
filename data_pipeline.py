import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def _parse_duration(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()
    if not text:
        return pd.NA

    if "season" in text:
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
        return 1

    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return pd.NA


def _target_encode(series):
    if series.empty:
        return pd.Series(dtype="float64")

    value_counts = series.value_counts(dropna=False)
    mapping = {value: count / len(series) for value, count in value_counts.items()}
    return series.map(mapping).fillna(0.0)


def prepare_data(csv_path="netflix_titles.csv", test_size=0.2, random_state=42):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    features = ["release_year", "rating", "duration", "listed_in", "description"]
    target = "type"

    if not all(col in df.columns for col in features + [target]):
        raise ValueError("CSV file must contain the required columns")

    df = df[[*features, target]].copy()
    df = df.rename(columns={"type": "target_type"})

    for col in ["release_year", "rating", "duration", "listed_in", "description"]:
        df[col] = df[col].astype("object")

    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["rating"] = df["rating"].fillna("Unknown")
    df["duration"] = df["duration"].fillna("0")
    df["listed_in"] = df["listed_in"].fillna("Unknown")
    df["description"] = df["description"].fillna("")

    df["duration_minutes_or_seasons"] = df["duration"].apply(_parse_duration)
    df["duration_minutes_or_seasons"] = pd.to_numeric(
        df["duration_minutes_or_seasons"], errors="coerce"
    )

    df["target_type"] = df["target_type"].fillna("Unknown")
    df["target"] = (df["target_type"] == "Movie").astype(int)

    text_feature = df["description"].fillna("")
    tfidf = TfidfVectorizer(max_features=50, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(text_feature)
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
        index=df.index,
    )

    encoded_rating = pd.get_dummies(df["rating"], prefix="rating")
    encoded_category = pd.get_dummies(df["listed_in"], prefix="listed_in")
    encoded_year = pd.DataFrame({"release_year": df["release_year"].fillna(df["release_year"].median())})
    encoded_duration = pd.DataFrame({"duration": df["duration_minutes_or_seasons"].fillna(0)})

    target_encoded_rating = _target_encode(df["rating"])
    target_encoded_rating = pd.DataFrame({"target_encoded_rating": target_encoded_rating})
    target_encoded_listed_in = _target_encode(df["listed_in"])
    target_encoded_listed_in = pd.DataFrame({"target_encoded_listed_in": target_encoded_listed_in})

    features_df = pd.concat(
        [
            encoded_year,
            encoded_duration,
            encoded_rating,
            encoded_category,
            target_encoded_rating,
            target_encoded_listed_in,
            tfidf_df,
        ],
        axis=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features_df,
        df["target"],
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test
