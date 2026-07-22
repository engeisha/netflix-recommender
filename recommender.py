import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from feature_engineering import engineer_features


DATASET_PATH = Path(__file__).resolve().parent / "cleaned_netflix_titles.csv"


def get_recommendations(title: str, top_n: int = 5) -> pd.DataFrame:
    df, tfidf_matrix = engineer_features(DATASET_PATH)

    if "title" not in df.columns:
        raise KeyError("The dataset must contain a 'title' column")

    title_matches = df["title"].astype(str).str.lower() == title.lower()
    if not title_matches.any():
        raise ValueError(f"Title '{title}' was not found in the dataset")

    idx = df.index[title_matches][0]
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    similar_indices = cosine_similarities.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n]

    recommendations = []
    for i in similar_indices:
        score = cosine_similarities[i]
        recommendations.append(
            {
                "title": df.loc[i, "title"],
                "genres": df.loc[i, "listed_in"],
                "rating": df.loc[i, "rating"],
                "similarity_score_percent": round(score * 100, 2),
            }
        )

    return pd.DataFrame(recommendations)


if __name__ == "__main__":
    query_title = sys.argv[1] if len(sys.argv) > 1 else "Blood & Water"
    recommendations = get_recommendations(query_title)
    print(f"Top recommendations for '{query_title}':")
    print(recommendations.to_string(index=False))
