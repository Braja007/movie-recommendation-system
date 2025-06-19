def generate():
    import pandas as pd
    import logging
    import joblib
    from preprocess import preprocess_text
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv("src/movies.csv")
    df['combined'] = df['overview']
    df['cleaned_text'] = df['combined'].apply(preprocess_text)

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

    cosine_sim = cosine_similarity(tfidf_matrix)

    joblib.dump(df, "src/df_cleaned.pkl")
    joblib.dump(cosine_sim, "src/cosine_sim.pkl")

    logging.info("✅ Preprocessing done and files saved.")

if __name__ == "__main__":
    generate()
