import pandas as pd
import logging
from preprocess import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

logging.info("🚀 Starting preprocessing...")

df = pd.read_csv("movies.csv")

logging.info("✅ Dataset loaded successfully. Total rows: %d", len(df))

# 👇 ADD THIS LINE if you only have a single column like 'overview'
df['combined'] = df['overview']  # or whichever column contains your movie descriptions

logging.info("🧹 Cleaning text...")
df['cleaned_text'] = df['combined'].apply(preprocess_text)
logging.info("✅ Text cleaned.")

logging.info("🔠 Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

logging.info("📐 Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix)

joblib.dump(df, "df_cleaned.pkl")
joblib.dump(cosine_sim, "cosine_sim.pkl")

logging.info("💾 Data saved to disk.")
logging.info("✅ Preprocessing complete.")
