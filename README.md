# 🎬 Movie Recommendation System

A content-based movie recommender app built with **Streamlit**, powered by **TF-IDF vectorization** and **cosine similarity**. Fetches posters and plots using the **OMDb API**.

---

## 🚀 Features

- 🔍 Search for a movie title
- 🎞️ Get top 5 similar movies using NLP
- 🖼️ Movie poster and plot fetched using OMDb API
- ⚡ Fast runtime using preprocessed `.pkl` files
- 📦 Built with modular Python files and Streamlit

---

## 🧠 How It Works

1. Movie data (`movies.csv`) is preprocessed using `nltk` and `sklearn`
2. Text vectorized using `TF-IDF`
3. Cosine similarity is calculated between movie descriptions
4. Top similar movies are shown with poster & plot using OMDb API

---

## 🛠️ Project Structure

movie-recommendation-system/
│
├── .gitignore                  # Ignore __pycache__, .pkl, .ipynb_checkpoints, etc.
├── README.md                   # Project overview, setup instructions
├── requirements.txt            # All required packages for pip install
├── Movie_recommendation_system.ipynb  # Notebook for prototyping (optional)
│
└── src/                        # All source code and data files
    ├── app.py                 # 🎯 Main Streamlit app (entry point)
    ├── movies.csv             # 🎬 Raw movie dataset
    ├── config.json            # 🔑 Stores your OMDb API key (not for public push!)
    ├── preprocess.py          # 🧹 Text preprocessing (stopword removal, stemming)
    ├── recommend.py           # 🤖 Movie recommendation logic using cosine similarity
    ├── omdb_utils.py          # 🖼️ Fetches movie plot + poster from OMDb API
    ├── generate_pickle.py     # 💾 Preprocess + save df_cleaned.pkl & cosine_sim.pkl
    ├── df_cleaned.pkl         # ✅ Cleaned movie data (generated from .csv)
    └── cosine_sim.pkl         # ✅ Cosine similarity matrix (TF-IDF based)
