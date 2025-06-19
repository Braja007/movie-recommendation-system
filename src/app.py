import os
import subprocess
import streamlit as st
import json

# ✅ Generate .pkl files on first run
if not os.path.exists("src/df_cleaned.pkl"):
    subprocess.run(["python", "src/generate_pickle.py"], check=True)

# ✅ Now import logic that relies on .pkl files
from recommend import df, recommend_movies
from omdb_utils import get_movie_details

# ✅ Load OMDB API key (locally from config.json OR from st.secrets on deployment)
try:
    config = json.load(open("config.json"))
    OMDB_API_KEY = config.get("OMDB_API_KEY")
except FileNotFoundError:
    OMDB_API_KEY = st.secrets["OMDB"]["OMDB_API_KEY"]

# ✅ Streamlit app UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender System")

movie_list = sorted(df["title"].dropna().unique())
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("🚀 Recommend Similar Movies"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend_movies(selected_movie)

        if recommendations is None or recommendations.empty:
            st.warning("No recommendations found.")
        else:
            for _, row in recommendations.iterrows():
                movie_title = row["title"]
                plot, poster_url = get_movie_details(movie_title, OMDB_API_KEY)

                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if poster_url != "N/A":
                            st.image(poster_url, width=100)
                        else:
                            st.markdown("❌ No poster")
                    with col2:
                        st.subheader(movie_title)
                        st.markdown(f"_{plot}_" if plot != "N/A" else "_Plot not available_")
