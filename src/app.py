import os
import json
import streamlit as st
from pathlib import Path

# Check and run preprocessing first
if not Path("src/df_cleaned.pkl").exists():
    import generate_pickle
    generate_pickle.generate()

# Now import only after .pkl files exist
from recommend import df, recommend_movies
from omdb_utils import get_movie_details

OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎮",
    layout="centered"
)

st.title(":clapper: Movie Recommender")

movie_list = sorted(df['title'].dropna().unique())
selected_movie = st.selectbox(":clapper: Select a movie:", movie_list)

if st.button(":rocket: Recommend Similar Movies"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend_movies(selected_movie)
        if recommendations is None or recommendations.empty:
            st.warning("Sorry, no recommendations found.")
        else:
            st.success("Top similar movies:")
            for _, row in recommendations.iterrows():
                movie_title = row['title']
                plot, poster = get_movie_details(movie_title, OMDB_API_KEY)
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if poster != "N/A":
                            st.image(poster, width=100)
                        else:
                            st.write(":x: No Poster Found")
                    with col2:
                        st.markdown(f"### {movie_title}")
                        st.markdown(f"*{plot}*" if plot != "N/A" else "_Plot not available_")
