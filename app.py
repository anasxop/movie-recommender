import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Load movie data
df = pd.read_csv("ml-latest-small/movies.csv")
df['genres'] = df['genres'].str.replace('|', ' ', regex=False)

# Prepare TF-IDF matrix on genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Lowercase titles for case-insensitive matching
df['title_lower'] = df['title'].str.lower()
indices = pd.Series(df.index, index=df['title_lower'])

def get_recommendations(title):
    title = title.lower()

    # 1. Partial matching: check if user input is contained in any movie title
    matches = df[df['title_lower'].str.contains(title)]
    
    if not matches.empty:
        idx = matches.index[0]
    else:
        # 2. Fuzzy matching fallback
        close_matches = get_close_matches(title, df['title_lower'], n=1, cutoff=0.5)
        if not close_matches:
            return ["Movie not found."]
        best_match = close_matches[0]
        idx = indices[best_match]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # top 5 similar
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

st.title("🎬 Movie Recommendation System by Anas, Sakshi and Kaif")

movie_name = st.text_input("Enter a movie title:")
if movie_name:
    st.write("Recommendations:")
    for rec in get_recommendations(movie_name):
        st.write(f"👉 {rec}")
