
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 CineMate Lite")
st.write("Conversational Movie Recommendation System")

movies = [
    ("Interstellar", "sci-fi drama space emotional science"),
    ("The Martian", "sci-fi adventure science survival comedy"),
    ("Arrival", "sci-fi drama aliens language mystery"),
    ("Toy Story", "animation comedy friendship family"),
    ("The Hangover", "comedy friends funny party"),
    ("Knives Out", "mystery crime comedy detective"),
    ("Inception", "sci-fi action thriller dreams"),
    ("The Conjuring", "horror thriller ghosts"),
    ("The Dark Knight", "action crime drama hero"),
    ("Finding Nemo", "animation adventure family friendship"),
]

titles = [m[0] for m in movies]
descriptions = [m[1] for m in movies]

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(descriptions)

query = st.chat_input("What kind of movie do you want?")

if query:
    st.write("**You:**", query)

    scores = cosine_similarity(
        vectorizer.transform([query]), matrix
    ).flatten()

    excluded = []
    text = query.lower()

    if "no horror" in text or "without horror" in text:
        excluded.append("horror")

    ranked = sorted(
        range(len(movies)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = [
        i for i in ranked
        if not any(g in descriptions[i] for g in excluded)
    ][:3]

    st.subheader("Top recommendations")

    for i in results:
        st.markdown(f"### {titles[i]}")
        st.write("Why it matches:", descriptions[i])
        st.write(f"Similarity score: {scores[i]:.2f}")
