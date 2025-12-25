# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="📚 Book Recommender", page_icon="📖", layout="wide")

# -------------------------------------------------
# PURPLE HAZE THEME (Improved with Dark Text)
# -------------------------------------------------
colors = {
    "bg": "linear-gradient(135deg, #f4efff 0%, #dcd6f7 100%)",
    "accent": "#6a11cb",
    "text": "#1a0033",
    "card_bg": "rgba(255, 255, 255, 0.97)"
}

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: {colors['bg']};
        color: {colors['text']};
    }}
    .title {{
        font-size: 52px;
        font-weight: 900;
        text-align: center;
        color: {colors['text']};
        text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
        margin-top: 10px;


        
    }}
    .subtitle {{
        font-size: 20px;
        text-align: center;
        color: #3a2c4e;
        margin-bottom: 25px;
    }}
    .book-card {{
        background: {colors['card_bg']};
        border-radius: 18px;
        padding: 12px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        transition: all 0.3s ease-in-out;
    }}
    .book-card:hover {{
        transform: scale(1.06);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }}
    .book-title {{
        font-weight: 700;
        color: {colors['text']};
        text-align: center;
        font-size: 15px;
        margin-top: 8px;
    }}
    .book-author {{
        text-align: center;
        color: #444;
        font-size: 13px;
    }}
    .section-box {{
        background: {colors['card_bg']};
        border-radius: 25px;
        padding: 50px 40px;
        margin: 40px auto;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        text-align: center;
        backdrop-filter: blur(12px);
        animation: fadeIn 0.8s ease-in-out;
    }}
    .section-header {{
        font-size: 34px;
        font-weight: 800;
        color: {colors['accent']};
        margin-bottom: 12px;
    }}
    .section-subtext {{
        font-size: 17px;
        color: #222;
        margin-bottom: 25px;
    }}
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: white !important;
        border-radius: 8px;
        border: 1px solid #b8a9d0;
        color: {colors['text']};
    }}
    label, .stSlider label, .stTextInput label {{
        color: {colors['text']} !important;
        font-weight: 600;
    }}
    .stSlider > div {{
        color: {colors['text']} !important;
    }}
    .stButton>button {{
        background: linear-gradient(90deg, {colors['accent']}, #9b59b6);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 8px 25px;
        border: none;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background: #2b1a40;
        transform: scale(1.05);
    }}
    @keyframes fadeIn {{
        from {{opacity: 0; transform: translateY(20px);}}
        to {{opacity: 1; transform: translateY(0);}}
    }}
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
books = pd.read_csv('Books.csv', low_memory=False)
users = pd.read_csv('Users.csv', low_memory=False)
ratings = pd.read_csv('Ratings.csv', low_memory=False)

ratings = ratings[ratings['Book-Rating'] > 0]
users = users[users['User-ID'].isin(ratings['User-ID'].unique())]
books = books[books['ISBN'].isin(ratings['ISBN'].unique())]

ratings_with_users = pd.merge(ratings, users, on='User-ID')
final_df = pd.merge(ratings_with_users, books, on='ISBN')

# Popularity-based top books
popularity_df = final_df.groupby('Book-Title').agg({'Book-Rating': ['count', 'mean']}).reset_index()
popularity_df.columns = ['Book-Title', 'Num-Ratings', 'Avg-Rating']
popular_books = popularity_df[popularity_df['Num-Ratings'] >= 200]
popular_books = popular_books.sort_values(by=['Avg-Rating', 'Num-Ratings'], ascending=False)
book_meta = final_df[['Book-Title', 'Book-Author', 'Image-URL-M']].drop_duplicates(subset=['Book-Title'])

# Collaborative filtering
df = final_df.copy()
MIN_RATINGS_PER_USER = 130
MIN_RATINGS_PER_BOOK = 130
user_counts = df['User-ID'].value_counts()
book_counts = df['Book-Title'].value_counts()
filtered_df = df[
    df['User-ID'].isin(user_counts[user_counts >= MIN_RATINGS_PER_USER].index) &
    df['Book-Title'].isin(book_counts[book_counts >= MIN_RATINGS_PER_BOOK].index)
]
user_item_matrix = filtered_df.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0)
book_similarity = cosine_similarity(user_item_matrix.T)
book_similarity_df = pd.DataFrame(book_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.title("📚 Navigation")
mode = st.sidebar.radio("Choose Mode:", ["📈 Top Books", "🔍 Personalized Recommendations"])

# -------------------------------------------------
# MAIN HEADER
# -------------------------------------------------
st.markdown(f'<div class="title">Book Recommendation System</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">💜 Powered by Cosine Similarity</div>', unsafe_allow_html=True)

# =================================================
# MODE 1: TOP BOOKS
# =================================================
if mode == "📈 Top Books":
    st.markdown(f"""
        <div class="section-box">
            <div class="section-header">📊 Top Rated Books</div>
            <div class="section-subtext">Select how many top books you’d like to explore!</div>
        </div>
    """, unsafe_allow_html=True)

    top_n = st.slider("📘 Number of Top Books to Display:", 10, 50, 20)
    top_books = popular_books.head(top_n).merge(book_meta, on='Book-Title', how='left')

    st.markdown(f"<h4 style='text-align:center;color:{colors['accent']};'>Top {top_n} Books:</h4>", unsafe_allow_html=True)

    cols = st.columns(5)
    for idx, row in enumerate(top_books.itertuples()):
        col = cols[idx % 5]
        with col:
            st.markdown(f"""
                <div class="book-card">
                    <img src="{row._5}" style="width:100%; border-radius:12px;">
                    <div class="book-title">{row._1}</div>
                    <div class="book-author">👤 {row._4}</div>
                    <div style="text-align:center; font-size:13px; color:#f39c12;">⭐ {row._3:.2f} | 💬 {int(row._2)} Ratings</div>
                </div>
            """, unsafe_allow_html=True)

# =================================================
# MODE 2: PERSONALIZED RECOMMENDATIONS
# =================================================
else:
    st.markdown(f"""
        <div class="section-box">
            <div class="section-header">🔍 Personalized Book Recommendations</div>
            <div class="section-subtext">Start typing a book name below to search quickly!</div>
        </div>
    """, unsafe_allow_html=True)

    # Animated search bar simulation (live typing filter)
    all_books = sorted(book_similarity_df.columns)
    search_query = st.text_input("🔎 Search Book Name:")

    if search_query:
        filtered_books = [b for b in all_books if search_query.lower() in b.lower()]
    else:
        filtered_books = all_books

    selected_book = st.selectbox("Select from Results:", filtered_books)
    num_recommend = st.slider("Number of Recommendations:", 1, 10, 5)

    if st.button("✨ Show Recommendations"):
        with st.spinner("🔄 Finding similar books..."):
            time.sleep(1.8)
            similar_books = (
                book_similarity_df[selected_book]
                .sort_values(ascending=False)
                .drop(selected_book, errors="ignore")
                .head(num_recommend)
            )
            rec_books = book_meta[book_meta['Book-Title'].isin(similar_books.index)]

        st.success(f"Top {num_recommend} Books similar to **{selected_book}**:")
        rec_cols = st.columns(5)
        for idx, row in enumerate(rec_books.itertuples()):
            col = rec_cols[idx % 5]
            with col:
                st.markdown(f"""
                    <div class="book-card">
                        <img src="{row._3}" style="width:100%; border-radius:12px;">
                        <div class="book-title">{row._1}</div>
                        <div class="book-author">👤 {row._2}</div>
                    </div>
                """, unsafe_allow_html=True)
