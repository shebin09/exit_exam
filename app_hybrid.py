import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Movie Recommenders: Content-Based + CF + Hybrid", layout="wide")

DATA_DIR = Path("ml-latest-small")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

@st.cache_data(show_spinner=False)
def load_movielens():
    if not DATA_DIR.exists():
        r = requests.get(MOVIELENS_URL, timeout=60)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(".")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies  = pd.read_csv(DATA_DIR / "movies.csv")
    return ratings, movies

@st.cache_data(show_spinner=False)
def prepare_content(movies_df: pd.DataFrame):
    m = movies_df.copy()
    m["genres"] = m["genres"].fillna("").replace("(no genres listed)", "", regex=False)
    m["content_soup"] = (
        m["genres"].str.replace("|", " ", regex=False)
                    .str.replace("-", "", regex=False)
                    .str.lower().str.strip()
    )
    m["title_lower"] = m["title"].str.lower().str.strip()
    title_to_index = pd.Series(m.index, index=m["title_lower"])
    return m, title_to_index

@st.cache_resource(show_spinner=False)
def build_tfidf_and_sim(movies_with_soup: pd.DataFrame):
    vec = TfidfVectorizer(token_pattern=r"[^ ]+")
    X = vec.fit_transform(movies_with_soup["content_soup"])
    sim = linear_kernel(X, X).astype(np.float32)
    return vec, sim

def recommend_similar_by_genre(title: str, movies: pd.DataFrame, title_to_index: pd.Series, sim: np.ndarray, top_k: int = 5):
    q = title.lower().strip()
    if q not in title_to_index:
        return pd.DataFrame(columns=["MovieId", "Title", "Similarity"])
    idx = int(title_to_index[q])
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [(i, s) for i, s in scores if i != idx][:top_k]
    rec_idx = [i for i, _ in scores]
    rec_scores = [float(s) for _, s in scores]
    return pd.DataFrame({
        "MovieId": movies.loc[rec_idx, "movieId"].values.astype(int),
        "Title": movies.loc[rec_idx, "title"].values,
        "Similarity": rec_scores
    })

# -------- CF without Surprise: load exported factors and predict with NumPy
@st.cache_resource(show_spinner=False)
def load_factors():
    uf = pd.read_csv(ART_DIR / "svd_user_factors.csv")
    it = pd.read_csv(ART_DIR / "svd_item_factors.csv")
    with open(ART_DIR / "svd_meta.json") as f:
        meta = json.load(f)
    k = int(meta["n_factors"])
    user_vecs = {}
    user_bu = {}
    for row in uf.itertuples(index=False):
        user_vecs[int(row.userId)] = np.array([getattr(row, f"f{i}") for i in range(k)], dtype=np.float64)
        user_bu[int(row.userId)] = float(row.bu)
    item_vecs = {}
    item_bi = {}
    for row in it.itertuples(index=False):
        item_vecs[int(row.movieId)] = np.array([getattr(row, f"f{i}") for i in range(k)], dtype=np.float64)
        item_bi[int(row.movieId)] = float(row.bi)
    mu = float(meta["global_mean"])
    return {"mu": mu, "k": k, "user_vecs": user_vecs, "user_bu": user_bu, "item_vecs": item_vecs, "item_bi": item_bi}

def top_n_for_user_numpy(user_id: int, factors: dict, ratings: pd.DataFrame, movies: pd.DataFrame, n: int = 5):
    mu = factors["mu"]
    user_vecs = factors["user_vecs"]
    user_bu = factors["user_bu"]
    item_vecs = factors["item_vecs"]
    item_bi = factors["item_bi"]

    if user_id not in user_vecs:
        return pd.DataFrame(columns=["MovieId", "Title", "PredictedRating"])

    seen = set(ratings.loc[ratings["userId"] == user_id, "movieId"].tolist())
    titles = movies.set_index("movieId")["title"]

    candidates = [mid for mid in titles.index if mid in item_vecs and mid not in seen]
    if not candidates:
        return pd.DataFrame(columns=["MovieId", "Title", "PredictedRating"])

    Qi = np.vstack([item_vecs[mid] for mid in candidates])     # (M, k)
    bi = np.array([item_bi[mid] for mid in candidates])        # (M,)
    pu = user_vecs[user_id]                                    # (k,)
    bu = user_bu[user_id]                                      # scalar

    preds = mu + bu + bi + Qi @ pu
    order = np.argsort(-preds)[:n]
    sel_ids = [int(candidates[i]) for i in order]
    sel_scores = [float(preds[i]) for i in order]
    sel_titles = [titles.get(mid, f"movieId={mid}") for mid in sel_ids]

    return pd.DataFrame({"MovieId": sel_ids, "Title": sel_titles, "PredictedRating": sel_scores})

# -------- Hybrid (RRF blend, CF-preferred)
def hybrid_recommendations_numpy(
    user_id: int,
    favorite_title: str,
    movies: pd.DataFrame,
    title_to_index: pd.Series,
    sim: np.ndarray,
    ratings: pd.DataFrame,
    factors: dict,
    top_each: int = 10,
    final_k: int = 10,
    w_cf: float = 0.6,
    w_cb: float = 0.4,
    rrf_k: int = 60
):
    cf_df = top_n_for_user_numpy(user_id, factors, ratings, movies, n=top_each)
    cb_df = recommend_similar_by_genre(favorite_title, movies, title_to_index, sim, top_k=top_each)

    rank_cf = {int(mid): rank + 1 for rank, mid in enumerate(cf_df["MovieId"].tolist())}
    rank_cb = {int(mid): rank + 1 for rank, mid in enumerate(cb_df["MovieId"].tolist())}

    all_ids = list(dict.fromkeys(list(rank_cf.keys()) + list(rank_cb.keys())))
    titles = movies.set_index("movieId")["title"]
    cf_pred = {int(r.MovieId): float(r.PredictedRating) for r in cf_df.itertuples(index=False)} if len(cf_df) else {}
    cb_sim  = {int(r.MovieId): float(r.Similarity)      for r in cb_df.itertuples(index=False)} if len(cb_df) else {}

    rows = []
    for mid in all_ids:
        s_cf = w_cf / (rrf_k + rank_cf[mid]) if mid in rank_cf else 0.0
        s_cb = w_cb / (rrf_k + rank_cb[mid]) if mid in rank_cb else 0.0
        score = s_cf + s_cb
        rows.append({
            "MovieId": int(mid),
            "Title": titles.get(mid, f"movieId={mid}"),
            "HybridScore": float(score),
            "Rank_CF": rank_cf.get(mid, np.nan),
            "Pred_CF": cf_pred.get(mid, np.nan),
            "Rank_CB": rank_cb.get(mid, np.nan),
            "Sim_CB": cb_sim.get(mid, np.nan),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(by=["HybridScore", "Pred_CF", "Sim_CB"], ascending=False, kind="mergesort")
    return out.head(final_k).reset_index(drop=True)

# -------- UI
st.title("Movie Recommenders: Content-Based + CF + Hybrid")
with st.spinner("Loading data and artifacts..."):
    ratings_df, movies_df = load_movielens()
    movies_prepped, title_to_index = prepare_content(movies_df)
    _, sim_mat = build_tfidf_and_sim(movies_prepped)
    factors = load_factors()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Section 1 — Content-Based (Genres)")
    selected_title = st.selectbox("Choose a movie", options=list(movies_prepped["title"].sort_values()), index=0)
    if selected_title:
        recs = recommend_similar_by_genre(selected_title, movies_prepped, title_to_index, sim_mat, top_k=5)
        st.dataframe(recs[["Title", "Similarity"]], use_container_width=True)

with col2:
    st.subheader("Section 2 — Collaborative Filtering (Exported SVD Factors)")
    min_uid = int(ratings_df["userId"].min())
    max_uid = int(ratings_df["userId"].max())
    uid = st.number_input("User ID", min_value=min_uid, max_value=max_uid, value=min_uid, step=1)
    if uid is not None:
        uid = int(uid)
        if uid not in factors["user_vecs"]:
            st.warning(f"User ID {uid} not found in exported factors. Try another between {min_uid} and {max_uid}.")
        else:
            top_recs = top_n_for_user_numpy(uid, factors, ratings_df, movies_df, n=5)
            st.dataframe(top_recs, use_container_width=True)

st.markdown("---")
st.subheader("Section 3 — Hybrid (RRF blend, CF-preferred)")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    fav_title = st.selectbox("Favorite movie", options=list(movies_prepped["title"].sort_values()), index=0)
with c2:
    min_uid = int(ratings_df["userId"].min())
    max_uid = int(ratings_df["userId"].max())
    hybrid_uid = st.number_input("User ID (for CF)", min_value=min_uid, max_value=max_uid, value=min_uid, step=1)
with c3:
    run_hybrid = st.button("Get Hybrid Recommendations", use_container_width=True)

if run_hybrid:
    uid = int(hybrid_uid)
    if uid not in factors["user_vecs"]:
        st.warning(f"User ID {uid} not present in exported user factors; hybrid will rely on content-based only.")
    hybrid = hybrid_recommendations_numpy(
        user_id=uid,
        favorite_title=fav_title,
        movies=movies_prepped,
        title_to_index=title_to_index,
        sim=sim_mat,
        ratings=ratings_df,
        factors=factors,
        top_each=10,
        final_k=10,
        w_cf=0.6,
        w_cb=0.4,
        rrf_k=60
    )
    if hybrid.empty:
        st.info("No hybrid recommendations could be generated. Try another user or movie.")
    else:
        st.dataframe(hybrid[["Title", "HybridScore", "Pred_CF", "Sim_CB"]], use_container_width=True)
        