import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# ENV
# =========================
load_dotenv()
OMDB_API_KEY = os.getenv("TMDB_API_KEY")  # Using TMDB_API_KEY env var for OMDB

OMDB_BASE = "http://www.omdbapi.com/"

if not OMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None

# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: str
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None

class TMDBMovieDetails(BaseModel):
    tmdb_id: str
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []

class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None

class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]

# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()

async def omdb_get(params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["apikey"] = OMDB_API_KEY
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(OMDB_BASE, params=q)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OMDB error {r.status_code}: {r.text}")
    data = r.json()
    if data.get("Response") == "False":
        raise HTTPException(status_code=404, detail=f"OMDB: {data.get('Error', 'Not found')}")
    return data

async def omdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await omdb_get({"s": query, "type": "movie", "page": page})

async def omdb_movie_details(imdb_id: str) -> TMDBMovieDetails:
    data = await omdb_get({"i": imdb_id, "plot": "full"})
    genre_list = [{"name": g.strip()} for g in data.get("Genre", "").split(",") if g.strip()]
    return TMDBMovieDetails(
        tmdb_id=data.get("imdbID", ""),
        title=data.get("Title", ""),
        overview=data.get("Plot"),
        release_date=data.get("Year", ""),
        poster_url=data.get("Poster") if data.get("Poster") != "N/A" else None,
        backdrop_url=None,
        genres=genre_list,
    )

async def omdb_cards_from_search(data: Dict[str, Any], limit: int = 20) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in data.get("Search", [])[:limit]:
        out.append(TMDBMovieCard(
            tmdb_id=m.get("imdbID", ""),
            title=m.get("Title", ""),
            poster_url=m.get("Poster") if m.get("Poster") != "N/A" else None,
            release_date=m.get("Year", ""),
        ))
    return out

async def omdb_search_first(query: str) -> Optional[dict]:
    data = await omdb_search_movies(query=query, page=1)
    results = data.get("Search", [])
    return results[0] if results else None

def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}
    for k, v in indices.items():
        title_to_idx[_norm_title(k)] = int(v)
    return title_to_idx

def get_local_idx_by_title(title: str) -> int:
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(status_code=404, detail=f"Title not found: '{title}'")

def tfidf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")
    idx = get_local_idx_by_title(query_title)
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)
    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        # FIXED: removed extra )
        out.append((title_i, float(scores[int(i)]))
        if len(out) >= top_n:
            break
    return out

async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    m = await omdb_search_first(title)
    if not m:
        return None
    return TMDBMovieCard(
        tmdb_id=m.get("imdbID", ""),
        title=m.get("Title", title),
        poster_url=m.get("Poster") if m.get("Poster") != "N/A" else None,
        release_date=m.get("Year", ""),
    )

# =========================
# STARTUP
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/home", response_model=List[TMDBMovieCard])
async def home(category: str = Query("popular"), limit: int = Query(24, ge=1, le=50)):
    # Hardcoded popular movies with REAL posters
    popular_movies = [
        {"imdbID": "tt0111161", "Title": "The Shawshank Redemption", "Poster": "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxNDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg", "Year": "1994"},
        {"imdbID": "tt0068646", "Title": "The Godfather", "Poster": "https://m.media-amazon.com/images/M/MV5BM2MyNjYxNmUtYTAwNi00MTYxLWJmNWYtYzZlOGI2ZjMyNDVmXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_.jpg", "Year": "1972"},
        {"imdbID": "tt0468569", "Title": "The Dark Knight", "Poster": "https://m.media-amazon.com/images/M/MV5BMTMxNTMwODM0NF5BMl5BanBnXkFtZTcwODAyMTk2Mw@@._V1_.jpg", "Year": "2008"},
        {"imdbID": "tt0109830", "Title": "Forrest Gump", "Poster": "https://m.media-amazon.com/images/M/MV5BNWIwODRlZTUtY2U3OS00NTU5LWI0ZDctOTBlNGQwYmFjNjM3L2ltZy8XIGlpbQAuZXhwbwoPTEU~._V1_.jpg", "Year": "1994"},
        {"imdbID": "tt0133093", "Title": "The Matrix", "Poster": "https://m.media-amazon.com/images/M/MV5BMjMxNzk1MTQyMl5BMl5BanBnXkFtZTgwMDM5ODI5NDE@._V1_.jpg", "Year": "1999"},
        {"imdbID": "tt1375666", "Title": "Inception", "Poster": "https://m.media-amazon.com/images/M/MV5BMjAxMzY3NjcxNF5BMl5BanBnXkFtZTcwNTI5OTM1Mw@@._V1_.jpg", "Year": "2010"},
        {"imdbID": "tt0816692", "Title": "Interstellar", "Poster": "https://m.media-amazon.com/images/M/MV5BZjg5Yjk1YjMtOGQ2ZC00NmVhLWIxMWYtZDg1Nzc4NTZkZGU1XkEyXkFqcGdeQXVyNTkzNDQ4NTc@._V1_.jpg", "Year": "2014"},
        {"imdbID": "tt4154796", "Title": "Avengers: Endgame", "Poster": "https://m.media-amazon.com/images/M/MV5BNjhmEMjgxZDctNDlhZi00NWRmLWFkOTUtZDQ2N2E3NmQ1NWU2XkEyXkFqcGdeQXVyNTkzNDQ4NTc@._V1_.jpg", "Year": "2019"},
    ]
    return [TMDBMovieCard(tmdb_id=m["imdbID"], title=m["Title"], poster_url=m["Poster"], release_date=m["Year"]) for m in popular_movies[:limit]]

@app.get("/tmdb/search")
async def tmdb_search(query: str = Query(..., min_length=1), page: int = Query(1, ge=1, le=10)):
    data = await omdb_search_movies(query=query, page=page)
    return {"results": data.get("Search", [])}

@app.get("/movie/id/{imdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(imdb_id: str):
    return await omdb_movie_details(imdb_id)

@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(tmdb_id: str = Query(...), limit: int = Query(18, ge=1, le=50)):
    details = await omdb_movie_details(tmdb_id)
    if not details.genres:
        return []
    genre_name = details.genres[0]["name"]
    search_query = f"{details.title} {genre_name}"
    data = await omdb_search_movies(search_query, page=1)
    cards = await omdb_cards_from_search(data, limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]

@app.get("/recommend/tfidf")
async def recommend_tfidf(title: str = Query(..., min_length=1), top_n: int = Query(10, ge=1, le=50)):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]

@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    best = await omdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404, detail=f"No movie found: {query}")
    
    imdb_id = best.get("imdbID", "")
    details = await omdb_movie_details(imdb_id)
    
    # TF-IDF recommendations
    tfidf_items: List[TFIDFRecItem] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        recs = []
    
    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))
    
    # Genre recommendations
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_name = details.genres[0]["name"]
        search_query = f"{details.title} {genre_name}"
        data = await omdb_search_movies(search_query, page=1)
        cards = await omdb_cards_from_search(data, limit=genre_limit)
        genre_recs = [c for c in cards if c.tmdb_id != imdb_id]
    
    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
