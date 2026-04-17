from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("PRODUCTS_CSV_PATH", BASE_DIR / "data" / "products.csv"))
TOP_K_DEFAULT = 12


SYNONYM_GROUPS: dict[str, list[str]] = {
    "ufficio": ["ufficio", "business", "business casual", "business elegante", "smart casual", "office", "workwear"],
    "elegante": ["elegante", "formale", "sartoriale", "raffinato", "business", "cerimonia", "classico"],
    "outfit": ["outfit", "look", "abbinamento", "completo", "coordinato"],
    "casual": ["casual", "tempo libero", "weekend", "smart casual", "relaxed"],
    "estate": ["estate", "estivo", "lino", "cotone", "fresco", "leggero", "seersucker"],
    "inverno": ["inverno", "invernale", "lana", "cashmere", "caldo", "autunno inverno"],
    "viaggio": ["viaggio", "travel", "performance", "easy care", "packable", "stretch"],
    "cerimonia": ["cerimonia", "evento", "evento formale", "abito elegante", "sartoriale", "formale"],
    "smart": ["smart casual", "business casual", "elegante", "versatile"],
    "ufficio elegante": ["ufficio", "business", "elegante", "smart casual", "formale", "raffinato"],
    "giacca": ["giacca", "blazer", "giacca classica", "giacca casual"],
    "camicia": ["camicia", "camicia elegante", "camicia business", "camicia classica"],
    "pantalone": ["pantalone", "pantaloni", "chino", "trousers"],
    "scarpe": ["scarpe", "mocassini", "sneakers", "derby"],
}

CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "jacket": {"giacca", "blazer", "giacche"},
    "shirt": {"camicia", "camicie"},
    "pants": {"pantalone", "pantaloni", "chino", "chinos"},
    "suit": {"abito", "abiti", "completo"},
    "knitwear": {"maglia", "maglione", "cardigan", "dolcevita", "girocollo", "polo in maglia"},
    "polo": {"polo"},
    "tshirt": {"t-shirt", "tshirt", "t shirt"},
    "shoes": {"scarpe", "mocassini", "sneakers", "stringate", "derby", "loafer"},
    "accessories": {"cravatta", "cintura", "pochette", "borsa", "zaino", "cappello", "sciarpa", "portafoglio", "calze"},
}

INTENT_BOOSTS: dict[str, dict[str, float]] = {
    "office": {"jacket": 2.0, "shirt": 2.0, "pants": 1.7, "suit": 2.4, "shoes": 1.2, "knitwear": 0.9, "polo": 0.6},
    "elegant": {"suit": 2.5, "jacket": 2.0, "shirt": 1.8, "pants": 1.5, "shoes": 1.2, "accessories": 1.0},
    "casual": {"polo": 1.7, "knitwear": 1.6, "tshirt": 1.5, "pants": 1.2, "shoes": 1.0, "jacket": 0.8},
    "summer": {"shirt": 1.5, "polo": 1.4, "pants": 1.2, "jacket": 0.7, "knitwear": 0.4},
    "winter": {"knitwear": 1.8, "jacket": 1.2, "suit": 0.9, "accessories": 0.8},
    "travel": {"jacket": 1.2, "pants": 1.4, "shirt": 1.0, "shoes": 0.8, "accessories": 0.7},
}


@dataclass
class SearchEngine:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: Any

    @classmethod
    def load(cls, csv_path: Path) -> "SearchEngine":
        df = pd.read_csv(csv_path).fillna("")
        df["search_text"] = df.apply(build_search_text, axis=1)
        df["normalized_tags"] = df["generated_tags"].map(normalize_text)
        df["normalized_title"] = df["title"].map(normalize_text)
        df["normalized_type"] = df["product_type"].map(normalize_text)
        df["category_family"] = df.apply(infer_category_family, axis=1)
        df["price_value"] = df["price"].map(parse_price_value)
        df["currency"] = df["price"].map(parse_currency)

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(df["search_text"])
        return cls(df=df, vectorizer=vectorizer, matrix=matrix)

    def search(self, query: str, limit: int = TOP_K_DEFAULT) -> dict[str, Any]:
        limit = max(1, min(limit, 48))
        normalized_query = normalize_text(query)
        expanded_query = expand_query(normalized_query)
        q_vec = self.vectorizer.transform([expanded_query])
        cosine_scores = linear_kernel(q_vec, self.matrix).flatten()
        keyword_scores = self.df.apply(lambda row: keyword_overlap_score(row, normalized_query), axis=1).to_numpy(dtype=float)
        intent_scores = self.df["category_family"].map(lambda cat: intent_category_boost(cat, normalized_query)).to_numpy(dtype=float)

        total_scores = cosine_scores * 0.65 + keyword_scores * 0.20 + intent_scores * 0.15

        tmp = self.df.copy()
        tmp["score"] = total_scores
        tmp = tmp[tmp["score"] > 0.01].sort_values("score", ascending=False)
        if tmp.empty:
            return {
                "query": query,
                "normalized_query": normalized_query,
                "count": 0,
                "products": [],
            }

        tmp = dedupe_variants(tmp)

        if wants_outfit(normalized_query):
            tmp = diversify_for_outfit(tmp)

        tmp = tmp.head(limit)
        products = [serialize_product(row) for _, row in tmp.iterrows()]

        return {
            "query": query,
            "normalized_query": normalized_query,
            "count": len(products),
            "products": products,
        }


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9àèéìòùç\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_price_value(price: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", str(price or ""))
    return float(match.group(1)) if match else None


def parse_currency(price: str) -> str | None:
    match = re.search(r"([A-Z]{3})", str(price or ""))
    return match.group(1) if match else None


def build_search_text(row: pd.Series) -> str:
    parts = [
        row.get("title", ""),
        row.get("product_type", ""),
        row.get("generated_description", ""),
        row.get("generated_tags", ""),
        row.get("generated_search_queries", ""),
        row.get("description", ""),
        row.get("color", ""),
    ]
    return normalize_text(" ".join(str(p) for p in parts if p))


def infer_category_family(row: pd.Series) -> str:
    haystack = normalize_text(" ".join([
        str(row.get("title", "")),
        str(row.get("product_type", "")),
        str(row.get("generated_tags", "")),
    ]))
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in haystack for kw in keywords):
            return category
    return "other"


def expand_query(query: str) -> str:
    parts = [query]
    for phrase, expansions in SYNONYM_GROUPS.items():
        if phrase in query:
            parts.extend(expansions)

    tokens = query.split()
    for token in tokens:
        if token in SYNONYM_GROUPS:
            parts.extend(SYNONYM_GROUPS[token])

    if "outfit" in query or "look" in query:
        parts.extend(["giacca", "camicia", "pantalone", "mocassini", "blazer", "abito"])
    if "ufficio" in query and "eleg" in query:
        parts.extend(["abito", "giacca", "camicia", "chino", "mocassini", "business", "raffinato"])
    return normalize_text(" ".join(parts))


def keyword_overlap_score(row: pd.Series, query: str) -> float:
    score = 0.0
    query_terms = set(query.split())
    tags = row.get("normalized_tags", "")
    title = row.get("normalized_title", "")
    product_type = row.get("normalized_type", "")
    search_text = row.get("search_text", "")

    for term in query_terms:
        if len(term) <= 2:
            continue
        if term in title:
            score += 1.2
        if term in tags:
            score += 1.0
        if term in product_type:
            score += 0.8
        if term in search_text:
            score += 0.2

    if "ufficio" in query and any(k in tags for k in ["business", "ufficio", "smart casual"]):
        score += 1.4
    if any(k in query for k in ["elegante", "eleganti", "formale"]) and any(k in tags for k in ["elegante", "formale", "sartoriale"]):
        score += 1.4
    if any(k in query for k in ["outfit", "look"]) and any(k in tags for k in ["versatile", "completo", "business", "smart casual"]):
        score += 0.8
    if "estate" in query and any(k in tags for k in ["lino", "fresco", "leggero"]):
        score += 1.0
    if "viaggio" in query and any(k in tags for k in ["travel", "performance", "stretch"]):
        score += 1.0

    return score


def intent_category_boost(category_family: str, query: str) -> float:
    score = 0.0
    if any(k in query for k in ["ufficio", "office", "business"]):
        score += INTENT_BOOSTS["office"].get(category_family, 0.0)
    if any(k in query for k in ["elegante", "eleganti", "formale", "cerimonia", "raffinato"]):
        score += INTENT_BOOSTS["elegant"].get(category_family, 0.0)
    if any(k in query for k in ["casual", "weekend", "informale"]):
        score += INTENT_BOOSTS["casual"].get(category_family, 0.0)
    if any(k in query for k in ["estate", "estivo", "lino", "fresco"]):
        score += INTENT_BOOSTS["summer"].get(category_family, 0.0)
    if any(k in query for k in ["inverno", "invernale", "lana", "cashmere"]):
        score += INTENT_BOOSTS["winter"].get(category_family, 0.0)
    if any(k in query for k in ["travel", "viaggio"]):
        score += INTENT_BOOSTS["travel"].get(category_family, 0.0)
    return score


def dedupe_variants(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values("score", ascending=False)
        .drop_duplicates(subset=["item_group_id"], keep="first")
        .reset_index(drop=True)
    )


def wants_outfit(query: str) -> bool:
    return any(k in query for k in ["outfit", "look", "completo", "abbinamento"])


def diversify_for_outfit(df: pd.DataFrame) -> pd.DataFrame:
    preferred_order = ["suit", "jacket", "shirt", "pants", "shoes", "accessories", "knitwear", "polo"]
    chosen: list[pd.Series] = []
    used_groups: set[str] = set()

    for category in preferred_order:
        subset = df[df["category_family"] == category].head(2)
        for _, row in subset.iterrows():
            group_id = str(row.get("item_group_id", ""))
            if group_id and group_id not in used_groups:
                chosen.append(row)
                used_groups.add(group_id)

    if len(chosen) < 12:
        for _, row in df.iterrows():
            group_id = str(row.get("item_group_id", ""))
            if group_id and group_id not in used_groups:
                chosen.append(row)
                used_groups.add(group_id)
            if len(chosen) >= 24:
                break

    if not chosen:
        return df
    return pd.DataFrame(chosen).sort_values("score", ascending=False).reset_index(drop=True)


def serialize_product(row: pd.Series) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "item_group_id": row.get("item_group_id"),
        "title": row.get("title"),
        "description": row.get("generated_description") or row.get("description"),
        "price": row.get("price_value"),
        "currency": row.get("currency"),
        "image": row.get("image_link"),
        "link": row.get("link"),
        "color": row.get("color"),
        "category": row.get("product_type"),
        "category_family": row.get("category_family"),
        "tags": [tag.strip() for tag in str(row.get("generated_tags", "")).split(",") if tag.strip()],
        "score": round(float(row.get("score", 0.0)), 4),
    }


engine = SearchEngine.load(DATA_PATH)

app = FastAPI(
    title="Boggi Product Search API",
    version="1.0.0",
    description="API di ricerca prodotti basata su CSV taggato.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/search?q=outfit eleganti da ufficio", "/health"],
        "products_loaded": int(len(engine.df)),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "products_loaded": int(len(engine.df))}


@app.get("/search")
def search(
    q: str = Query(..., min_length=2, description="Query utente, es. outfit eleganti da ufficio"),
    limit: int = Query(TOP_K_DEFAULT, ge=1, le=48),
) -> dict[str, Any]:
    return engine.search(q, limit=limit)
