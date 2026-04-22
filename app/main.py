from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("PRODUCTS_CSV_PATH", BASE_DIR / "data" / "products.csv"))
TOP_K_DEFAULT = 12


SYNONYM_GROUPS: dict[str, list[str]] = {
    "ufficio": ["ufficio", "business", "business casual", "smart casual", "office", "workwear", "meeting"],
    "elegante": ["elegante", "formale", "sartoriale", "raffinato", "classico", "cerimonia"],
    "outfit": ["outfit", "look", "abbinamento", "completo", "coordinato"],
    "casual": ["casual", "tempo libero", "weekend", "relaxed", "smart casual"],
    "estate": ["estate", "estivo", "lino", "cotone", "fresco", "leggero"],
    "inverno": ["inverno", "invernale", "lana", "cashmere", "caldo"],
    "viaggio": ["viaggio", "travel", "performance", "stretch", "easy care", "packable"],
    "cerimonia": ["cerimonia", "evento", "evento formale", "abito elegante", "sartoriale", "formale"],
    "smart": ["smart casual", "business casual", "elegante", "versatile"],
    "ufficio elegante": ["ufficio", "business", "elegante", "smart casual", "formale", "raffinato"],
    "giacca": ["giacca", "blazer", "jacket"],
    "camicia": ["camicia", "shirt"],
    "pantalone": ["pantalone", "pantaloni", "trousers", "chino"],
    "scarpe": ["scarpe", "mocassini", "sneakers", "loafer", "derby", "ballerine"],
    "borsa": ["borsa", "bag", "shopper", "tote bag", "crossbody", "shoulder bag"],
}


CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "suit": {"abito", "abiti", "tailored", "tailoring", "completo", "suit"},
    "jacket": {"giacca", "blazer", "jacket"},
    "shirt": {"camicia", "camicie", "shirt"},
    "pants": {"pantalone", "pantaloni", "trousers", "chino", "chinos", "jeans"},
    "knitwear": {"maglia", "maglione", "cardigan", "knitwear", "dolcevita", "pullover"},
    "polo": {"polo"},
    "hoodie": {"felpa", "hoodie", "sweatshirt"},
    "shoes": {
        "scarpe",
        "shoe",
        "sneakers",
        "sneaker",
        "mocassini",
        "loafer",
        "derby",
        "ballerine",
        "stivaletti",
        "sandali",
        "boots",
        "boot",
    },
    "bags": {
        "borsa",
        "borsette",
        "shopper",
        "tote bag",
        "hobo bag",
        "crossbody",
        "shoulder bag",
        "zaino",
        "backpack",
        "bag",
        "pouch",
        "clutch",
    },
    "accessories": {
        "cintura",
        "belt",
        "portafoglio",
        "wallet",
        "sciarpa",
        "scarf",
        "cappello",
        "hat",
        "cravatta",
        "tie",
        "calze",
        "socks",
        "pelletteria",
        "piccola pelletteria",
        "gloves",
        "guanti",
    },
}


INTENT_BOOSTS: dict[str, dict[str, float]] = {
    "office": {
        "suit": 1.6,
        "jacket": 1.4,
        "shirt": 1.4,
        "pants": 1.2,
        "shoes": 1.2,
        "bags": 1.1,
        "accessories": 1.0,
    },
    "elegant": {
        "suit": 1.7,
        "jacket": 1.3,
        "shirt": 1.0,
        "shoes": 1.4,
        "bags": 1.0,
        "accessories": 1.3,
    },
    "casual": {
        "hoodie": 1.7,
        "polo": 1.3,
        "knitwear": 1.1,
        "pants": 1.1,
        "shoes": 1.3,
        "bags": 0.9,
    },
    "summer": {
        "shirt": 1.2,
        "pants": 1.0,
        "shoes": 0.9,
        "bags": 0.7,
        "accessories": 0.8,
    },
    "winter": {
        "knitwear": 1.4,
        "hoodie": 1.0,
        "pants": 1.0,
        "shoes": 1.0,
        "accessories": 1.2,
        "bags": 0.6,
    },
    "travel": {
        "bags": 1.6,
        "shoes": 1.0,
        "accessories": 0.9,
        "hoodie": 0.8,
        "pants": 0.8,
    },
}


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_price_value(price: str) -> float | None:
    raw = str(price or "").replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)", raw)
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
        row.get("gender", ""),
    ]
    return normalize_text(" ".join(str(p) for p in parts if p))


def infer_category_family(row: pd.Series) -> str:
    haystack = normalize_text(
        " ".join(
            [
                str(row.get("title", "")),
                str(row.get("product_type", "")),
                str(row.get("generated_tags", "")),
            ]
        )
    )

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
        parts.extend(
            [
                "giacca",
                "camicia",
                "pantalone",
                "scarpe",
                "borsa",
                "accessori",
                "blazer",
            ]
        )

    if "ufficio" in query and ("eleg" in query or "smart" in query or "business" in query):
        parts.extend(["business", "raffinato", "meeting", "classico"])

    return normalize_text(" ".join(parts))


def keyword_overlap_score(row: pd.Series, query: str) -> float:
    score = 0.0
    query_terms = set(query.split())

    tags = str(row.get("normalized_tags", ""))
    title = str(row.get("normalized_title", ""))
    product_type = str(row.get("normalized_type", ""))
    search_text = str(row.get("search_text", ""))

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

    if "ufficio" in query and any(k in tags for k in ["business", "ufficio", "smart casual", "office"]):
        score += 1.4

    if any(k in query for k in ["elegante", "eleganti", "formale", "raffinato"]) and any(
        k in tags for k in ["elegante", "formale", "sartoriale", "raffinato"]
    ):
        score += 1.4

    if any(k in query for k in ["outfit", "look"]) and any(
        k in tags for k in ["versatile", "completo", "business", "smart casual", "casual"]
    ):
        score += 0.8

    if any(k in query for k in ["estate", "estivo", "fresco"]) and any(
        k in tags for k in ["lino", "fresco", "leggero", "cotone"]
    ):
        score += 1.0

    if any(k in query for k in ["viaggio", "travel"]) and any(
        k in tags for k in ["travel", "performance", "stretch", "easy care"]
    ):
        score += 1.0

    return score


def intent_category_boost(category_family: str, query: str) -> float:
    score = 0.0

    if any(k in query for k in ["ufficio", "office", "business", "meeting"]):
        score += INTENT_BOOSTS["office"].get(category_family, 0.0)

    if any(k in query for k in ["elegante", "eleganti", "formale", "cerimonia", "raffinato"]):
        score += INTENT_BOOSTS["elegant"].get(category_family, 0.0)

    if any(k in query for k in ["casual", "weekend", "informale", "relaxed"]):
        score += INTENT_BOOSTS["casual"].get(category_family, 0.0)

    if any(k in query for k in ["estate", "estivo", "lino", "fresco", "leggero"]):
        score += INTENT_BOOSTS["summer"].get(category_family, 0.0)

    if any(k in query for k in ["inverno", "invernale", "lana", "cashmere", "caldo"]):
        score += INTENT_BOOSTS["winter"].get(category_family, 0.0)

    if any(k in query for k in ["travel", "viaggio"]):
        score += INTENT_BOOSTS["travel"].get(category_family, 0.0)

    return score


def filter_by_gender(df: pd.DataFrame, gender: str | None) -> pd.DataFrame:
    if not gender:
        return df

    g = normalize_text(gender)

    male_values = {"male", "uomo", "menswear", "man", "men"}
    female_values = {"female", "donna", "womenswear", "woman", "women"}

    if g in male_values:
        return df[df["gender_normalized"].isin(male_values | {"unisex"})]

    if g in female_values:
        return df[df["gender_normalized"].isin(female_values | {"unisex"})]

    if g == "unisex":
        return df[df["gender_normalized"] == "unisex"]

    return df


def dedupe_variants(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["dedupe_key"] = tmp["item_group_id"].astype(str).str.strip()
    tmp.loc[tmp["dedupe_key"] == "", "dedupe_key"] = tmp["id"].astype(str)

    return (
        tmp.sort_values("score", ascending=False)
        .drop_duplicates(subset=["dedupe_key"], keep="first")
        .drop(columns=["dedupe_key"])
        .reset_index(drop=True)
    )


def wants_outfit(query: str) -> bool:
    return any(k in query for k in ["outfit", "look", "completo", "abbinamento"])


def diversify_for_outfit(df: pd.DataFrame) -> pd.DataFrame:
    preferred_order = ["suit", "jacket", "shirt", "pants", "shoes", "bags", "accessories", "knitwear", "polo"]
    chosen: list[pd.Series] = []
    used_groups: set[str] = set()

    for category in preferred_order:
        subset = df[df["category_family"] == category].head(2)
        for _, row in subset.iterrows():
            group_id = str(row.get("item_group_id", "")).strip() or str(row.get("id", "")).strip()
            if group_id and group_id not in used_groups:
                chosen.append(row)
                used_groups.add(group_id)

    if len(chosen) < 12:
        for _, row in df.iterrows():
            group_id = str(row.get("item_group_id", "")).strip() or str(row.get("id", "")).strip()
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
        "gender": row.get("gender"),
        "category": row.get("product_type"),
        "category_family": row.get("category_family"),
        "tags": [tag.strip() for tag in str(row.get("generated_tags", "")).split(",") if tag.strip()],
        "score": round(float(row.get("score", 0.0)), 4),
    }


@dataclass
class SearchEngine:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: Any

    @classmethod
    def load(cls, csv_path: Path) -> "SearchEngine":
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig").fillna("")

        df = df.rename(
            columns={
                "ID": "id",
                "tags": "generated_tags",
            }
        )

        if "generated_tags" not in df.columns:
            df["generated_tags"] = ""

        if "generated_description" not in df.columns:
            df["generated_description"] = df.get("description", "")

        if "generated_search_queries" not in df.columns:
            df["generated_search_queries"] = ""

        if "gender" not in df.columns:
            df["gender"] = ""

        df["generated_tags"] = (
            df["generated_tags"]
            .astype(str)
            .str.replace("|", ",", regex=False)
            .str.replace(r"\s*,\s*", ", ", regex=True)
            .str.strip()
        )

        df["search_text"] = df.apply(build_search_text, axis=1)
        df["normalized_tags"] = df["generated_tags"].map(normalize_text)
        df["normalized_title"] = df["title"].map(normalize_text)
        df["normalized_type"] = df["product_type"].map(normalize_text)
        df["category_family"] = df.apply(infer_category_family, axis=1)
        df["price_value"] = df["price"].map(parse_price_value)
        df["currency"] = df["price"].map(parse_currency)
        df["gender_normalized"] = df["gender"].map(normalize_text)

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(df["search_text"])

        return cls(df=df, vectorizer=vectorizer, matrix=matrix)

    def search(self, query: str, limit: int = TOP_K_DEFAULT, gender: str | None = None) -> dict[str, Any]:
        limit = max(1, min(limit, 48))
        normalized_query = normalize_text(query)
        expanded_query = expand_query(normalized_query)

        q_vec = self.vectorizer.transform([expanded_query])
        cosine_scores = linear_kernel(q_vec, self.matrix).flatten()

        keyword_scores = self.df.apply(
            lambda row: keyword_overlap_score(row, normalized_query),
            axis=1,
        ).to_numpy(dtype=float)

        intent_scores = self.df["category_family"].map(
            lambda cat: intent_category_boost(cat, normalized_query)
        ).to_numpy(dtype=float)

        total_scores = cosine_scores * 0.65 + keyword_scores * 0.20 + intent_scores * 0.15

        tmp = self.df.copy()
        tmp["score"] = total_scores
        tmp = filter_by_gender(tmp, gender)
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


engine = SearchEngine.load(DATA_PATH)

app = FastAPI(
    title="Product Search API",
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
        "endpoints": [
            "/search?q=outfit elegante ufficio&gender=male",
            "/search?q=look casual weekend&gender=female",
            "/health",
        ],
        "products_loaded": int(len(engine.df)),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "products_loaded": int(len(engine.df))}


@app.get("/search")
def search(
    q: str = Query(..., min_length=2, description="Query utente, es. outfit elegante ufficio"),
    limit: int = Query(TOP_K_DEFAULT, ge=1, le=48),
    gender: str | None = Query(None, description="male, female, unisex"),
) -> dict[str, Any]:
    return engine.search(q, limit=limit, gender=gender)