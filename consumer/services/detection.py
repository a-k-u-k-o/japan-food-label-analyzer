"""Allergen and additive detection from ingredient text.
Matches tokens against the allergen and Japanese food additive CSVs in data/."""

import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

allergen_master   = pd.read_csv(DATA_DIR / "allergen_master.csv")
allergen_tokens    = pd.read_csv(DATA_DIR / "allergen_tokens.csv")
ingredient_sources = pd.read_csv(DATA_DIR / "ingredient_sources.csv")
additives_df       = pd.read_csv(DATA_DIR / "japanese_food_additives.csv", on_bad_lines="skip")

id_to_allergen: dict[int, str] = dict(zip(
    allergen_master["allergen_id"].astype(int),
    allergen_master["canonical_name"].str.lower().str.strip(),
))

token_to_allergen_id: dict[str, int] = {}
for _, row in allergen_tokens.iterrows():
    if pd.notna(row["allergen_id"]) and pd.notna(row["token"]):
        token_to_allergen_id[str(row["token"]).lower().strip()] = int(row["allergen_id"])

for _, row in ingredient_sources.iterrows():
    if pd.notna(row["allergen_id"]) and pd.notna(row["ingredient"]):
        key = str(row["ingredient"]).lower().strip()
        if key not in token_to_allergen_id:
            token_to_allergen_id[key] = int(row["allergen_id"])

# Longest-first so multi-word phrases match before single words
sorted_tokens = sorted(token_to_allergen_id.keys(), key=len, reverse=True)

additives_df["name_normalized"] = additives_df["name"].str.lower().str.strip()

STOPWORDS = {
    "water", "salt", "sugar", "oil", "flour", "starch", "extract", "powder",
    "acid", "colour", "color", "red", "yellow", "blue", "black", "white",
    "green", "dry", "dried", "raw", "natural", "modified", "refined",
    "processed", "organic", "pure", "sauce",
}

# Japan's mandatory-labelling allergens (the original 7 plus walnut)
JP_MANDATORY = {"egg", "milk", "wheat", "buckwheat", "peanut", "shrimp", "crab", "walnut"}


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _expand_parens(text: str) -> list[str]:
    """Split on commas, pulling parenthesised sub-ingredients out as separate tokens."""
    tokens = []
    for group in re.findall(r"\(([^)]+)\)", text):
        tokens.extend(t.strip() for t in group.split(","))
    outer = re.sub(r"\([^)]+\)", "", text)
    tokens.extend(t.strip() for t in outer.split(","))
    return [t for t in tokens if t]


def _tokenize(raw: str) -> list[str]:
    """Break a raw ingredient string into normalised tokens."""
    cleaned = _normalize(raw)
    tokens  = _expand_parens(cleaned)
    final: list[str] = []
    for t in tokens:
        final.extend(x.strip() for x in t.split(";"))
    return [t for t in final if len(t) > 1]


def detect_allergens(ingredient_text: str, return_df: bool = False):
    """Detect allergens in an ingredient string.
    Returns a sorted list of names, or a match DataFrame if return_df=True."""
    tokens  = _tokenize(ingredient_text)
    results = []
    seen: set = set()

    for token in tokens:
        if token in token_to_allergen_id:
            aid = token_to_allergen_id[token]
            key = (token, aid)
            if key not in seen:
                seen.add(key)
                results.append({
                    "ingredient_token": token,
                    "matched_term":     token,
                    "allergen_id":      aid,
                    "allergen_name":    id_to_allergen.get(aid, "unknown"),
                    "match_type":       "exact",
                })
        else:
            for known in sorted_tokens:
                if known in token and known not in STOPWORDS and len(known) >= 3:
                    aid = token_to_allergen_id[known]
                    key = (token, aid)
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            "ingredient_token": token,
                            "matched_term":     known,
                            "allergen_id":      aid,
                            "allergen_name":    id_to_allergen.get(aid, "unknown"),
                            "match_type":       "substring",
                        })
                    break

    empty_cols = ["ingredient_token", "matched_term", "allergen_id", "allergen_name", "match_type"]
    if not results:
        return pd.DataFrame(columns=empty_cols) if return_df else []

    df = (
        pd.DataFrame(results)
        .drop_duplicates(subset=["ingredient_token", "allergen_id"])
        .sort_values("allergen_name")
        .reset_index(drop=True)
    )
    return df if return_df else sorted(df["allergen_name"].unique().tolist())


def _name_match(token: str, additive_name: str) -> bool:
    """True if token equals the additive name or is a close prefix match (>=60% length)."""
    if token == additive_name:
        return True
    if token in STOPWORDS or len(token) < 6:
        return False
    if additive_name.startswith(token):
        return len(token) / len(additive_name) >= 0.6
    if token.startswith(additive_name):
        return len(additive_name) / len(token) >= 0.6
    return False


def detect_additives(ingredient_text: str, return_df: bool = False):
    """Detect Japanese food additives in an ingredient string.
    Returns a sorted list of names, or a match DataFrame if return_df=True."""
    tokens  = _tokenize(ingredient_text)
    results = []
    seen: set = set()

    for token in tokens:
        for _, row in additives_df.iterrows():
            additive_id   = str(int(row["additive_id"]))
            additive_name = str(row["name_normalized"]).strip()
            orig_name     = str(row["name"]).strip()
            note          = str(row["note"]).strip()

            id_matched   = bool(re.search(rf"(?<!\d){re.escape(additive_id)}(?!\d)", token))
            name_matched = _name_match(token, additive_name)

            if id_matched or name_matched:
                key = (token, additive_id)
                if key not in seen:
                    seen.add(key)
                    results.append({
                        "ingredient_token": token,
                        "match_type":       "id" if id_matched else "name",
                        "additive_id":      int(row["additive_id"]),
                        "additive_name":    orig_name,
                        "note":             note,
                    })

    empty_cols = ["ingredient_token", "match_type", "additive_id", "additive_name", "note"]
    if not results:
        return pd.DataFrame(columns=empty_cols) if return_df else []

    df = (
        pd.DataFrame(results)
        .drop_duplicates(subset=["additive_name"])
        .sort_values("additive_id")
        .reset_index(drop=True)
    )
    return df if return_df else sorted(df["additive_name"].unique().tolist())


def analyze_ingredients(ingredient_text: str) -> dict:
    """Run both detectors and return allergens, additives (id, name) and the two detail DataFrames."""
    allergen_df = detect_allergens(ingredient_text, return_df=True)
    additive_df = detect_additives(ingredient_text, return_df=True)

    allergens = sorted(allergen_df["allergen_name"].unique().tolist()) if not allergen_df.empty else []
    additives = (
        list(zip(additive_df["additive_id"].tolist(), additive_df["additive_name"].tolist()))
        if not additive_df.empty else []
    )
    seen_ids: set = set()
    unique_additives = []
    for aid, aname in additives:
        if aid not in seen_ids:
            seen_ids.add(aid)
            unique_additives.append((aid, aname))

    return {
        "allergens":   allergens,
        "additives":   unique_additives,
        "allergen_df": allergen_df,
        "additive_df": additive_df,
    }