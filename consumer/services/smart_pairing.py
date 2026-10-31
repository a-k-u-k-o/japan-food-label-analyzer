"""Suggests products from the database that best complement a scanned product toward a single-meal DRI target.
Ranks candidates by a weighted composite score plus Pareto tiers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# composite = W_NUTRITION*l2 + W_DIET*diet_bonus - W_ADDITIVES*additive_penalty - W_SALT*salt_penalty
W_NUTRITION = 0.60
W_DIET      = 0.15
W_ADDITIVES = 0.15
W_SALT      = 0.10

# salt is scored separately from the L2 gap vector
L2_NUTRIENTS      = ["calories", "protein_g", "fat_g", "carbohydrate_g"]
PAIRING_NUTRIENTS = L2_NUTRIENTS + ["salt_g"]

# a candidate's salt may not exceed this multiple of the remaining salt gap
SALT_EXCESS_MULTIPLIER = 3.0
SALT_FLOOR_CAP_G       = 0.3   # g; floor when the gap is zero

HIGH_RISK_LABELS = {"high", "medium"}

# maps DB nutrition labels to normalised keys
_NUTR_KEY_MAP = {
    "energy": "calories", "calories": "calories",
    "protein": "protein_g",
    "fat": "fat_g", "total fat": "fat_g",
    "carbohydrate": "carbohydrate_g", "carbohydrates": "carbohydrate_g",
    "total carbohydrate": "carbohydrate_g",
    "sugar": "sugar_g", "sugars": "sugar_g",
    "fibre": "fibre_g", "dietary fibre": "fibre_g", "fiber": "fibre_g",
    "salt": "salt_g", "salt equivalent": "salt_g", "sodium": "salt_g",
    "saturated fat": "saturated_fat_g", "saturates": "saturated_fat_g",
}


def _strip_num(val_str) -> float | None:
    """Pull the first number out of a value such as "5.8g"."""
    if val_str is None:
        return None
    m = re.search(r"[\d.]+", str(val_str))
    return float(m.group()) if m else None


def parse_db_nutrition(nutrition_cell) -> dict:
    """Parse a DB Nutrition JSON cell into a normalised dict."""
    result = {k: None for k in [
        "calories", "protein_g", "fat_g", "saturated_fat_g",
        "carbohydrate_g", "sugar_g", "fibre_g", "salt_g",
    ]}
    if pd.isna(nutrition_cell):
        return result
    try:
        raw = json.loads(str(nutrition_cell))
    except Exception:
        return result
    for raw_key, raw_val in raw.items():
        norm_key = _NUTR_KEY_MAP.get(raw_key.strip().lower())
        if norm_key:
            result[norm_key] = _strip_num(raw_val)
    return result


def _build_allergen_scanner(token_to_allergen_id, id_to_allergen):
    """Return a closure that scans a text block for allergens."""
    sorted_toks = sorted(token_to_allergen_id.keys(), key=len, reverse=True)

    def _scan(text: str) -> list[str]:
        found: set[str] = set()
        text_l = text.lower()
        for token in sorted_toks:
            if token in text_l:
                aid   = token_to_allergen_id[token]
                aname = id_to_allergen.get(aid)
                if aname:
                    found.add(aname)
        return sorted(found)

    return _scan


def detect_allergens_db(
    ingredients_text: str,
    token_to_allergen_id: dict,
    id_to_allergen: dict,
) -> tuple[list[str], list[str]]:
    """Split ingredient text into in-product and cross-contact parts and scan each for allergens.
    Returns (in_product, cross_contact)."""
    if not ingredients_text or pd.isna(ingredients_text):
        return [], []

    text_lower = str(ingredients_text).lower()
    _scan = _build_allergen_scanner(token_to_allergen_id, id_to_allergen)

    cross_markers = [
        r"\(some ingredients[:\s]", r"\(contains[:\s]",
        r"\(may contain[:\s]",      r"\(traces of[:\s]",
    ]
    cross_start = len(text_lower)
    for marker in cross_markers:
        m = re.search(marker, text_lower)
        if m and m.start() < cross_start:
            cross_start = m.start()

    return _scan(text_lower[:cross_start]), _scan(text_lower[cross_start:])


def detect_additives_db(ingredients_text: str, additives_df: pd.DataFrame) -> list[dict]:
    """Find additives from the additives table named in the ingredient text.
    Returns dicts with name and risk_level."""
    if not ingredients_text or pd.isna(ingredients_text):
        return []
    text_lower = str(ingredients_text).lower()

    # the name column differs between CSV versions
    name_col = next(
        (c for c in additives_df.columns
        if c.strip().lower() in ("name_en", "name", "additive_name", "additive")),
        additives_df.columns[0],
    )

    hits: list[dict] = []
    seen: set[str]   = set()
    for _, row in additives_df.sort_values(
        by=name_col, key=lambda s: s.str.len(), ascending=False
    ).iterrows():
        name = str(row.get(name_col, "") or "").strip().lower()
        if not name or name in seen:
            continue
        if name in text_lower:
            hits.append({
                "name":       str(row.get(name_col, name)),
                "risk_level": str(row.get("risk_level", row.get("Risk_Level", row.get("risk", "")))),
            })
            seen.add(name)
    return hits


def _dri_per_meal(dri_raw: dict, meals: int = 3) -> dict:
    """Split the daily DRI targets evenly across meals."""
    carb_mid = (dri_raw["carb_min_g"] + dri_raw["carb_max_g"]) / 2
    fat_mid  = (dri_raw["fat_min_g"]  + dri_raw["fat_max_g"])  / 2
    return {
        "calories":       dri_raw["calories"]  / meals,
        "protein_g":      dri_raw["protein_g"] / meals,
        "fat_g":          fat_mid              / meals,
        "carbohydrate_g": carb_mid             / meals,
        "salt_g":         dri_raw["sodium_g"]  / meals,
    }


def _l2_nutrition_score(gap_vec: np.ndarray, product_vec: np.ndarray) -> float:
    """1 minus the normalised L2 distance between the nutrient gap and what the product supplies (capped at the gap)."""
    capped       = np.minimum(product_vec, gap_vec)
    distance     = np.linalg.norm(gap_vec - capped)
    max_distance = np.linalg.norm(gap_vec)
    if max_distance < 1e-9:
        return 1.0
    return 1.0 - (distance / max_distance)


def _diet_bonus(scanned_diet: dict, db_ingredients: str, diet_lists: dict) -> float:
    """1.0 if the product also fits a diet the scanned product passed, else 0.0."""
    passing = [cat for cat, res in scanned_diet.items() if res.get("status") == "yes"]
    if not passing or not db_ingredients:
        return 0.0
    text_lower = str(db_ingredients).lower()
    for cat in passing:
        blocklist = diet_lists.get(f"{cat}_block", [])
        if not any(b.lower() in text_lower for b in blocklist):
            return 1.0
    return 0.0


def _additive_penalty(additives: list[dict]) -> float:
    """Penalty in [0, 1) that saturates with the count of high/medium-risk additives."""
    flagged = sum(
        1 for a in additives
        if str(a.get("risk_level", "")).lower() in HIGH_RISK_LABELS
    )
    return 1.0 - np.exp(-0.4 * flagged)


def _salt_penalty(db_salt: float | None, salt_gap: float) -> float:
    """Penalty in [0, 1] for salt above the remaining salt gap."""
    if (salt_gap or 0.0) <= 0.0:
        db_salt_val = db_salt or 0.0
        return 0.0 if db_salt_val < 0.3 else min(db_salt_val / 0.5, 1.0)
    excess = max((db_salt or 0.0) - salt_gap, 0.0)
    return min(excess / salt_gap, 1.0)


def _is_dominated(a: dict, b: dict) -> bool:
    """True if b is at least as good as a on every score component and strictly better on one."""
    def _obj(x):
        sb = x["score_breakdown"]
        return (
             sb["l2_nutrition"],
             sb["diet_bonus"],
            -sb["additive_penalty"],
            -sb["salt_penalty"],
        )
    ao, bo = _obj(a), _obj(b)
    return all(bv >= av for av, bv in zip(ao, bo)) and any(bv > av for av, bv in zip(ao, bo))


def smart_pair(
    scanned_nutrition: dict,
    scanned_diet:      dict,
    dri_raw:           dict,
    food_db:           pd.DataFrame,
    token_to_allergen_id: dict,
    id_to_allergen:    dict,
    additives_df:      pd.DataFrame,
    diet_lists:        dict,
    user_allergens:    list[str] | None = None,
    top_n:             int = 5,
    meals:             int = 3,
    product_name:      str = "Scanned product",
) -> dict:
    """Rank database products that best fill the per-meal nutrient gap left by the scanned product.
    Returns the meal target, remaining gap, top_n pairings, excluded count and user allergens."""
    user_allergens = [a.strip().lower() for a in (user_allergens or [])]
    meal_target    = _dri_per_meal(dri_raw, meals)

    def _gap(target, consumed):
        return max((target or 0.0) - (consumed or 0.0), 0.0)

    gap_after_scan = {
        n: _gap(meal_target.get(n), scanned_nutrition.get(n))
        for n in PAIRING_NUTRIENTS
    }

    gap_vec_norm = np.array([
        gap_after_scan[n] / max(meal_target.get(n, 1.0), 1e-9)
        for n in L2_NUTRIENTS
    ])

    candidates: list[dict] = []

    for _, row in food_db.iterrows():
        db_nutr        = parse_db_nutrition(row.get("Nutrition"))
        db_ingredients = str(row.get("Ingredients", "") or "")

        in_prod, cross = detect_allergens_db(
            db_ingredients, token_to_allergen_id, id_to_allergen
        )
        all_allergens = sorted(set(in_prod + cross))
        additives     = detect_additives_db(db_ingredients, additives_df)

        # exclude products containing the user's allergens
        excluded = False
        exclusion_reason = ""
        if user_allergens:
            conflicts = [a for a in all_allergens if a in user_allergens]
            if conflicts:
                excluded = True
                exclusion_reason = f"Contains allergen(s): {', '.join(conflicts)}"

        # exclude products over the salt cap
        if not excluded:
            salt_gap   = gap_after_scan.get("salt_g", 0.0)
            db_salt    = db_nutr.get("salt_g") or 0.0
            salt_cap_g = max(salt_gap * SALT_EXCESS_MULTIPLIER, SALT_FLOOR_CAP_G)
            if db_salt > salt_cap_g:
                excluded = True
                exclusion_reason = (
                    f"Salt {db_salt:.2f}g exceeds cap {salt_cap_g:.2f}g"
                )

        product_vec_norm = np.array([
            (db_nutr.get(n) or 0.0) / max(meal_target.get(n, 1.0), 1e-9)
            for n in L2_NUTRIENTS
        ])
        l2_score    = _l2_nutrition_score(gap_vec_norm, product_vec_norm)
        d_bonus     = _diet_bonus(scanned_diet, db_ingredients, diet_lists)
        add_penalty = _additive_penalty(additives)
        s_penalty   = _salt_penalty(db_nutr.get("salt_g"), gap_after_scan.get("salt_g", 0.0))

        composite = (
              W_NUTRITION * l2_score
            + W_DIET      * d_bonus
            - W_ADDITIVES * add_penalty
            - W_SALT      * s_penalty
        )

        gap_if_added = {
            n: round(max(gap_after_scan[n] - (db_nutr.get(n) or 0.0), 0.0), 2)
            for n in PAIRING_NUTRIENTS
        }

        candidates.append({
            "product_id": int(row.get("Product-ID") or 0),
            "name_en":     str(row.get("Name-EN", "")),
            "name_jp":     str(row.get("Name-JP", "")),
            "category":    str(row.get("Category", "")),
            "nutrition":   db_nutr,
            "gap_before":  {k: round(v, 2) for k, v in gap_after_scan.items()},
            "gap_after":   gap_if_added,
            "score_breakdown": {
                "composite":        round(composite,    4),
                "l2_nutrition":     round(l2_score,     4),
                "diet_bonus":       round(d_bonus,      4),
                "additive_penalty": round(add_penalty,  4),
                "salt_penalty":     round(s_penalty,    4),
            },
            "allergens": {
                "in_product":    in_prod,
                "cross_contact": cross,
            },
            "additives":        additives,
            "excluded":         excluded,
            "exclusion_reason": exclusion_reason,
        })

    eligible   = [c for c in candidates if not c["excluded"]]
    ineligible = [c for c in candidates if     c["excluded"]]

    # Pareto tier: 1 + number of candidates that dominate this one
    if eligible:
        n     = len(eligible)
        tiers = [1] * n
        for i in range(n):
            for j in range(n):
                if i != j and _is_dominated(eligible[i], eligible[j]):
                    tiers[i] += 1
        for c, t in zip(eligible, tiers):
            c["pareto_tier"] = t
        eligible.sort(key=lambda x: (x["pareto_tier"], -x["score_breakdown"]["composite"]))

    return {
        "scanned_product": product_name,
        "meal_dri_target": {k: round(v, 2) for k, v in meal_target.items()},
        "gap_after_scan":  {k: round(v, 2) for k, v in gap_after_scan.items()},
        "top_pairings":    eligible[:top_n],
        "excluded_count":  len(ineligible),
        "user_allergens":  user_allergens,
    }