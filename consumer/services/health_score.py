"""Personalised 0-100 health score for Japanese food products.
Combines ingredient quality, macros, sodium and additive risk; sodium and macro thresholds scale to the user's DRI."""

from __future__ import annotations
import re

# Additive risk tiers: 0 benign, 1 caution, 2 avoid
ADDITIVE_TIERS: dict[str, int] = {
    # benign
    "vitamin c": 0, "ascorbic acid": 0, "tocopherol": 0, "vitamin e": 0,
    "vitamin b1": 0, "vitamin b2": 0, "beta-carotene": 0,
    "lecithin": 0, "pectin": 0, "agar": 0, "xanthan gum": 0,
    "guar gum": 0, "locust bean gum": 0, "gellan gum": 0,
    "citric acid": 0, "lactic acid": 0, "malic acid": 0,
    "calcium carbonate": 0, "sodium bicarbonate": 0,
    "beeswax": 0, "carnauba wax": 0,
    # caution
    "sodium nitrite": 1, "sodium nitrate": 1,
    "potassium sorbate": 1, "sodium benzoate": 1,
    "bha": 1, "bht": 1, "tbhq": 1,
    "carrageenan": 1, "modified starch": 1,
    "high-fructose corn syrup": 1, "glucose syrup": 1, "starch syrup": 1,
    "acesulfame": 1, "sucralose": 1, "aspartame": 1,
    "sodium glutamate": 1, "msg": 1, "monosodium glutamate": 1,
    "caramel color": 1, "caramel colour": 1,
    "artificial flavor": 1, "artificial flavoring": 1,
    # avoid
    "potassium bromate": 2, "brominated vegetable oil": 2,
    "red 40": 2, "yellow 5": 2, "yellow 6": 2, "blue 1": 2, "blue 2": 2,
    "red 3": 2, "titanium dioxide": 2,
    "propyl gallate": 2, "propylene glycol": 2,
    "azodicarbonamide": 2,
    "caramel color iv": 2, "caramel colour iv": 2,
}

RED_FLAG_INGREDIENTS = [
    "sugar", "high-fructose corn syrup", "glucose syrup", "starch syrup",
    "white sugar", "brown sugar", "corn syrup", "fructose", "dextrose",
    "refined flour", "wheat flour", "bleached flour",
    "lard", "shortening", "partially hydrogenated",
    "palm oil", "palm shortening",
    "artificial flavor", "artificial colour", "artificial color",
]

POSITIVE_INGREDIENTS = [
    "whole grain", "whole wheat", "oat", "brown rice", "quinoa",
    "olive oil", "flaxseed", "chia", "walnut", "almond",
    "spinach", "kale", "broccoli", "carrot", "tomato",
    "legume", "lentil", "chickpea", "kidney bean", "edamame",
    "tofu", "natto", "miso", "tempeh",
    "green tea", "matcha",
]


def _tokenize(text: str) -> list[str]:
    """Lowercase and split an ingredient string into clean tokens."""
    text = text.lower()
    text = re.sub(r"[()（）\[\]{}]", " ", text)
    tokens = [t.strip().strip(".,;:") for t in re.split(r"[,、/]", text)]
    return [t for t in tokens if len(t) > 1]


def _get_num(nutrition: dict, *keys) -> float | None:
    """First numeric value found under any of the given keys, else None."""
    for k in keys:
        v = nutrition.get(k)
        if v is not None:
            try:
                return float(re.search(r"\d+\.?\d*", str(v)).group())
            except (AttributeError, ValueError):
                pass
    return None


# Ingredient quality (0-30)
def _score_ingredients(ingredients_flat: str) -> tuple[int, list[str], list[str]]:
    """Score 0-30 for ingredient quality: red-flag ingredients near the top cost points, whole foods add.
    Returns (score, flags, boosts)."""
    tokens = _tokenize(ingredients_flat)
    if not tokens:
        return 15, ["No ingredients data"], []

    score  = 30
    flags  = []
    boosts = []

    for i, tok in enumerate(tokens[:10]):
        for rf in RED_FLAG_INGREDIENTS:
            if rf in tok:
                penalty = 8 if i < 3 else 4
                score  -= penalty
                flags.append(f"{'⚠️ High' if i < 3 else 'Low'} position: {rf.title()}")
                break

    for tok in tokens:
        for pos in POSITIVE_INGREDIENTS:
            if pos in tok:
                score += 2
                boosts.append(f"Contains {pos.title()}")
                break

    if len(tokens) == 1:
        boosts.append("Minimal ingredients")
    elif len(tokens) > 20:
        flags.append("Many ingredients (highly processed)")
        score -= 3

    return max(0, min(30, score)), flags[:4], boosts[:3]


# Macronutrient balance (0-30)
def _score_macros(
    nutrition:    dict,
    dri_raw:      dict | None,
    serving_kcal: float | None,
) -> tuple[int, list[str], list[str]]:
    """Score 0-30 from sugar, saturated fat, fibre and protein, scaled to serving calories when known.
    Returns (score, flags, boosts)."""
    flags  = []
    boosts = []
    earned = 0
    possible = 0
    kcal = serving_kcal

    # sugar (up to 12 pts)
    sugar = _get_num(nutrition, "sugar", "sugar_g", "sugars")
    carb_check = _get_num(nutrition, "carbohydrate", "carbohydrate_g", "carbs")
    if sugar is not None and carb_check is not None and abs(sugar - carb_check) < 0.1:
        sugar = None
    if sugar is not None:
        possible += 12
        threshold = max(5.0, (kcal * 0.10 / 4)) if kcal else 10.0
        if sugar > threshold * 2:
            earned += 0;  flags.append(f"Very high sugar: {sugar}g")
        elif sugar > threshold:
            earned += 6;  flags.append(f"High sugar: {sugar}g")
        else:
            earned += 12; boosts.append(f"Low sugar: {sugar}g")

    # saturated fat (up to 10 pts)
    sat_fat = _get_num(nutrition, "saturated_fat", "saturated_fat_g")
    if sat_fat is not None:
        possible += 10
        threshold = max(2.0, (kcal * 0.10 / 9)) if kcal else 4.0
        if sat_fat > threshold * 2:
            earned += 0;  flags.append(f"Very high saturated fat: {sat_fat}g")
        elif sat_fat > threshold:
            earned += 5;  flags.append(f"High saturated fat: {sat_fat}g")
        else:
            earned += 10; boosts.append(f"Low saturated fat: {sat_fat}g")

    # fibre (up to 5 pts)
    fibre = _get_num(nutrition, "fibre", "fibre_g", "fiber", "dietary_fibre")
    if fibre is not None:
        possible += 5
        if fibre >= 3.0:
            earned += 5; boosts.append(f"Good fibre: {fibre}g")
        elif fibre >= 1.5:
            earned += 3; boosts.append(f"Some fibre: {fibre}g")
        else:
            earned += 1

    # protein (up to 3 pts)
    protein = _get_num(nutrition, "protein", "protein_g")
    if protein is not None:
        possible += 3
        threshold = max(5.0, (kcal * 0.15 / 4)) if kcal else 5.0
        if protein >= threshold:
            earned += 3; boosts.append(f"Good protein: {protein}g")
        else:
            earned += 1

    if possible == 0:
        return 15, ["No nutrition data — macros unscored"], []

    score = round(earned / possible * 30)
    return max(0, min(30, score)), flags[:4], boosts[:4]


# Sodium (0-20)
def _score_sodium(
    nutrition: dict,
    dri_raw:   dict | None,
) -> tuple[int, list[str], list[str]]:
    """Score 0-20 from the share of the daily sodium upper limit (UL) in one serving.
    Returns (score, flags, boosts)."""
    score  = 20
    flags  = []
    boosts = []

    salt = _get_num(nutrition, "salt_g", "salt", "sodium_g", "sodium")
    if salt is None:
        return 10, ["No sodium data — sodium unscored"], []

    # OCR sometimes returns mg instead of g; no serving has more than 10g of salt
    if salt > 10:
        salt = round(salt / 1000, 3)

    # sodium (g) = salt (g) / 2.54
    sodium_g = round(salt / 2.54, 4)

    # daily reference is the sodium UL (2.3 g, adults 19-50), same as dri_raw["sodium_g"]
    daily_sodium_ul = float((dri_raw or {}).get("sodium_g") or 2.3)

    pct_daily = sodium_g / daily_sodium_ul * 100

    # bands as % of the 2.3 g UL: >50% heavy, >25% moderate, >10% mild, else low-sodium boost
    if pct_daily > 50:
        score -= 16
        flags.append(f"Very high sodium: {pct_daily:.0f}% of daily UL")
    elif pct_daily > 25:
        score -= 8
        flags.append(f"Moderate sodium: {pct_daily:.0f}% of daily UL")
    elif pct_daily > 10:
        score -= 3
        flags.append(f"Some sodium: {pct_daily:.0f}% of daily UL")
    else:
        boosts.append(f"Low sodium: {pct_daily:.0f}% of daily UL")

    return max(0, min(20, score)), flags, boosts


# Additive risk (0-20)
def _score_additives(
    additives: list[tuple],
) -> tuple[int, list[str], list[str]]:
    """Score 0-20 by additive risk tier (avoid -8 each, caution -3 each).
    Returns (score, flags, boosts)."""
    if not additives:
        return 20, [], ["No additives detected"]

    score  = 20
    flags  = []
    boosts = []
    tier2_found = []
    tier1_found = []

    for _id, name in additives:
        name_l = name.lower()
        tier   = ADDITIVE_TIERS.get(name_l, 1)
        if tier == 2:
            tier2_found.append(name)
        elif tier == 1:
            tier1_found.append(name)

    score -= len(tier2_found) * 8
    score -= len(tier1_found) * 3

    for n in tier2_found[:2]:
        flags.append(f"Avoid additive: {n}")
    for n in tier1_found[:3]:
        flags.append(f"Caution additive: {n}")

    if not tier2_found and not tier1_found:
        boosts.append("Only benign additives")

    return max(0, min(20, score)), flags[:5], boosts


def compute_health_score(
    ingredients_flat: str,
    nutrition:        dict | None,
    additives:        list[tuple] | None,
    dri_raw:          dict | None = None,
) -> dict:
    """Compute the 0-100 health score: verdict (70+ Healthy, 40-69 Moderate, below 40 Unhealthy), letter grade,
    per-section scores and the top flags and boosts."""
    nutrition = nutrition or {}
    additives = additives or []

    serving_kcal = _get_num(nutrition, "calories", "calories_kcal")

    sa, fa, ba = _score_ingredients(ingredients_flat)
    sb, fb, bb = _score_macros(nutrition, dri_raw, serving_kcal)
    sc, fc, bc = _score_sodium(nutrition, dri_raw)
    sd, fd, bd = _score_additives(additives)

    total = sa + sb + sc + sd

    if total >= 70:   verdict = "Healthy"
    elif total >= 40: verdict = "Moderate"
    else:             verdict = "Unhealthy"

    if total >= 85:   grade = "A"
    elif total >= 70: grade = "B"
    elif total >= 55: grade = "C"
    elif total >= 40: grade = "D"
    else:             grade = "F"

    all_flags  = fa + fb + fc + fd
    all_boosts = ba + bb + bc + bd

    return {
        "score":   total,
        "verdict": verdict,
        "grade":   grade,
        "sections": {
            "ingredients": {"score": sa, "max": 30, "flags": fa, "boosts": ba},
            "macros":      {"score": sb, "max": 30, "flags": fb, "boosts": bb},
            "sodium":      {"score": sc, "max": 20, "flags": fc, "boosts": bc},
            "additives":   {"score": sd, "max": 20, "flags": fd, "boosts": bd},
        },
        "top_flags":  list(dict.fromkeys(all_flags))[:4],
        "top_boosts": list(dict.fromkeys(all_boosts))[:3],
    }