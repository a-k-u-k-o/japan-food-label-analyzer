"""Rule-based health classification from the nutrition values found on a label.
Missing nutrients are skipped rather than filled with defaults."""

from typing import Optional

# Per-serving thresholds, based on 食事摂取基準2020
THRESHOLDS = {
    "calories": {
        "low":    (None, 200),
        "medium": (200,  400),
        "high":   (400,  None),
        "unit":   "kcal",
        "weight": 2,        # importance in the weighted score
        "invert": False,    # False = high is bad
    },
    "fat": {
        "low":    (None, 5),
        "medium": (5,    15),
        "high":   (15,   None),
        "unit":   "g",
        "weight": 1,
        "invert": False,
    },
    "carbohydrates": {
        "low":    (None, 15),
        "medium": (15,   45),
        "high":   (45,   None),
        "unit":   "g",
        "weight": 1,
        "invert": False,
    },
    "sodium": {
        "low":    (None, 0.5),
        "medium": (0.5,  1.5),
        "high":   (1.5,  None),
        "unit":   "g",
        "weight": 3,        # sodium counts most
        "invert": False,
    },
    "sugar": {
        "low":    (None, 5),
        "medium": (5,    15),
        "high":   (15,   None),
        "unit":   "g",
        "weight": 2,
        "invert": False,
    },
    "protein": {
        "low":    (None, 3),
        "medium": (3,    10),
        "high":   (10,   None),
        "unit":   "g",
        "weight": 1,
        "invert": True,     # high is good
    },
    "fiber": {
        "low":    (None, 1),
        "medium": (1,    4),
        "high":   (4,    None),
        "unit":   "g",
        "weight": 1,
        "invert": True,     # high is good
    },
}

# Points per level; inverted nutrients (protein, fiber) score the other way
SCORE_MAP_NORMAL = {"low": 100, "medium": 60, "high": 10}
SCORE_MAP_INVERT = {"low": 10,  "medium": 60, "high": 100}


def classify_level(nutrient: str, value: float) -> str:
    """Return 'low', 'medium', or 'high' for a nutrient value."""
    if nutrient not in THRESHOLDS:
        return "unknown"
    t = THRESHOLDS[nutrient]
    low_max    = t["low"][1]
    medium_max = t["medium"][1]
    if low_max is not None and value <= low_max:
        return "low"
    if medium_max is not None and value <= medium_max:
        return "medium"
    return "high"


def classify_health(nutrition: dict) -> dict:
    """Score and classify a product using only the nutrients present in the dict.
    Returns verdict, health_score, confidence, nutrition_levels, reasons and found/missing lists."""
    levels   = {}
    scores   = []
    weights  = []
    reasons  = []
    found    = []
    missing  = []

    for nutrient, config in THRESHOLDS.items():
        val = nutrition.get(nutrient)

        if val is None:
            missing.append(nutrient)
            continue

        found.append(nutrient)
        level = classify_level(nutrient, val)
        levels[nutrient] = level

        if config["invert"]:
            score = SCORE_MAP_INVERT[level]
        else:
            score = SCORE_MAP_NORMAL[level]

        scores.append(score * config["weight"])
        weights.append(config["weight"])

        unit = config["unit"]
        if nutrient == "calories":
            if level == "low":
                reasons.append(f"Positive: Low calorie ({val:.0f} kcal) — good for weight management")
            elif level == "high":
                reasons.append(f"Concern: High calories ({val:.0f} kcal) — nearly 25% of daily intake")

        elif nutrient == "sodium":
            if level == "low":
                reasons.append(f"Positive: Very low sodium ({val:.2f}g) — heart-friendly")
            elif level == "medium":
                reasons.append(f"Caution: Moderate sodium ({val:.2f}g) — monitor daily intake")
            elif level == "high":
                reasons.append(f"Concern: High sodium ({val:.2f}g) — exceeds per-meal recommendation")

        elif nutrient == "sugar":
            if level == "low":
                reasons.append(f"Positive: Very low sugar ({val:.1f}g)")
            elif level == "medium":
                reasons.append(f"Caution: Moderate sugar ({val:.1f}g)")
            elif level == "high":
                reasons.append(f"Concern: High sugar ({val:.1f}g) — may spike blood glucose")

        elif nutrient == "fat":
            if level == "low":
                reasons.append(f"Positive: Low fat ({val:.1f}g)")
            elif level == "high":
                reasons.append(f"Concern: High fat ({val:.1f}g)")

        elif nutrient == "protein":
            if level == "high":
                reasons.append(f"Positive: High protein ({val:.1f}g) — supports muscle health")
            elif level == "low":
                reasons.append(f"Note: Low protein ({val:.1f}g)")

        elif nutrient == "fiber":
            if level == "high":
                reasons.append(f"Positive: Good fiber content ({val:.1f}g) — supports digestion")
            elif level == "low" and val < 0.5:
                reasons.append(f"Note: Very low fiber ({val:.1f}g)")

    if not scores:
        return {
            "verdict":           "unknown",
            "health_score":      None,
            "confidence":        0,
            "nutrition_levels":  {},
            "reasons":           ["Could not extract enough nutrition data to classify."],
            "nutrients_found":   [],
            "nutrients_missing": list(THRESHOLDS.keys()),
        }

    health_score = int(sum(scores) / sum(weights))

    if health_score >= 75:
        verdict = "healthy"
    elif health_score >= 45:
        verdict = "moderate"
    else:
        verdict = "concerning"

    # high sodium or sugar always forces a concerning verdict
    if levels.get("sodium") == "high" or levels.get("sugar") == "high":
        verdict = "concerning"
        health_score = min(health_score, 35)

    # confidence: share of key nutrients found, plus a small bonus per extra nutrient
    key_nutrients = ["calories", "sodium", "fat", "sugar"]
    found_key = sum(1 for n in key_nutrients if nutrition.get(n) is not None)
    confidence = int((found_key / len(key_nutrients)) * 100)

    bonus = min(20, len(found) * 3)
    confidence = min(100, confidence + bonus)

    if not reasons:
        reasons.append("Nutritional values are within acceptable ranges.")

    return {
        "verdict":           verdict,
        "health_score":      health_score,
        "confidence":        confidence,
        "nutrition_levels":  levels,
        "reasons":           reasons,
        "nutrients_found":   found,
        "nutrients_missing": missing,
    }


def generate_summary(nutrition: dict, verdict: str) -> str:
    """Return a short plain-English summary of the product for the given verdict."""
    cal = nutrition.get("calories")
    na  = nutrition.get("sodium")
    sug = nutrition.get("sugar")
    fat = nutrition.get("fat")

    parts = []
    if cal is not None:
        parts.append(f"{cal:.0f} kcal")
    if na is not None:
        parts.append(f"{na:.2f}g sodium")
    if sug is not None:
        parts.append(f"{sug:.1f}g sugar")
    if fat is not None:
        parts.append(f"{fat:.1f}g fat")

    nutrition_str = ", ".join(parts) if parts else "limited nutritional data"

    verdict_text = {
        "healthy":    "appears to be a healthy choice",
        "moderate":   "has moderate nutritional values",
        "concerning": "raises some nutritional concerns",
        "unknown":    "could not be fully classified",
    }.get(verdict, "has been analyzed")

    summary = f"This product {verdict_text} per serving. It contains {nutrition_str}. "

    if verdict == "healthy":
        summary += "Suitable for regular consumption as part of a balanced diet."
    elif verdict == "moderate":
        summary += "Moderate consumption is recommended, especially for those monitoring sodium or sugar intake."
    elif verdict == "concerning":
        summary += "Consume sparingly — particularly for individuals with hypertension, diabetes, or weight management goals."
    else:
        summary += "Please check the raw OCR output for more details."

    return summary


if __name__ == "__main__":
    cases = [
        ("Full milk label",
         {"calories": 137, "protein": 6.8, "fat": 7.8,
          "carbohydrates": 9.9, "sodium": 0.21, "calcium": 227}),
        ("High sodium snack",
         {"calories": 280, "sodium": 2.1, "fat": 12.0}),
        ("Only calories known",
         {"calories": 59}),
        ("Empty — nothing found",
         {}),
        ("Very healthy",
         {"calories": 45, "protein": 8, "fat": 0.5,
          "sodium": 0.05, "sugar": 1.0, "fiber": 4.0}),
    ]

    for desc, nutr in cases:
        r = classify_health(nutr)
        print(f"\n{desc}")
        print(f"  Verdict:    {r['verdict']}")
        print(f"  Score:      {r['health_score']}")
        print(f"  Confidence: {r['confidence']}%")
        print(f"  Found:      {r['nutrients_found']}")
        print(f"  Missing:    {r['nutrients_missing']}")
