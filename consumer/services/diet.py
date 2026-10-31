"""Rule-based diet classification (vegan, vegetarian, keto, diabetic) for Japanese food labels.
Each verdict is a dict with status (yes/no/caution/uncertain), label, reasons and positives."""

from __future__ import annotations
import json
import re
from pathlib import Path

from pathlib import Path

_DATA = Path(__file__).resolve().parents[2] / "data" / "diet_blocklists.json"
_lists: dict = {}


def _load():
    """Lazily load the diet blocklists JSON into the module-level cache."""
    global _lists
    if not _lists:
        with open(_DATA, encoding="utf-8") as f:
            _lists = json.load(f)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split an ingredient string on commas and brackets into clean tokens."""
    text = text.lower()
    text = re.sub(r"[()（）\[\]{}]", " ", text)
    tokens = [t.strip().strip(".,;:") for t in re.split(r"[,、/]", text)]
    return [t for t in tokens if len(t) > 1]


def _match_any(tokens: list[str], blocklist: list[str]) -> list[str]:
    """Return blocklist items found in tokens (substring match, near-duplicates collapsed)."""
    hits = []
    matched_tokens = set()
    # Longest-first so specific terms win over generic ones
    for block in sorted(blocklist, key=len, reverse=True):
        for tok in tokens:
            if tok in matched_tokens:
                continue
            if block in tok or tok in block:
                # "sugar" + "white sugar" -> keep only "white sugar"
                if not any(block in existing or existing in block for existing in hits):
                    hits.append(block)
                matched_tokens.add(tok)
                break
    return hits


def _carbs_from_nutrition(nutrition: dict | None) -> float | None:
    """Return carbohydrate grams from a nutrition dict, or None if missing."""
    if not nutrition:
        return None
    for key in ["carbohydrate_g", "carbohydrate", "carbs"]:
        val = nutrition.get(key)
        if val is not None:
            try:
                # values may carry units, e.g. "5.8g"
                return float(re.search(r"\d+\.?\d*", str(val)).group())
            except (AttributeError, ValueError):
                pass
    return None


def _classify_vegan(tokens: list[str], allergens: list[str]) -> dict:
    """Vegan verdict from ingredient tokens and detected allergens."""
    _load()
    block_hits    = _match_any(tokens, _lists["vegan_block"])
    uncertain_hits = _match_any(tokens, _lists["vegan_uncertain"])

    animal_allergens = [a for a in allergens
                        if a in {"egg","milk","shrimp","crab","salmon","mackerel",
                                 "squid","abalone","salmon roe","beef","pork","chicken"}]
    all_blocks = list(dict.fromkeys(block_hits + animal_allergens))

    if all_blocks:
        return {
            "status":    "no",
            "label":     "Not Vegan",
            "reasons":   [f"Contains {h.title()}" for h in all_blocks[:3]],
            "positives": [],
        }
    if uncertain_hits:
        return {
            "status":    "uncertain",
            "label":     "Possibly Vegan",
            "reasons":   [f"Ambiguous ingredient: {h.title()}" for h in uncertain_hits[:2]],
            "positives": ["No obvious animal products detected"],
        }
    return {
        "status":    "yes",
        "label":     "Vegan-Friendly",
        "reasons":   [],
        "positives": ["No animal-derived ingredients detected"],
    }


def _classify_vegetarian(tokens: list[str], allergens: list[str]) -> dict:
    """Vegetarian verdict (meat and fish excluded, dairy and egg allowed)."""
    _load()
    block_hits     = _match_any(tokens, _lists["vegetarian_block"])
    uncertain_hits = _match_any(tokens, _lists["vegetarian_uncertain"])

    meat_fish_allergens = [a for a in allergens
                           if a in {"shrimp","crab","salmon","mackerel","squid",
                                    "abalone","salmon roe","beef","pork","chicken"}]
    all_blocks = list(dict.fromkeys(block_hits + meat_fish_allergens))

    if all_blocks:
        return {
            "status":    "no",
            "label":     "Not Vegetarian",
            "reasons":   [f"Contains {h.title()}" for h in all_blocks[:3]],
            "positives": [],
        }
    if uncertain_hits:
        return {
            "status":    "uncertain",
            "label":     "Possibly Vegetarian",
            "reasons":   [f"Ambiguous: {h.title()}" for h in uncertain_hits[:2]],
            "positives": ["No obvious meat or fish detected"],
        }
    return {
        "status":    "yes",
        "label":     "Vegetarian-Friendly",
        "reasons":   [],
        "positives": ["No meat or fish ingredients detected"],
    }


def _classify_keto(tokens: list[str], nutrition: dict | None) -> dict:
    """Keto verdict from per-serving carbs (if known) and ingredient blocklists."""
    _load()
    carbs = _carbs_from_nutrition(nutrition)
    limit = _lists.get("keto_carb_limit_g", 10)

    block_hits   = _match_any(tokens, _lists["keto_block"])
    caution_hits = _match_any(tokens, _lists["keto_caution"])

    reasons   = []
    positives = []

    if carbs is not None:
        if carbs > limit:
            reasons.append(f"Carbs: {carbs}g per serving (limit ~{limit}g)")
        else:
            positives.append(f"Low carbs: {carbs}g per serving")

    if block_hits:
        reasons += [f"Contains {h.title()}" for h in block_hits[:3]]

    if reasons:
        return {
            "status":    "no",
            "label":     "Not Keto",
            "reasons":   reasons[:3],
            "positives": positives,
        }
    if caution_hits:
        return {
            "status":    "caution",
            "label":     "Keto — Use Caution",
            "reasons":   [f"Check quantity: {h.title()}" for h in caution_hits[:2]],
            "positives": positives or ["No high-carb ingredients found"],
        }
    if carbs is None and not block_hits:
        return {
            "status":    "uncertain",
            "label":     "Possibly Keto",
            "reasons":   ["No nutrition data to verify carb count"],
            "positives": ["No high-carb ingredients detected in list"],
        }
    return {
        "status":    "yes",
        "label":     "Keto-Friendly",
        "reasons":   [],
        "positives": positives or ["Low-carb ingredients, no sugar detected"],
    }


def _classify_diabetic(tokens: list[str], nutrition: dict | None) -> dict:
    """Diabetic-friendliness verdict from per-serving carbs and sugar blocklists."""
    _load()
    carbs         = _carbs_from_nutrition(nutrition)
    caution_limit = _lists.get("diabetic_carb_caution_g", 30)
    block_limit   = _lists.get("diabetic_carb_block_g", 50)

    block_hits        = _match_any(tokens, _lists["diabetic_block"])
    caution_hits      = _match_any(tokens, _lists["diabetic_caution"])
    friendly_sweeteners = _match_any(tokens, _lists["diabetic_friendly_sweeteners"])

    reasons   = []
    positives = []

    if carbs is not None:
        if carbs >= block_limit:
            reasons.append(f"High carbs: {carbs}g per serving")
        elif carbs >= caution_limit:
            caution_hits = caution_hits or ["moderate carb content"]
        else:
            positives.append(f"Moderate carbs: {carbs}g per serving")

    if block_hits:
        reasons += [f"Contains {h.replace('-',' ').title()}" for h in block_hits[:3]]

    if friendly_sweeteners:
        positives.append(f"Uses low-GI sweetener: {friendly_sweeteners[0].title()}")

    if reasons:
        return {
            "status":    "no",
            "label":     "Not Diabetic-Friendly",
            "reasons":   reasons[:3],
            "positives": positives,
        }
    if caution_hits:
        return {
            "status":    "caution",
            "label":     "Diabetic — Use Caution",
            "reasons":   [f"Monitor intake: {h.replace('-',' ').title()}"
                          for h in caution_hits[:2]],
            "positives": positives or [],
        }
    return {
        "status":    "yes",
        "label":     "Diabetic-Friendly",
        "reasons":   [],
        "positives": positives or ["No high-GI sugars or excessive carbs detected"],
    }


def classify_diet(
    ingredients_flat: str,
    nutrition: dict | None = None,
    allergens: list[str] | None = None,
) -> dict:
    """Classify a product as vegan, vegetarian, keto and diabetic-friendly.
    Takes the flat English ingredient string, optional nutrition dict and allergen names."""
    tokens   = _tokenize(ingredients_flat)
    allergens = [a.lower() for a in (allergens or [])]

    return {
        "vegan":      _classify_vegan(tokens, allergens),
        "vegetarian": _classify_vegetarian(tokens, allergens),
        "keto":       _classify_keto(tokens, nutrition),
        "diabetic":   _classify_diabetic(tokens, nutrition),
    }