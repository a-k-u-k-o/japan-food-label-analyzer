"""Compares a product's per-100g nutrition against competitor products in its category.
Also detects the most likely product category from nutrition values and ingredients.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import re
import os


_NUTRIENTS = ["calories", "protein", "fat", "carbohydrates", "sodium", "sugar", "fiber"]

_KEY_ALIASES: Dict[str, List[str]] = {
    "calories":      ["calories", "energy_kcal", "calorie", "kcal"],
    "protein":       ["protein", "protein_g"],
    "fat":           ["fat", "total_fat", "fat_g", "total_fat_g"],
    "carbohydrates": ["carbohydrates", "carbs_g", "total_carbohydrate_g", "carbs"],
    "sodium":        ["sodium", "sodium_g", "salt_equivalent"],
    "sugar":         ["sugar", "sugars", "sugar_g"],
    "fiber":         ["fiber", "dietary_fiber_g", "dietary_fibre", "fibre"],
}

_LOWER_IS_BETTER = {"calories", "fat", "sodium", "sugar", "carbohydrates"}


# Per category: {"label": str, "keywords": [str, ...], "products": [{name, calories, protein, ...}]}
def _load_product_database() -> Dict[str, dict]:
    """Load data/benchmark_products.json, searching upward from this file so the working directory doesn't matter."""
    _here = os.path.dirname(os.path.abspath(__file__))
    # walk up to the project root, where data/ lives
    for _ in range(4):
        candidate = os.path.join(_here, "data", "benchmark_products.json")
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # keys starting with _ are documentation only
            return {k: v for k, v in raw.items() if not k.startswith("_")}
        _here = os.path.dirname(_here)

    raise FileNotFoundError(
        "benchmark_products.json not found. "
        "Expected at: <project_root>/data/benchmark_products.json"
    )


PRODUCT_DATABASE: Dict[str, dict] = _load_product_database()


# Kept so older callers don't break.
def prefetch_category(category: str) -> None:
    """No-op kept for backward compatibility."""
    pass

def is_off_cached(category: str) -> bool:
    """Always False; kept for backward compatibility."""
    return False


def _get_val(nd: dict, metric: str) -> Optional[float]:
    """Return the first parseable value among the metric's key aliases, or None."""
    for alias in _KEY_ALIASES.get(metric, [metric]):
        v = nd.get(alias)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _parse_serving_g(nd: dict) -> Optional[float]:
    """Return the serving size in grams, or None if the label is per-100g or the size is unknown."""
    if nd.get("per_100g"):
        return None

    for key in ("serving_size", "serving_size_g"):
        v = nd.get(key)
        if v is None:
            continue
        s = str(v).strip()

        if re.search(r"100\s*[gml]?\s*[当あ]た[りリ]", s):
            return None

        try:
            g = float(s.replace("g", "").replace("ml", "").strip())
            if 1 <= g <= 500:
                return g
        except (TypeError, ValueError):
            pass

        m = re.search(r"(\d+(?:\.\d+)?)\s*g(?![\w])", s)
        if m:
            g = float(m.group(1))
            if 1 <= g <= 500:
                return g

        m = re.search(r"(\d+(?:\.\d+)?)\s*ml", s)
        if m:
            g = float(m.group(1))
            if 1 <= g <= 500:
                return g

    return None


def _to_per_100g(nd: dict, category: str = "") -> Tuple[dict, float, str]:
    """Scale nutrition values to per-100g; returns (scaled_dict, scale_factor, note).
    A scale_factor of 1.0 means the values were already per-100g or couldn't be scaled."""
    if nd.get("per_100g"):
        return nd, 1.0, "Label is per-100g."

    serving_g = _parse_serving_g(nd)

    if serving_g:
        if abs(serving_g - 100.0) < 0.5:
            return nd, 1.0, "Label is per-100g (100g serving)."
        scale = 100.0 / serving_g
        scaled = {k: v for k, v in nd.items()}
        for m in _NUTRIENTS:
            v = _get_val(nd, m)
            if v is not None:
                scaled[m] = round(v * scale, 2)
        return scaled, scale, f"Scaled from per-serving ({serving_g:.1f}g) → per-100g (×{scale:.2f})."

    cal = _get_val(nd, "calories")
    if cal is not None and cal > 300:
        return nd, 1.0, "Calorie density indicates values are per-100g."

    _CAT_RANGES = {
        "instant_noodles":   (60, 110), "potato_snacks":    (15, 35),
        "chocolate":         ( 4,  15), "beverages":       (100, 500),
        "rice_crackers":     ( 5,  20), "packaged_meals":  (100, 250),
        "supplements":       (20,  60), "soft_candy":      (  3,   8),
        "soup":              (10,  25), "instant_beverages":(2,   20),
    }
    if cal is not None and 0 < cal < 200:
        search = {category: _CAT_RANGES[category]} if category in _CAT_RANGES else _CAT_RANGES
        best_g, best_diff = None, float("inf")
        for _, (lo, hi) in search.items():
            for g in range(lo, hi + 1, max(1, (hi - lo) // 10)):
                cal_100g = cal * (100.0 / g)
                if 80 <= cal_100g <= 700 and abs(cal_100g - 400) < best_diff:
                    best_diff, best_g = abs(cal_100g - 400), g
        if best_g:
            scale = 100.0 / best_g
            scaled = {k: v for k, v in nd.items()}
            for m in _NUTRIENTS:
                v = _get_val(nd, m)
                if v is not None:
                    scaled[m] = round(v * scale, 2)
            return scaled, scale, (
                f"Serving size not on label — inferred ~{best_g}g for {category or 'this category'}. "
                f"Scaled ×{scale:.1f} to per-100g. Verify using Debug JSON."
            )

    return nd, 1.0, (
        "Could not determine serving size — values used as-is. "
        "Verify serving size in Debug JSON."
    )


def normalise_to_100g(nd: dict, category: str = "") -> Tuple[dict, float, str]:
    """Backward-compatible alias for _to_per_100g."""
    return _to_per_100g(nd, category)


def detect_category(nutrition_dict: dict, ingredients_text: str = "") -> str:
    """Return the PRODUCT_DATABASE category key that best fits the product.
    Nutrition heuristics (3-4 pts) outweigh keyword hits (2 pts) to avoid false matches."""
    text = (ingredients_text or "").lower()

    nd, _, _ = _to_per_100g(nutrition_dict)

    cal    = float(_get_val(nd, "calories")      or 0)
    prot   = float(_get_val(nd, "protein")       or 0)
    sodium = float(_get_val(nd, "sodium")        or 0)
    sugar  = float(_get_val(nd, "sugar")         or 0)
    fat    = float(_get_val(nd, "fat")           or 0)
    fiber  = float(_get_val(nd, "fiber")         or 0)
    carbs  = float(_get_val(nd, "carbohydrates") or 0)

    scores: Dict[str, int] = {cat: 0 for cat in PRODUCT_DATABASE}

    # keyword matches, 2 pts each
    for cat, data in PRODUCT_DATABASE.items():
        for kw in data["keywords"]:
            if kw in text:
                scores[cat] += 2

    # nutrition heuristics, 3-4 pts
    if sodium > 4.0:
        scores["instant_noodles"] += 4
        scores["soup"]            += 2

    if prot > 20 and cal < 500:
        scores["supplements"] += 4

    if cal < 60 and sodium < 0.2 and carbs < 12 and fat < 2 and prot < 2:
        scores["beverages"] += 4

    if 280 <= cal <= 490 and carbs > 60 and fiber < 0.5 and sodium < 1.0:
        scores["instant_beverages"] += 4

    if cal < 80 and sodium > 0.3 and prot >= 0.5 and fat < 3 and fiber < 1.5:
        scores["soup"] += 4

    if fat > 18 and cal > 440 and sugar < 20:
        scores["potato_snacks"] += 4

    if fat > 20 and cal > 460 and sugar > 28:
        scores["chocolate"] += 4

    if carbs > 75 and fat < 8 and sodium < 0.15:
        scores["soft_candy"] += 3

    if sugar > 55 and fat < 8:
        scores["soft_candy"] += 3

    if carbs > 65 and fat < 15 and 0.5 < sodium < 2.5 and cal > 350:
        scores["rice_crackers"] += 3

    if 40 < cal < 200 and prot > 1.5 and fiber >= 1.0 and fat < 8:
        scores["packaged_meals"] += 3

    if fat > 15 and cal > 440 and 5 <= sugar <= 45:
        scores["potato_snacks"] += 2

    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else "packaged_meals"


def benchmark(nutrition_dict: dict, category: str) -> dict:
    """Compare a product against its category peers on a per-100g basis.
    Per-serve values are scaled first, since peer serving sizes vary too much to compare directly."""
    if category not in PRODUCT_DATABASE:
        return {"error": f"Unknown category: {category}"}

    peers     = PRODUCT_DATABASE[category]["products"]
    cat_label = PRODUCT_DATABASE[category]["label"]

    serving_g = _parse_serving_g(nutrition_dict)
    our_nd, scale, norm_note = _to_per_100g(nutrition_dict, category)

    _sugar_inferred = False
    if category in ("soft_candy", "chocolate") and _get_val(our_nd, "sugar") is None:
        carbs_val = _get_val(our_nd, "carbohydrates")
        if carbs_val:
            our_nd = dict(our_nd)
            our_nd["sugar"] = round(carbs_val * 0.85, 2)
            _sugar_inferred = True
            norm_note += " Sugar not declared — estimated as 85% of carbohydrates."

    peer_data = [{n: p[n] for n in _NUTRIENTS if n in p} for p in peers]

    comparisons: Dict[str, dict] = {}
    better_count = total_metrics = 0

    for metric in _NUTRIENTS:
        our_val   = _get_val(our_nd, metric)
        peer_vals = [p[metric] for p in peer_data if metric in p and p[metric] is not None]

        if our_val is None or not peer_vals:
            continue

        avg      = sum(peer_vals) / len(peer_vals)
        best     = min(peer_vals) if metric in _LOWER_IS_BETTER else max(peer_vals)
        worst    = max(peer_vals) if metric in _LOWER_IS_BETTER else min(peer_vals)
        diff_pct = ((our_val - avg) / avg * 100) if avg != 0 else 0.0
        better   = (our_val <= avg) if metric in _LOWER_IS_BETTER else (our_val >= avg)

        comparisons[metric] = {
            "our_value":       round(our_val, 2),
            "category_avg":    round(avg, 2),
            "best_in_class":   round(best, 2),
            "worst_in_class":  round(worst, 2),
            "diff_vs_avg_pct": round(diff_pct, 1),
            "better_than_avg": better,
            "n_peers":         len(peer_vals),
            "inferred":        (_sugar_inferred and metric == "sugar"),
        }
        total_metrics += 1
        if better:
            better_count += 1

    strengths  = [m for m, d in comparisons.items() if d["better_than_avg"]]
    weaknesses = [m for m, d in comparisons.items() if not d["better_than_avg"]]

    def _our_margin_vs_peer(pd: dict) -> float:
        """Average normalised advantage over one peer across all metrics; positive means we are better."""
        diffs = []
        for m in _NUTRIENTS:
            our = _get_val(our_nd, m)
            if our is None or m not in pd or pd[m] is None:
                continue
            peer_val = float(pd[m])
            denom    = max(abs(peer_val), abs(our), 0.01)
            raw_diff = (peer_val - our) / denom
            if m not in _LOWER_IS_BETTER:
                raw_diff = -raw_diff
            diffs.append(raw_diff)
        return sum(diffs) / len(diffs) if diffs else 0.0

    peer_margins = [_our_margin_vs_peer(pd) for pd in peer_data]
    worse_peers  = sum(1 for m in peer_margins if m > 0)
    percentile   = round(worse_peers / len(peer_data) * 100) if peer_data else 50

    return {
        "category_label":        cat_label,
        "peers":                 peers,
        "comparisons":           comparisons,
        "percentile":            percentile,
        "better_than_avg_count": better_count,
        "total_metrics":         total_metrics,
        "strengths":             strengths,
        "weaknesses":            weaknesses,
        "data_source":           f"Open Food Facts barcodes + official JP brand pages — {len(peers)} verified products",
        "n_live_peers":          0,
        "normalisation_note":    norm_note,
        "scale_factor":          scale,
        "sugar_inferred":        _sugar_inferred,
    }