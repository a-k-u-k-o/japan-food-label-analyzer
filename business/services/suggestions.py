"""Generates prioritised label and product improvement suggestions from nutrition, benchmark and compliance results.
Suggestions are enriched via Groq (key from GROQ_API_KEY, loaded with python-dotenv) and fall back to built-in text."""

import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Suggestion:
    """A single improvement suggestion with its rationale, how-to, priority and category."""
    title:    str
    detail:   str
    impact:   str
    how:      str
    priority: str      # "high" | "medium" | "low"
    category: str      # "Compliance" | "Reformulation" | "Positioning"


_groq_client = None

def _init_groq():
    """Create the Groq client once from GROQ_API_KEY and cache it.
    Returns None if the key is missing or the client can't be created."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        _groq_client = Groq(api_key=api_key)
        return _groq_client
    except Exception:
        return None


def _ai_enrich_batch(suggestions_raw: list, nd: dict) -> dict:
    """Enrich all suggestions in a single Groq call; returns {idx: {"impact": str, "how": str}}.
    Returns an empty dict on any failure so callers keep their fallback text."""
    client = _init_groq()
    if client is None:
        return {}

    nutrition_clean = {
        k: v for k, v in nd.items()
        if k in ("calories","protein","fat","carbohydrates","sodium","sugar","fiber")
        and v is not None
    }

    issues_text = "\n".join(
        f"{s['idx']}. [{s['category']}] {s['title']}: {s['detail']}"
        for s in suggestions_raw
    )

    prompt = f"""You are a senior Japanese food product consultant (JAS, MHLW, CAA expert).

Product nutrition per serving: {json.dumps(nutrition_clean)}

For EACH numbered issue below, write:
- impact: 1-2 sentences on commercial importance in Japan (convenience stores, retail, regulations)
- how: 1-2 sentences of concrete implementation steps

Issues:
{issues_text}

Respond ONLY as a JSON object mapping the issue number (as string) to {{impact, how}}.
No markdown. No extra keys. Example format:
{{"1": {{"impact": "...", "how": "..."}}, "2": {{"impact": "...", "how": "..."}}}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a Japanese food regulatory and retail expert. Respond only in valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1200,   # ~90 tokens per suggestion, up to ~13 suggestions
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$",     "", text)
        return json.loads(text)
    except Exception:
        return {}


def _get(d: dict, *keys) -> Optional[float]:
    """Return the first of the given keys that holds a numeric value, or None."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _make_suggestion(title, detail, fallback_impact, fallback_how,
                     priority, category, nd, bench_comparisons,
                     ai_enrichments: dict = None, idx: int = None):
    """Build a Suggestion, using the AI enrichment for idx when present and the fallback text otherwise."""
    ai_impact, ai_how = None, None
    if ai_enrichments and idx is not None:
        entry = ai_enrichments.get(str(idx), {})
        ai_impact = entry.get("impact")
        ai_how    = entry.get("how")

    return Suggestion(
        title    = title,
        detail   = detail,
        impact   = ai_impact if ai_impact else fallback_impact,
        how      = ai_how    if ai_how    else fallback_how,
        priority = priority,
        category = category,
    )


def _compliance_suggestions(compliance_reports: Dict[str, Any],
                             nd: dict) -> List[Suggestion]:
    """Convert compliance errors/warnings into prioritised suggestions."""
    suggestions = []

    for std_key, report in compliance_reports.items():
        for issue in report.issues:
            if issue.severity == "error":
                suggestions.append(_make_suggestion(
                    title    = f"Fix Missing Declaration: {issue.code}",
                    detail   = issue.message,
                    fallback_impact = (
                        "Required for legal sale in Japan. Missing fields can result "
                        "in product recall or regulatory fine from the CAA."
                    ),
                    fallback_how = (
                        "Add the missing nutrient value to the 栄養成分表示 table on packaging. "
                        "Engage a registered nutritional testing laboratory for accurate values."
                    ),
                    priority = "high",
                    category = "Compliance",
                    nd       = nd,
                    bench_comparisons = {},
                ))
            elif issue.severity == "warning":
                suggestions.append(_make_suggestion(
                    title    = f"Address Compliance Warning: {issue.code}",
                    detail   = issue.message,
                    fallback_impact = (
                        "Resolving this avoids regulatory scrutiny and enables "
                        "health-oriented marketing claims."
                    ),
                    fallback_how = (
                        "Review label wording and nutrient quantities against the latest "
                        "CAA / MHLW guidelines before next production run."
                    ),
                    priority = "medium",
                    category = "Compliance",
                    nd       = nd,
                    bench_comparisons = {},
                ))

    return suggestions


def _sodium_suggestions(nd: dict, bench_report: dict) -> List[Suggestion]:
    """Suggest sodium reduction when salt equivalent exceeds 3 g per serving."""
    suggestions = []
    sodium = _get(nd, "sodium", "sodium_mg", "sodium_g")
    if sodium is None:
        return suggestions

    sodium_g = sodium / 1000 if sodium > 10 else sodium
    salt_g   = sodium_g * 2.54
    bench_comparisons = bench_report.get("comparisons", {})
    bench_sodium      = bench_comparisons.get("sodium", {})
    worse_than_avg    = bench_sodium and not bench_sodium.get("better_than_avg", True)

    if salt_g > 3.0:
        detail = (
            f"Current salt equivalent is {salt_g:.2f} g/serving. "
            f"MHLW's low-salt (減塩) claim threshold requires a 25% reduction "
            f"vs. the standard product and ≤120% of category average. "
            + (f"Also behind category average on sodium." if worse_than_avg else "")
        )
        suggestions.append(_make_suggestion(
            title    = "Reduce Sodium to Qualify for Low-Salt (減塩) Claim",
            detail   = detail,
            fallback_impact = (
                "Reduced-sodium variants command a 10–20% price premium in Japanese "
                "retail and attract health-conscious consumers aged 40+, a key "
                "demographic for premium food spending."
            ),
            fallback_how = (
                "Partially substitute NaCl with potassium chloride (塩化カリウム, up to 30%), "
                "enhance umami depth with kombu extract or MSG to maintain taste perception, "
                "and retest with a consumer panel before launch."
            ),
            priority = "high" if salt_g > 4.5 else "medium",
            category = "Reformulation",
            nd       = nd,
            bench_comparisons = bench_comparisons,
        ))

    return suggestions


def _fat_suggestions(nd: dict, bench_report: dict) -> List[Suggestion]:
    """Suggest fat reductions when fat or saturated fat make up too much of the energy."""
    suggestions = []
    fat      = _get(nd, "fat", "total_fat", "fat_g")
    sat_fat  = _get(nd, "saturated_fat", "saturated_fat_g")
    calories = _get(nd, "calories", "energy_kcal")

    if fat is None or calories is None or calories == 0:
        return suggestions

    bench_comparisons = bench_report.get("comparisons", {})
    fat_pct = (fat * 9 / calories) * 100

    if fat_pct > 35:
        worse  = bench_comparisons.get("fat", {})
        worse  = worse and not worse.get("better_than_avg", True)
        detail = (
            f"Fat provides {fat_pct:.0f}% of energy (CAA guideline: <30%). "
            f"High-fat positioning limits health claim eligibility."
            + (" Product is also above category average for fat." if worse else "")
        )
        suggestions.append(_make_suggestion(
            title    = "Reduce Total Fat Content",
            detail   = detail,
            fallback_impact = (
                "Lower fat enables 'light' or 低脂肪 claims, broadening retailer "
                "listing opportunities and appealing to diet-conscious shoppers."
            ),
            fallback_how = (
                "Review frying oil quantity and temperature. Consider baking or "
                "air-frying process variants. Substitute part of the fat with "
                "inulin or resistant starch to maintain mouthfeel."
            ),
            priority = "medium",
            category = "Reformulation",
            nd       = nd,
            bench_comparisons = bench_comparisons,
        ))

    if sat_fat is not None and calories > 0:
        sf_pct = (sat_fat * 9 / calories) * 100
        if sf_pct > 10:
            suggestions.append(_make_suggestion(
                title    = "Lower Saturated Fat for Heart-Healthy Positioning",
                detail   = (
                    f"Saturated fat is {sf_pct:.0f}% of energy. "
                    f"CAA heart-healthy claim threshold requires <7% of energy from sat-fat."
                ),
                fallback_impact = (
                    "A 'heart-healthy' (心臓に優しい) positioning unlocks premium shelf "
                    "placement in health food sections and hospital/pharmacy retail channels."
                ),
                fallback_how = (
                    "Replace palm oil or animal fat with high-oleic sunflower or rapeseed oil. "
                    "Target a 30% reduction in saturated fat per serving."
                ),
                priority = "medium",
                category = "Reformulation",
                nd       = nd,
                bench_comparisons = bench_comparisons,
            ))

    return suggestions


def _sugar_suggestions(nd: dict, bench_report: dict) -> List[Suggestion]:
    """Suggest sugar reduction when sugar is a large share of energy or above the category average."""
    suggestions = []
    sugar    = _get(nd, "sugar", "sugars_g", "added_sugars_g")
    calories = _get(nd, "calories", "energy_kcal")

    if sugar is None or calories is None or calories == 0:
        return suggestions

    bench_comparisons = bench_report.get("comparisons", {})
    sugar_pct   = (sugar * 4 / calories) * 100
    bench_sugar = bench_comparisons.get("sugar", {})
    worse       = bench_sugar and not bench_sugar.get("better_than_avg", True)

    if sugar_pct > 40:
        suggestions.append(_make_suggestion(
            title    = "Reduce Added Sugars to Enable 低糖 (Low-Sugar) Claim",
            detail   = (
                f"Sugars account for {sugar_pct:.0f}% of energy. "
                f"CAA 低糖 claim requires ≤2.5 g sugar per 100 kcal."
                + (" Currently behind category competitors on sugar." if worse else "")
            ),
            fallback_impact = (
                "Low-sugar variants are the fastest-growing segment in Japanese "
                "functional food retail. Commands 15%+ price premium."
            ),
            fallback_how = (
                "Partially substitute sucrose with erythritol (エリスリトール) or trehalose "
                "(トレハロース) — both approved under Japanese food additive regulations. "
                "Reduce by 20% per iteration with consumer taste panel validation."
            ),
            priority = "high" if sugar_pct > 60 else "medium",
            category = "Reformulation",
            nd       = nd,
            bench_comparisons = bench_comparisons,
        ))
    elif worse:
        suggestions.append(_make_suggestion(
            title    = "Benchmark Sugar Against Category Leaders",
            detail   = (
                f"Sugar content is above category average "
                f"({bench_sugar.get('our_value')} vs avg "
                f"{bench_sugar.get('category_avg')} g)."
            ),
            fallback_impact = (
                "Competitive parity on sugar enables side-by-side comparison "
                "marketing and prevents loss of health-focused shoppers."
            ),
            fallback_how = (
                "Reduce sweetener quantity by 10% and compensate with natural "
                "flavour enhancement (e.g. vanilla, fruit concentrates)."
            ),
            priority = "low",
            category = "Reformulation",
            nd       = nd,
            bench_comparisons = bench_comparisons,
        ))

    return suggestions


def _protein_suggestions(nd: dict, bench_report: dict) -> List[Suggestion]:
    """Suggest raising protein when it is a low share of energy and below the category average."""
    suggestions = []
    protein  = _get(nd, "protein", "protein_g")
    calories = _get(nd, "calories", "energy_kcal")

    if protein is None or calories is None or calories == 0:
        return suggestions

    bench_comparisons = bench_report.get("comparisons", {})
    protein_pct   = (protein * 4 / calories) * 100
    bench_protein = bench_comparisons.get("protein", {})
    worse         = bench_protein and not bench_protein.get("better_than_avg", True)

    if protein_pct < 8 and worse:
        suggestions.append(_make_suggestion(
            title    = "Increase Protein to Meet High-Protein Positioning",
            detail   = (
                f"Protein is {protein_pct:.0f}% of energy and below category average "
                f"({bench_protein.get('our_value')} vs avg "
                f"{bench_protein.get('category_avg')} g). "
                f"JAS 高たんぱく claim requires ≥20% of energy from protein."
            ),
            fallback_impact = (
                "'High protein' (高たんぱく) is the #1 functional claim driving Japanese "
                "food innovation. Products with this label see 25–40% higher velocity "
                "in convenience store channels."
            ),
            fallback_how = (
                "Add isolated soy protein (大豆タンパク), whey, or pea protein. "
                "A 5–8 g protein increase per serving typically achieves claim threshold "
                "without significant taste impact."
            ),
            priority = "medium",
            category = "Reformulation",
            nd       = nd,
            bench_comparisons = bench_comparisons,
        ))

    return suggestions


def _fiber_suggestions(nd: dict, bench_report: dict) -> List[Suggestion]:
    """Suggest adding fibre when it is under 3 g per serving."""
    suggestions = []
    fiber = _get(nd, "fiber", "dietary_fibre", "fiber_g", "dietary_fiber_g")
    bench_comparisons = bench_report.get("comparisons", {})
    bench_fiber       = bench_comparisons.get("fiber", {})
    worse             = bench_fiber and not bench_fiber.get("better_than_avg", True)

    if fiber is not None and fiber < 3:
        suggestions.append(_make_suggestion(
            title    = "Add Dietary Fibre for 機能性表示食品 (Function Claim)",
            detail   = (
                f"Current fibre is {fiber:.1f} g/serving. "
                f"JAS 機能性表示食品 gut-health claim requires ≥3 g dietary fibre."
                + (" Also below category average." if worse else "")
            ),
            fallback_impact = (
                "機能性表示食品 status allows on-pack health claims reviewed by CAA, "
                "commanding a 20–30% retail price premium and preferential "
                "health food aisle placement."
            ),
            fallback_how = (
                "Add 2–4 g inulin (イヌリン), chicory root fibre, or resistant dextrin "
                "(難消化性デキストリン). These are tasteless, dissolve easily, and are "
                "fully approved under Japanese food additive regulations."
            ),
            priority = "medium" if fiber < 1.5 else "low",
            category = "Reformulation",
            nd       = nd,
            bench_comparisons = bench_comparisons,
        ))

    return suggestions


def _label_positioning_suggestions(nd: dict, bench_report: dict,
                                    compliance_reports: dict) -> List[Suggestion]:
    """Suggest marketing and export positioning opportunities based on benchmark and compliance results."""
    suggestions = []
    calories  = _get(nd, "calories", "energy_kcal")
    protein   = _get(nd, "protein",  "protein_g")
    bench_comparisons = bench_report.get("comparisons", {})
    strengths  = bench_report.get("strengths", [])
    percentile = bench_report.get("percentile", 50)

    if percentile >= 70:
        suggestions.append(_make_suggestion(
            title    = "Leverage 'Better Than Average' Positioning",
            detail   = (
                f"This product outperforms {percentile}% of category benchmarks. "
                f"Strengths: {', '.join(strengths) if strengths else 'multiple metrics'}."
            ),
            fallback_impact = (
                "'Best in category' shelf-talkers and comparison advertising "
                "increase trial rate by 12–18% (Nielsen Japan Shopper Study)."
            ),
            fallback_how = (
                "Work with a Japanese marketing agency to create a comparison chart "
                "infographic for point-of-sale materials and e-commerce listings "
                "(Amazon Japan, Rakuten)."
            ),
            priority = "low",
            category = "Positioning",
            nd       = nd,
            bench_comparisons = bench_comparisons,
        ))

    if calories is not None and calories < 100:
        suggestions.append(_make_suggestion(
            title    = "Apply for Low-Calorie (低カロリー) Claim",
            detail   = (
                f"At {calories:.0f} kcal/serving, this product meets CAA's "
                f"low-calorie threshold (<100 kcal per serving for solid foods)."
            ),
            fallback_impact = (
                "低カロリー on-pack claim increases purchase intent by 22% among "
                "female shoppers aged 20–40 in Japanese market research."
            ),
            fallback_how = (
                "Submit a 機能性表示食品 notification to CAA or use the standard "
                "低カロリー label under Food Labelling Standards Article 7. "
                "Confirm serving size definition matches regulatory criteria."
            ),
            priority = "medium",
            category = "Positioning",
            nd       = nd,
            bench_comparisons = bench_comparisons,
        ))

    compliance_errors = sum(
        1 for r in compliance_reports.values()
        for i in r.issues if i.severity == "error"
    )
    if compliance_errors == 0 and protein is not None and protein > 5:
        suggestions.append(_make_suggestion(
            title    = "Pursue Export Certification (Halal / Kosher)",
            detail   = (
                "With no critical compliance issues and adequate protein, this product "
                "has export potential to Muslim-majority and Jewish markets."
            ),
            fallback_impact = (
                "Japanese food exports to Southeast Asia grew 18% YoY (JETRO 2023). "
                "Halal certification opens a market of 240M+ consumers in Indonesia alone."
            ),
            fallback_how = (
                "Apply through Japan Halal Association (日本ハラール協会) or a "
                "JAKIM-recognised certifier. Removing pork-derived gelatin is typically "
                "the main ingredient substitution required."
            ),
            priority = "low",
            category = "Positioning",
            nd       = nd,
            bench_comparisons = bench_comparisons,
        ))

    return suggestions


def generate_suggestions(
    nutrition_dict: dict,
    bench_report: dict,
    compliance_reports: dict,
) -> List[Suggestion]:
    """Return improvement suggestions sorted by priority.
    All are built with fallback text first, then enriched together in one batched Groq call."""
    bench_ok = "error" not in bench_report

    raw_suggestions: List[Suggestion] = []
    raw_suggestions += _compliance_suggestions(compliance_reports, nutrition_dict)
    raw_suggestions += _sodium_suggestions(nutrition_dict, bench_report if bench_ok else {})
    raw_suggestions += _fat_suggestions(nutrition_dict, bench_report if bench_ok else {})
    raw_suggestions += _sugar_suggestions(nutrition_dict, bench_report if bench_ok else {})
    raw_suggestions += _protein_suggestions(nutrition_dict, bench_report if bench_ok else {})
    raw_suggestions += _fiber_suggestions(nutrition_dict, bench_report if bench_ok else {})
    raw_suggestions += _label_positioning_suggestions(
        nutrition_dict, bench_report if bench_ok else {}, compliance_reports
    )

    if not raw_suggestions:
        return []

    batch_input = [
        {"idx": i + 1, "title": s.title, "detail": s.detail, "category": s.category}
        for i, s in enumerate(raw_suggestions)
    ]
    enrichments = _ai_enrich_batch(batch_input, nutrition_dict)

    enriched: List[Suggestion] = []
    for i, s in enumerate(raw_suggestions):
        entry = enrichments.get(str(i + 1), {})
        enriched.append(Suggestion(
            title    = s.title,
            detail   = s.detail,
            impact   = entry.get("impact") or s.impact,
            how      = entry.get("how")    or s.how,
            priority = s.priority,
            category = s.category,
        ))

    priority_order = {"high": 0, "medium": 1, "low": 2}
    enriched.sort(key=lambda s: priority_order.get(s.priority, 3))
    return enriched
