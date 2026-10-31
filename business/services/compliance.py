"""JAS and Food Labelling Standards (食品表示基準) compliance checks on extracted label data.
Based on the Food Labelling Act (2015), JAS Act and MHLW/CAA reference values."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ComplianceIssue:
    """A single finding with a severity, code and message."""
    severity: str   # "error", "warning", "info"
    code: str       # e.g. "JAS-001"
    message: str


@dataclass
class ComplianceReport:
    """Result of one compliance check: score, summary, issues and passed checks."""
    standard: str
    passed: bool
    score: int              # 0-100, or None if not applicable
    summary: str
    issues: List[ComplianceIssue] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    not_applicable: bool = False   # True when the standard doesn't apply to this product


# Helpers

def _get(d: dict, *keys, default=None):
    """Safely retrieve a numeric value from nutrition_dict."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


def _score(issues: List[ComplianceIssue]) -> int:
    """Score out of 100: -25 per error, -10 per warning."""
    n_err  = sum(1 for i in issues if i.severity == "error")
    n_warn = sum(1 for i in issues if i.severity == "warning")
    return max(0, 100 - n_err * 25 - n_warn * 10)


def _get_salt_g(nd: dict):
    """Return salt equivalent in grams; "sodium" is already stored as 食塩相当量 in g."""
    val = _get(nd, "sodium", "sodium_g")
    if val is not None:
        # >20 is assumed to be mg
        if val > 20:
            return val / 1000
        return val
    return None


def _has_serving_size(nd: dict) -> bool:
    """True if a serving size was captured as a number or a non-empty string."""
    if _get(nd, "serving_size", "serving_size_g") is not None:
        return True
    for key in ("serving_size", "serving_size_str", "serving"):
        v = nd.get(key)
        if v is not None and str(v).strip() not in ("", "None", "null"):
            return True
    return False


# Organic claim detection (JAS Organic is voluntary, so only run when the label claims it)

_ORGANIC_KEYWORDS = [
    "有機",
    "オーガニック",
    "有機jas",
    "有機jasマーク",
    "有機jасマーク",   # cyrillic-a variant (OCR artefact)
    "organic",
    "jas organic",
]

# Fields that might carry product-level claims or the full label text
_CLAIM_FIELDS = [
    "ingredients", "label_text", "product_name", "product_type",
    "claims", "certifications", "category", "description",
    "raw_text", "full_text",
]

def _claims_organic(nd: dict) -> bool:
    """True only if the label data contains an explicit organic claim."""
    combined = " ".join(
        str(nd.get(f, "") or "").lower()
        for f in _CLAIM_FIELDS
    )
    return any(kw in combined for kw in _ORGANIC_KEYWORDS)


# Food Labelling Standards

_DRV = {
    "energy_kcal":     2200,
    "protein_g":       81,
    "fat_g":           73,
    "saturated_fat_g": 16,
    "carbs_g":         320,
    "dietary_fibre_g": 21,
    "sodium_mg":       2000,
    "salt_g":          7.5,
}


def _check_food_labelling_standards(nd: dict) -> ComplianceReport:
    """Check the mandatory "Big 5" nutrient declarations under the Food Labelling Standards."""
    issues: List[ComplianceIssue] = []
    passes: List[str] = []

    energy  = _get(nd, "calories",      "energy_kcal")
    protein = _get(nd, "protein",       "protein_g")
    fat     = _get(nd, "fat",           "total_fat",  "fat_g")
    carbs   = _get(nd, "carbohydrates", "carbs_g",    "total_carbohydrate_g")
    sodium  = _get(nd, "sodium",        "sodium_g",   "sodium_mg")

    big5 = {
        "JAS-001": ("Energy (熱量 kcal)",               energy),
        "JAS-002": ("Protein (たんぱく質)",              protein),
        "JAS-003": ("Fat (脂質)",                        fat),
        "JAS-004": ("Carbohydrate (炭水化物)",           carbs),
        "JAS-005": ("Sodium / Salt equiv. (食塩相当量)", sodium),
    }
    for code, (name, val) in big5.items():
        if val is None:
            issues.append(ComplianceIssue("error", code,
                f"'{name}' is mandatory under Japan's Food Labelling Standards "
                f"(食品表示基準 Article 3) but was not found on the label."))
        else:
            passes.append(f"{name} declared ({val:.2f}).")

    serving_known = _has_serving_size(nd)
    if not serving_known:
        issues.append(ComplianceIssue("warning", "JAS-006",
            "Serving size (1食分量) could not be confirmed from the label. "
            "Japan's Food Labelling Standards require the nutrition table to clearly state "
            "the reference amount (e.g. 1本(45g)当たり or 100g当たり). "
            "Per-serving checks below are based on raw extracted values — "
            "if the label is per-serving (not per-100g), those values may be understated."))
    else:
        sv = _get(nd, "serving_size", "serving_size_g")
        passes.append(f"Serving size declared ({sv:.1f} g)." if sv else "Serving size present.")

    salt_g = _get_salt_g(nd)
    if salt_g is not None:
        if not serving_known:
            # No confirmed serving size, so the 3g per-serving limit does not apply;
            # use a per-100g proxy (>1.5g salt equivalent is high).
            if salt_g > 1.5:
                issues.append(ComplianceIssue("info", "JAS-007",
                    f"Salt equivalent ({salt_g:.2f} g per extracted unit) appears elevated. "
                    f"Serving size unknown — cannot confirm per-serving amount. "
                    f"Verify the label's reference unit before assessing sodium compliance."))
            else:
                passes.append(f"Salt equivalent appears within range ({salt_g:.2f} g); serving size unconfirmed.")
        elif salt_g > 3.0:
            issues.append(ComplianceIssue("warning", "JAS-007",
                f"Salt equivalent ({salt_g:.2f} g/serving) is high. "
                f"MHLW target is <6 g/day total; single servings above 3 g "
                f"are flagged as high-salt (高塩分)."))
        else:
            passes.append(f"Salt equivalent within acceptable range ({salt_g:.2f} g/serving).")

    if energy is not None and protein is not None and fat is not None and carbs is not None:
        calc = protein * 4 + fat * 9 + carbs * 4
        if calc > 0:
            diff_pct = abs(energy - calc) / calc * 100
            if diff_pct > 25:
                issues.append(ComplianceIssue("warning", "JAS-008",
                    f"Declared energy ({energy:.0f} kcal) differs by {diff_pct:.0f}% "
                    f"from macro-calculated value ({calc:.0f} kcal). "
                    f"JAS requires accurate nutrient calculation methods (dietary fibre "
                    f"and sugar alcohols may account for small differences)."))
            else:
                passes.append(f"Declared energy consistent with macronutrient calculation "
                              f"({energy:.0f} kcal declared vs {calc:.0f} kcal calculated).")

    passed = not any(i.severity == "error" for i in issues)
    n_warn = sum(1 for i in issues if i.severity == "warning")

    if passed and n_warn == 0:
        summary = "Fully compliant with Japan Food Labelling Standards (食品表示基準)."
    elif passed:
        summary = f"Mandatory fields present; {n_warn} warning(s) to address."
    else:
        n_err = sum(1 for i in issues if i.severity == "error")
        summary = (f"{n_err} mandatory nutrient declaration(s) missing under "
                   f"Japan Food Labelling Standards.")

    return ComplianceReport(
        standard="Japan Food Labelling Standards (食品表示基準)",
        passed=passed,
        score=_score(issues),
        summary=summary,
        issues=issues,
        passed_checks=passes,
    )


# Nutrient function claims

def _check_jas_nutrient_function(nd: dict) -> ComplianceReport:
    """Check nutrient ratios against CAA limits for 栄養機能食品 function claims."""
    issues: List[ComplianceIssue] = []
    passes: List[str] = []

    serving_known = _has_serving_size(nd)
    serving_caveat = (
        " (Note: serving size unconfirmed — value may be per-serving or per-100g; "
        "verify before using for compliance decisions.)"
        if not serving_known else ""
    )

    energy  = _get(nd, "calories",      "energy_kcal")
    fat     = _get(nd, "fat",           "total_fat",       "fat_g")
    sat_fat = _get(nd, "saturated_fat", "sat_fat",         "saturated_fat_g")
    sugar   = _get(nd, "sugar",         "sugars_g",        "added_sugars_g")
    fibre   = _get(nd, "fiber",         "dietary_fibre",   "fiber_g", "dietary_fiber_g")
    protein = _get(nd, "protein",       "protein_g")

    # Fat: <30% of energy; only a warning when a low-fat claim is made
    raw_text = str(nd.get("raw_text", "") or "")
    has_fat_claim = any(kw in raw_text for kw in [
        "低脂肪", "無脂肪", "ノンファット", "low fat", "fat free",
        "脂肪ゼロ", "脂質ゼロ",
    ])
    if fat is not None and energy is not None and energy > 0:
        fat_pct = (fat * 9 / energy) * 100
        if fat_pct > 30 and has_fat_claim:
            issues.append(ComplianceIssue("warning", "JAS-NF-001",
                f"Fat provides {fat_pct:.1f}% of energy. CAA guideline is <30%; "
                f"this product cannot carry low-fat function claims."))
        elif fat_pct > 30:
            passes.append(
                f"Fat energy ratio: {fat_pct:.1f}% of energy "
                f"(above 30% threshold — only relevant if making low-fat claims)."
            )
        else:
            passes.append(f"Fat energy ratio acceptable ({fat_pct:.1f}% of energy).")

    # Saturated fat: <7% of energy
    if sat_fat is not None and energy is not None and energy > 0:
        sf_pct = (sat_fat * 9 / energy) * 100
        if sf_pct > 7:
            issues.append(ComplianceIssue("info", "JAS-NF-002",
                f"Saturated fat provides {sf_pct:.1f}% of energy. "
                f"CAA limit for saturated-fat function claims is <7% of energy."))
        else:
            passes.append(f"Saturated fat ratio compliant ({sf_pct:.1f}% of energy).")

    # Sugars: >10% of energy (ratio-based, valid without serving size)
    if sugar is not None and energy is not None and energy > 0:
        sug_pct = (sugar * 4 / energy) * 100
        if sug_pct > 10:
            issues.append(ComplianceIssue("info", "JAS-NF-003",
                f"Sugars contribute {sug_pct:.1f}% of energy. "
                f"CAA guidance recommends <10% from free sugars."))
        else:
            passes.append(f"Sugar energy ratio acceptable ({sug_pct:.1f}%).")

    has_fibre_claim = any(kw in raw_text for kw in [
        "食物繊維", "ファイバー", "fiber", "fibre", "食物せんい",
    ])
    if fibre is not None:
        if fibre >= 3:
            passes.append(f"Dietary fibre sufficient for function claim ({fibre:.1f} g).")
        elif has_fibre_claim:
            issues.append(ComplianceIssue("warning", "JAS-NF-004",
                f"Dietary fibre ({fibre:.1f} g) is low. JAS 栄養機能食品 fibre "
                f"function claim requires >=3 g per serving.{serving_caveat}"))
        else:
            passes.append(f"Dietary fibre: {fibre:.1f} g per serving.")
    elif has_fibre_claim:
        issues.append(ComplianceIssue("info", "JAS-NF-005",
            "Dietary fibre (食物繊維) not declared despite fibre-related claim. "
            "Declare the fibre content in the nutrition table."))

    has_protein_claim = any(kw in raw_text for kw in [
        "高たんぱく", "プロテイン", "protein", "たんぱく質豊富",
    ])
    if protein is not None:
        drv_pct = (protein / _DRV["protein_g"]) * 100
        if drv_pct >= 15:
            passes.append(f"Protein provides {drv_pct:.1f}% of Japanese DRV.")
        elif has_protein_claim:
            issues.append(ComplianceIssue("warning", "JAS-NF-006",
                f"Protein ({protein:.1f} g) provides only {drv_pct:.1f}% of DRV; "
                f"a protein function claim requires >=15% DRV per serving.{serving_caveat}"))
        else:
            passes.append(f"Protein: {protein:.1f} g per serving ({drv_pct:.1f}% DRV).")

    passed = not any(i.severity == "error" for i in issues)
    n_warn = sum(1 for i in issues if i.severity == "warning")

    if passed and n_warn == 0:
        summary = "Meets CAA nutrient function food (栄養機能食品) guidelines."
    elif passed:
        summary = f"Mostly compliant with CAA nutrient guidelines; {n_warn} area(s) of concern."
    else:
        summary = "Does not fully meet CAA nutrient function food criteria."

    return ComplianceReport(
        standard="JAS Nutrient Function Claims (栄養機能食品)",
        passed=passed,
        score=_score(issues),
        summary=summary,
        issues=issues,
        passed_checks=passes,
    )


# JAS Organic

def _check_jas_organic(nd: dict) -> ComplianceReport:
    """Screen label data for JAS Organic violations (synthetic additives, trans fat).
    Returns not_applicable when the product makes no organic claim."""
    if not _claims_organic(nd):
        return ComplianceReport(
            standard="JAS Organic (有機JASマーク)",
            passed=True,
            score=None,   # UI must render this as "N/A" not "100"
            summary=(
                "This product makes no 有機 (organic) claim — "
                "JAS Organic standard does not apply."
            ),
            issues=[],
            passed_checks=[
                "No 有機 / オーガニック claim detected on label. "
                "JAS Organic (有機JASマーク) certification is a voluntary standard "
                "that only applies to products displaying the mark or using the "
                "words 有機 / オーガニック. This check has been skipped."
            ],
            not_applicable=True,
        )

    issues: List[ComplianceIssue] = []
    passes: List[str] = []

    ingredients = str(nd.get("ingredients", "") or "").lower()

    # Additives prohibited under 有機JAS (MAFF Notification No. 1605); English and Japanese terms
    prohibited = [
        ("artificial colour",  "JAS-ORG-001", "artificial colour (合成着色料)"),
        ("artificial flavor",  "JAS-ORG-002", "artificial flavour (合成香料)"),
        ("artificial flavour", "JAS-ORG-002", "artificial flavour (合成香料)"),
        ("sodium benzoate",    "JAS-ORG-003", "sodium benzoate / 安息香酸Na"),
        ("potassium sorbate",  "JAS-ORG-004", "potassium sorbate / ソルビン酸K"),
        ("aspartame",          "JAS-ORG-005", "aspartame / アスパルテーム"),
        ("acesulfame",         "JAS-ORG-006", "acesulfame potassium / アセスルファムK"),
        ("high fructose",      "JAS-ORG-007", "high-fructose corn syrup (果糖ぶどう糖液糖)"),
        ("hydrogenated",       "JAS-ORG-008", "hydrogenated oil (水素添加油脂)"),
        ("合成着色料",          "JAS-ORG-001", "artificial colour (合成着色料)"),
        ("合成香料",            "JAS-ORG-002", "artificial flavour (合成香料)"),
        ("安息香酸",            "JAS-ORG-003", "sodium benzoate / 安息香酸Na"),
        ("ソルビン酸",          "JAS-ORG-004", "potassium sorbate / ソルビン酸K"),
        ("アスパルテーム",      "JAS-ORG-005", "aspartame / アスパルテーム"),
        ("アセスルファム",      "JAS-ORG-006", "acesulfame potassium / アセスルファムK"),
        ("果糖ぶどう糖液糖",    "JAS-ORG-007", "high-fructose corn syrup (果糖ぶどう糖液糖)"),
        ("水素添加",            "JAS-ORG-008", "hydrogenated oil (水素添加油脂)"),
    ]

    found_codes = set()
    for term, code, label in prohibited:
        if term in ingredients and code not in found_codes:
            issues.append(ComplianceIssue("error", code,
                f"'{label}' is prohibited under JAS Organic (有機JAS) standards "
                f"(MAFF Notification No. 1605)."))
            found_codes.add(code)

    if not found_codes and ingredients:
        passes.append("No prohibited synthetic additives detected in ingredient list.")
    elif not ingredients:
        issues.append(ComplianceIssue("info", "JAS-ORG-009",
            "Ingredient list unavailable; cannot perform full 有機JAS additive screening."))

    trans_fat = _get(nd, "trans_fat", "trans_fat_g")
    if trans_fat is not None and trans_fat > 0.3:
        issues.append(ComplianceIssue("error", "JAS-ORG-010",
            f"Trans fat ({trans_fat:.1f} g) detected. Hydrogenated oils producing "
            f"trans fats are prohibited under 有機JAS certification."))
    elif trans_fat is not None:
        passes.append(f"Trans fat at acceptable level ({trans_fat:.1f} g).")

    passes.append(
        "Note: Full 有機JAS certification requires MAFF-accredited body inspection; "
        "this is an indicative label-based check only."
    )

    passed = not any(i.severity == "error" for i in issues)
    n_err = sum(1 for i in issues if i.severity == "error")

    summary = (
        "No obvious 有機JAS violations detected from label data."
        if passed else
        f"{n_err} ingredient(s) incompatible with 有機JAS (JAS Organic) certification."
    )

    return ComplianceReport(
        standard="JAS Organic (有機JASマーク)",
        passed=passed,
        score=_score(issues),
        summary=summary,
        issues=issues,
        passed_checks=passes,
        not_applicable=False,
    )


# Allergens

_MANDATORY_ALLERGENS = {
    "Wheat", "Buckwheat", "Egg", "Milk", "Peanut", "Shrimp", "Crab", "Walnut"
}

_RECOMMENDED_ALLERGENS = {
    "Almond", "Abalone", "Squid", "Salmon roe", "Orange", "Cashew", "Kiwi",
    "Beef", "Sesame", "Salmon", "Mackerel", "Soy", "Chicken", "Banana",
    "Pork", "Matsutake mushroom", "Peach", "Yam", "Apple", "Gelatin", "Walnut",
}

_INGREDIENT_ALLERGEN_SIGNALS = {
    "Wheat":      ["小麦", "小麦粉", "小麦でん粉", "小麦たんぱく", "グルテン",
                   "醤油", "しょうゆ", "麸", "パン粉", "スパゲッティ"],
    "Buckwheat":  ["そば", "蕎麦", "そば粉"],
    "Egg":        ["卵", "たまご", "全卵", "卵黄", "卵白", "乾燥卵", "卵殻"],
    "Milk":       ["乳", "牛乳", "脱脂粉乳", "全粉乳", "バター", "チーズ",
                   "生クリーム", "練乳", "ホエイ", "カゼイン", "ヨーグルト",
                   "乳糖", "乳清"],
    "Peanut":     ["落花生", "ピーナッツ", "落花生油", "ピーナッツバター"],
    "Shrimp":     ["えび", "海老", "エビ"],
    "Crab":       ["かに", "蟹", "カニ"],
    "Walnut":     ["くるみ", "クルミ"],
    "Soy":        ["大豆", "大豆油", "大豆たんぱく", "大豆由来", "豆腐",
                   "味噌", "みそ", "納豆"],
    "Sesame":     ["ごま", "胡麻", "ごま油"],
    "Salmon":     ["さけ", "鮭", "サーモン"],
    "Mackerel":   ["さば", "鯖"],
    "Squid":      ["いか", "烏賊"],
    "Beef":       ["牛肉", "ビーフ", "牛エキス"],
    "Pork":       ["豚肉", "ポーク", "豚エキス", "ゼラチン"],
    "Chicken":    ["鶏肉", "チキン", "鶏エキス"],
    "Almond":     ["アーモンド"],
    "Cashew":     ["カシューナッツ"],
    "Banana":     ["バナナ"],
    "Apple":      ["りんご"],
    "Orange":     ["オレンジ"],
    "Peach":      ["もも"],
    "Kiwi":       ["キウイ"],
    "Gelatin":    ["ゼラチン"],
    "Abalone":    ["あわび"],
    "Salmon roe": ["いくら"],
    "Yam":        ["やまいも", "山芋"],
    "Matsutake mushroom": ["まつたけ", "松茸"],
}

_SIGNAL_EXCLUSIONS = {
    "Milk": ["乳化剤", "乳酸", "乳酸菌", "乳酸カルシウム", "乳酸ナトリウム",
             "乳糖アルコール", "離乳食", "未焼成Ca"],
    "Egg":  ["卵形", "卵巣"],
    "Soy":  [],
}


def _allergens_in_text(text: str) -> set:
    """Return English names of allergens whose signal terms appear in text.
    Terms in _SIGNAL_EXCLUSIONS are masked first (same logic as extractor.py)."""
    found = set()
    for allergen, signals in _INGREDIENT_ALLERGEN_SIGNALS.items():
        exclusions = _SIGNAL_EXCLUSIONS.get(allergen, [])
        temp = text
        for excl in exclusions:
            temp = temp.replace(excl, "░" * len(excl))
        for signal in signals:
            if signal in temp:
                found.add(allergen)
                break
    return found


def _check_allergen_labelling(nd: dict) -> ComplianceReport:
    """Cross-check declared allergens against those detected in the ingredient text."""
    issues: List[ComplianceIssue] = []
    passes: List[str] = []

    ingredients_text = str(nd.get("ingredients", "") or "")
    raw_text         = str(nd.get("raw_text",     "") or "")
    declared_list    = nd.get("allergens_in_product", []) or []

    declared = set(declared_list)

    if not ingredients_text and not raw_text:
        return ComplianceReport(
            standard="Allergen Labelling (アレルゲン表示)",
            passed=True,
            score=None,
            summary="No ingredient text available — allergen cross-check skipped.",
            issues=[],
            passed_checks=["Ingredient text not extracted; allergen validation could not run."],
            not_applicable=True,
        )

    detected_in_ingredients = _allergens_in_text(ingredients_text)

    mandatory_undeclared   = []
    recommended_undeclared = []

    for allergen in detected_in_ingredients:
        if allergen in declared:
            passes.append(
                f"{allergen} — detected in ingredients and correctly declared ✓"
            )
            continue

        if allergen in _MANDATORY_ALLERGENS:
            mandatory_undeclared.append(allergen)
        elif allergen in _RECOMMENDED_ALLERGENS:
            recommended_undeclared.append(allergen)


    # Undeclared mandatory allergens are errors (illegal to sell); recommended ones are info
    for allergen in sorted(mandatory_undeclared):
        issues.append(ComplianceIssue(
            severity="error",
            code=f"JAS-ALG-{allergen[:3].upper()}",
            message=(
                f"'{allergen}' is a mandatory allergen (特定原材料) detected in the "
                f"ingredient list but NOT found in the allergen declaration. "
                f"This is a legal violation under Japan's Food Labelling Standards "
                f"(食品表示基準 Article 3, Schedule 4). The product cannot be legally "
                f"sold in Japan without declaring this allergen."
            )
        ))

    for allergen in sorted(recommended_undeclared):
        issues.append(ComplianceIssue(
            severity="info",
            code=f"JAS-ALG-R{allergen[:3].upper()}",
            message=(
                f"'{allergen}' is a recommended allergen (特定原材料に準ずるもの) "
                f"detected in the ingredient list but not declared in the allergen "
                f"section. The CAA strongly recommends declaring this allergen "
                f"to protect consumers with sensitivities."
            )
        ))

    if not declared and detected_in_ingredients:
        issues.append(ComplianceIssue(
            severity="warning",
            code="JAS-ALG-SEC",
            message=(
                "No allergen declaration section (アレルギー物質含む) could be "
                "detected on this label. Japan's Food Labelling Standards require "
                "an explicit allergen statement for all products containing any of "
                "the 8 mandatory allergens."
            )
        ))
    elif not declared and not detected_in_ingredients:
        passes.append(
            "No allergens detected in ingredient list and no declaration section found. "
            "If this product genuinely contains no allergens, this is acceptable."
        )

    correctly_declared_mandatory = [
        a for a in detected_in_ingredients
        if a in _MANDATORY_ALLERGENS and a in declared
    ]
    if correctly_declared_mandatory and not mandatory_undeclared:
        passes.append(
            f"All detected mandatory allergens correctly declared: "
            f"{', '.join(sorted(correctly_declared_mandatory))}"
        )

    if not detected_in_ingredients:
        passes.append(
            "No allergen-containing ingredients detected in ingredient text."
        )

    passed  = not any(i.severity == "error" for i in issues)
    n_err   = sum(1 for i in issues if i.severity == "error")
    n_warn  = sum(1 for i in issues if i.severity == "warning")
    n_info  = sum(1 for i in issues if i.severity == "info")

    if n_err == 0 and n_warn == 0:
        summary = (
            f"Allergen labelling compliant — "
            f"{len(correctly_declared_mandatory)} mandatory allergen(s) correctly declared."
            if correctly_declared_mandatory else
            "No allergen issues detected."
        )
    elif n_err > 0:
        summary = (
            f"{n_err} mandatory allergen(s) present in ingredients but NOT declared — "
            f"illegal to sell in Japan without correction."
        )
    else:
        summary = (
            f"{n_warn} allergen labelling warning(s); "
            f"{n_info} recommended allergen(s) undeclared."
        )

    return ComplianceReport(
        standard="Allergen Labelling (アレルゲン表示)",
        passed=passed,
        score=_score(issues),
        summary=summary,
        issues=issues,
        passed_checks=passes,
        not_applicable=False,
    )


# Mandatory label fields (食品表示基準 Article 3; importers carry full legal responsibility)

# (code, severity, japanese_markers, english_name, law_ref, fix_hint)
_MANDATORY_FIELDS = [
    (
        "JAS-MND-001", "error",
        ["名称", "品名"],
        "Legal product category name (名称)",
        "食品表示基準 Article 3, Item 1",
        "Add 名称: followed by the legal category name (e.g. 名称: スナック菒子). "
        "This is distinct from the brand/product name."
    ),
    (
        "JAS-MND-002", "error",
        ["内容量"],
        "Net weight / content quantity (内容量)",
        "食品表示基準 Article 3, Item 3",
        "Add 内容量: followed by the net weight in metric units (e.g. 内容量: 58g). "
        "Metric units are mandatory."
    ),
    (
        "JAS-MND-003", "error",
        ["賞味期限", "消費期限"],
        "Best-before or use-by date (賞味期限 / 消費期限)",
        "食品表示基準 Article 3, Item 5",
        "Add 賞味期限 (best-before) or 消費期限 (use-by). "
        "Required even on packages with <30cm2 labelling surface."
    ),
    (
        "JAS-MND-004", "error",
        ["製造者", "販売者", "輸入者", "製造元", "販売元", "発売元"],
        "Manufacturer / distributor / importer name & address (製造者/販売者)",
        "食品表示基準 Article 3, Item 7",
        "Add 製造者: (or 販売者: / 輸入者: for imports) followed by full company name and address."
    ),
    (
        "JAS-MND-005", "warning",
        ["保存方法"],
        "Storage / preservation method (保存方法)",
        "食品表示基準 Article 3, Item 6",
        "Add 保存方法: followed by storage instructions."
    ),
    (
        "JAS-MND-006", "warning",
        ["原材料名", "原料名"],
        "Ingredients list (原材料名)",
        "食品表示基準 Article 3, Item 2",
        "Add 原材料名: followed by all ingredients in descending order by weight. "
        "Additives must be listed separately after a slash (/) from main ingredients."
    ),
]

# Ingredient categories needing origin labelling (22 food groups since March 2022)
_ORIGIN_REQUIRED_INGREDIENTS = [
    "牛肉", "豚肉", "鶏肉", "羊肉", "馬肉", "鴨肉",
    "髦", "さけ", "まぐろ", "かつお", "さば", "いわし", "たら", "ぶり",
    "えび", "海老", "かに", "蟹", "ほたて", "いか", "たこ",
    "米", "小麦", "大豆", "とうもろこし", "じゃがいも",
    "にんじん", "たまねぎ", "キャベツ", "ほうれん草",
    "りんご", "みかん", "もも", "ぶどう", "バナナ", "いちご",
]

_ORIGIN_DECLARATION_KEYWORDS = [
    "原産地", "原材料原産地", "原産国", "国産", "輸入",
    "産", "県産", "都産", "道産", "府産",
    "made in", "産地",
]

# Additive function category terms that must be separated by /
_ADDITIVE_FUNCTION_TERMS = [
    "乳化劑", "増粘劑", "着色料", "保存料", "酸化防止劑",
    "甘味料", "香料", "膨張劑", "調味料", "酸味料",
    "ph調整劑", "澂白劑", "発色劑", "防カビ劑", "光沢劑",
    "強化劑", "消泡劑", "製造用劑", "苦味料", "軟化劑",
    "アミノ酸", "ビタミンc", "ビタミンe",
]

_SLASH_VARIANTS = ["/", "／", "・", "、/", "/、"]


# OCR plausibility: Atwater factors (kcal/g) and physically impossible values
_ATWATER = {"protein": 4.0, "fat": 9.0, "carbohydrates": 4.0}

_IMPOSSIBLE_THRESHOLDS = [
    ("sodium",        10.0,   "g",    "Sodium > 10g/serving is physically impossible for any real food — OCR likely misread mg as g. Expected range: 0.001–5g"),
    ("calories",    2000.0,   "kcal", "Calories > 2000 kcal/serving exceeds a full day's intake — likely an OCR extraction error"),
    ("fat",          200.0,   "g",    "Fat > 200g/serving is physically impossible — likely an OCR extraction error"),
    ("protein",      200.0,   "g",    "Protein > 200g/serving is physically impossible — likely an OCR extraction error"),
    ("carbohydrates",500.0,   "g",    "Carbohydrates > 500g/serving is physically impossible — likely an OCR extraction error"),
]


def _check_ingredients_compliance(nd: dict) -> ComplianceReport:
    """Scan ingredient text for aspartame/PKU, GMO declaration and restricted-additive problems."""
    issues: List[ComplianceIssue] = []
    passes: List[str] = []

    ingredients = str(nd.get("ingredients", "") or "")
    raw_text    = str(nd.get("raw_text",    "") or "")

    full_text    = (ingredients + " " + raw_text).lower()
    ing_lower    = ingredients.lower()

    if not ingredients and not raw_text:
        return ComplianceReport(
            standard="Ingredients Compliance (成分・添加物規制)",
            passed=True,
            score=None,
            summary="No ingredient text available — ingredients compliance check skipped.",
            issues=[],
            passed_checks=["Ingredient text not extracted; compliance scan could not run."],
            not_applicable=True,
        )

    aspartame_found = any(t in full_text for t in _ASPARTAME_TERMS)

    if aspartame_found:
        pku_warning_found = any(t in full_text for t in _PKU_WARNING_TERMS)
        if pku_warning_found:
            passes.append(
                "Aspartame detected and mandatory L-フェニルアラニン (PKU) warning present ✓"
            )
        else:
            issues.append(ComplianceIssue(
                severity="error",
                code="JAS-ING-001",
                message=(
                    "アスパルテーム (aspartame) detected in ingredients but the mandatory "
                    "PKU warning 'L-フェニルアラニン化合物を含む' is absent. "
                    "This warning is legally required under 食品表示基準 Schedule 4, "
                    "Additive Column 3 to protect individuals with phenylketonuria (PKU). "
                    "Fix: Add 'L-フェニルアラニン化合物を含む' immediately after アスパルテーム "
                    "in the ingredient/additive list — e.g. アスパルテーム・L-フェニルアラニン化合物."
                )
            ))
    else:
        passes.append("No aspartame detected — PKU warning not required ✓")

    gmo_declared = any(t in full_text for t in _GMO_DECLARATION_TERMS)

    # Ignore exempt derived ingredients and crops that only appear in the
    # allergen bracket (一部に〜を含む), which is a cross-contact note
    import re as _re
    ing_for_gmo = ing_lower

    for exempt in _GMO_EXEMPT_DERIVED:
        ing_for_gmo = ing_for_gmo.replace(exempt, "▓" * len(exempt))

    ing_for_gmo = _re.sub(r'[（(]一部に[^）)]*[）)]', '', ing_for_gmo)

    triggering_crops = [
        eng for jp, eng in _GMO_REGULATED_CROPS
        if jp in ing_for_gmo
    ]

    if triggering_crops:
        if gmo_declared:
            passes.append(
                f"GMO-regulated crop(s) detected ({', '.join(triggering_crops[:3])}) "
                f"and GMO/IP declaration present ✓"
            )
        else:
            issues.append(ComplianceIssue(
                severity="warning",
                code="JAS-ING-002",
                message=(
                    f"GMO-regulated crop(s) detected as ingredients: "
                    f"{', '.join(triggering_crops)}. "
                    f"No 遺伝子組換え (GMO) or 遺伝子組換えでない (non-GMO) declaration found. "
                    f"Since April 2023, Japan's revised GMO labelling rules require an explicit "
                    f"declaration for regulated crops used as primary ingredients "
                    f"(消費者庁 GMO labelling reform, 食品表示基準 Article 7). "
                    f"Fix: Add '遺伝子組換えでない' after each non-GMO ingredient name, "
                    f"or '遺伝子組換え不分別' if segregation is not guaranteed."
                )
            ))
    else:
        passes.append(
            "No GMO-regulated primary crops detected — GMO declaration not required ✓"
        )

    any_additive_issue = False
    for terms, code, severity, message in _RESTRICTED_ADDITIVES:
        if any(t in full_text for t in terms):
            issues.append(ComplianceIssue(
                severity=severity,
                code=code,
                message=message,
            ))
            any_additive_issue = True

    if not any_additive_issue:
        passes.append(
            "No prohibited or improperly labelled additives detected ✓"
        )

    passed = not any(i.severity == "error" for i in issues)
    n_err  = sum(1 for i in issues if i.severity == "error")
    n_warn = sum(1 for i in issues if i.severity == "warning")

    if n_err == 0 and n_warn == 0:
        summary = "Ingredient and additive compliance — no issues detected."
    elif n_err > 0:
        summary = (
            f"{n_err} ingredient labelling violation(s) require immediate correction "
            f"before the product can be sold in Japan."
        )
    else:
        summary = (
            f"{n_warn} ingredient labelling warning(s) — review recommended "
            f"before market entry."
        )

    return ComplianceReport(
        standard="Ingredients Compliance (成分・添加物規制)",
        passed=passed,
        score=_score(issues),
        summary=summary,
        issues=issues,
        passed_checks=passes,
        not_applicable=False,
    )

def _check_ocr_plausibility(nd: dict) -> ComplianceReport:
    """Check nutrition values for impossible amounts and Atwater calorie mismatches (likely OCR errors)."""
    issues: List[ComplianceIssue] = []
    passes: List[str] = []

    cal   = _get(nd, "calories")
    prot  = _get(nd, "protein")
    fat   = _get(nd, "fat")
    carbs = _get(nd, "carbohydrates")

    if cal is None and prot is None and fat is None and carbs is None:
        return ComplianceReport(
            standard="Nutrition Data Plausibility (データ整合性)",
            passed=True,
            score=None,
            summary="No nutrition values extracted — plausibility check skipped.",
            issues=[],
            passed_checks=["No nutrition values available to validate."],
            not_applicable=True,
        )

    any_impossible = False
    for key, threshold, unit, explanation in _IMPOSSIBLE_THRESHOLDS:
        val = _get(nd, key)
        if val is not None and val > threshold:
            any_impossible = True
            issues.append(ComplianceIssue(
                severity="warning",
                code="JAS-PLB-001",
                message=(
                    f"Implausible value: {key} = {val}{unit}. {explanation}. "
                    f"Please re-check the raw OCR output and correct this value "
                    f"before relying on compliance scores."
                )
            ))
        elif val is not None:
            passes.append(f"{key} = {val} — within plausible range \u2713")

    # Atwater cross-check: needs all macros and calories, none flagged impossible
    if cal is not None and prot is not None and fat is not None and carbs is not None and not any_impossible:
        calculated = (prot * 4.0) + (fat * 9.0) + (carbs * 4.0)
        diff_pct   = abs(cal - calculated) / calculated * 100 if calculated > 0 else 0

        if diff_pct <= 20:
            passes.append(
                f"Calorie cross-check: declared {cal:.0f} kcal vs "
                f"Atwater calculated {calculated:.0f} kcal "
                f"(\u00b1{diff_pct:.1f}% — within acceptable range) \u2713"
            )
        elif diff_pct <= 35:
            issues.append(ComplianceIssue(
                severity="info",
                code="JAS-PLB-002",
                message=(
                    f"Calorie cross-check: declared {cal:.0f} kcal but Atwater "
                    f"calculation gives {calculated:.0f} kcal ({diff_pct:.1f}% difference). "
                    f"Differences up to ~20% are normal (fibre, sugar alcohols, rounding). "
                    f"A {diff_pct:.1f}% gap may indicate an OCR error — please verify "
                    f"the raw nutrition table values."
                )
            ))
        else:
            issues.append(ComplianceIssue(
                severity="warning",
                code="JAS-PLB-002",
                message=(
                    f"Calorie mismatch: declared {cal:.0f} kcal but Atwater "
                    f"calculation gives {calculated:.0f} kcal ({diff_pct:.1f}% difference). "
                    f"This gap is too large to be explained by fibre or rounding. "
                    f"Likely cause: OCR misread one or more values. "
                    f"Fix: Check raw OCR output and correct protein/fat/carbohydrate values."
                )
            ))
    elif cal is not None and (prot is None or fat is None or carbs is None):
        passes.append(
            "Calorie cross-check skipped — not all macros available for Atwater calculation"
        )

    passed  = not any(i.severity == "warning" for i in issues)
    n_warn  = sum(1 for i in issues if i.severity == "warning")
    n_info  = sum(1 for i in issues if i.severity == "info")

    if n_warn == 0 and n_info == 0:
        summary = "All nutrition values are internally consistent and plausible."
    elif n_warn > 0:
        summary = (
            f"{n_warn} implausible value(s) detected — likely OCR errors. "
            f"Compliance scores may be unreliable until corrected."
        )
    else:
        summary = f"Minor calorie discrepancy detected — worth verifying raw OCR values."

    return ComplianceReport(
        standard="Nutrition Data Plausibility (データ整合性)",
        passed=passed,
        score=_score(issues),
        summary=summary,
        issues=issues,
        passed_checks=passes,
        not_applicable=False,
    )


# Nutrient content claim thresholds (CAA Food Labelling Standards Article 7)
def _check_nutrient_claims(nd: dict) -> ComplianceReport:
    """List which CAA nutrient content claims the product can and cannot make (Article 7)."""
    issues: List[ComplianceIssue] = []
    passes: List[str] = []

    cal   = _get(nd, "calories")
    fat   = _get(nd, "fat")
    sodium = _get(nd, "sodium")   # in grams (食塩相当量)
    sugar = _get(nd, "sugar")
    prot  = _get(nd, "protein")
    fiber = _get(nd, "fiber")
    carbs = _get(nd, "carbohydrates")

    # 食塩相当量 (g NaCl) -> sodium (mg Na): salt = Na x 2.54
    sodium_mg = (sodium / 2.54 * 1000) if sodium is not None else None

    # Labels usually give per-serving values; per-100g needs the serving weight
    serving_note = "(per serving — per-100g checks require serving weight data)"

    eligible   = []
    ineligible = []
    unknown    = []

    if cal is not None:
        if cal < 5:
            eligible.append(("ノンカロリー", f"{cal:.0f} kcal/serving < 5 kcal threshold"))
        elif cal < 40:
            eligible.append(("低カロリー (solid food)", f"{cal:.0f} kcal/serving < 40 kcal threshold"))
        elif cal < 100:
            eligible.append(("低カロリー (may qualify)", f"{cal:.0f} kcal — verify per-100g is <100 kcal"))
        else:
            ineligible.append(("低カロリー / ノンカロリー", f"{cal:.0f} kcal/serving exceeds all calorie claim thresholds"))
    else:
        unknown.append("Calorie claims — calories not extracted")

    if fat is not None:
        if fat < 0.5:
            eligible.append(("無脂肪 (fat-free)", f"{fat:.1f}g fat/serving < 0.5g threshold"))
        elif fat < 3.0:
            eligible.append(("低脂肪 (low-fat)", f"{fat:.1f}g fat/serving < 3g threshold"))
        else:
            ineligible.append(("無脂肪 / 低脂肪", f"{fat:.1f}g fat/serving — need <0.5g for fat-free, <3g for low-fat"))
    else:
        unknown.append("Fat claims — fat not extracted")

    # Sodium thresholds are per 100g, so convert with serving_size_g when known
    serving_g = _get(nd, "serving_size", "serving_size_g")

    if sodium_mg is not None:
        if serving_g and serving_g > 0:
            sodium_mg_per_100g = sodium_mg / serving_g * 100

            if sodium_mg_per_100g < 5:
                eligible.append(("ナトリウムオフ",
                    f"{sodium_mg_per_100g:.0f}mg Na/100g < 5mg threshold"))
            if sodium_mg_per_100g < 120:
                eligible.append(("低ナトリウム (low sodium)",
                    f"{sodium_mg_per_100g:.0f}mg Na/100g < 120mg threshold ✓"))
            else:
                ineligible.append(("低ナトリウム / ナトリウムオフ",
                    f"{sodium_mg_per_100g:.0f}mg Na/100g — need <120mg/100g for low-sodium claim"))
        else:
            # No serving size: compare per serving as a proxy
            if sodium_mg < 120:
                eligible.append(("低ナトリウム (low sodium — verify per 100g)",
                    f"{sodium_mg:.0f}mg Na/serving < 120mg — confirm serving weight to verify"))
            else:
                ineligible.append(("低ナトリウム / ナトリウムオフ",
                    f"{sodium_mg:.0f}mg Na/serving — need <120mg/100g; serving weight unknown"))
        # 減塩 needs a reference product to compare against; only flag it when claimed
        raw_text_local = str(nd.get("raw_text", "") or "")
        if "減塩" in raw_text_local:
            issues.append(ComplianceIssue(
                severity="warning",
                code="JAS-CLM-004",
                message=(
                    f"減塩 (reduced salt) claim detected on label. "
                    f"This claim requires: (1) ≥25% less sodium than a reference product, "
                    f"AND (2) <120mg Na/100g. "
                    f"Verify reduction percentage against your standard product."
                )
            ))
    else:
        unknown.append("Sodium claims — sodium not extracted")

    if sugar is not None:
        if sugar < 0.5:
            eligible.append(("無糖 (sugar-free)", f"{sugar:.1f}g sugar/serving < 0.5g threshold"))
        elif sugar < 5.0:
            eligible.append(("低糖 (low sugar)", f"{sugar:.1f}g sugar/serving < 5g threshold"))
        else:
            ineligible.append(("無糖 / 低糖", f"{sugar:.1f}g sugar/serving — need <0.5g for sugar-free, <5g for low-sugar"))
    else:
        unknown.append("Sugar claims — sugar not extracted")

    if prot is not None and cal is not None and cal > 0:
        prot_pct_energy = (prot * 4.0 / cal) * 100
        if prot_pct_energy >= 20:
            eligible.append(("高たんぱく (high protein)", f"Protein is {prot_pct_energy:.1f}% of energy ≥ 20% threshold"))
        elif prot_pct_energy >= 10:
            eligible.append(("たんぱく質源 (protein source)", f"Protein is {prot_pct_energy:.1f}% of energy ≥ 10% threshold"))
        else:
            ineligible.append(("高たんぱく / たんぱく質源", f"Protein is only {prot_pct_energy:.1f}% of energy — need ≥10% for protein source, ≥20% for high-protein"))
    else:
        unknown.append("Protein claims — protein or calories not extracted")

    if fiber is not None:
        if fiber >= 3.0:
            eligible.append(("食物繊維高含有 (high fibre)", f"{fiber:.1f}g fibre/serving ≥ 3g threshold"))
        elif fiber >= 1.5:
            eligible.append(("食物繊維含有 (fibre source)", f"{fiber:.1f}g fibre/serving ≥ 1.5g threshold"))
        else:
            ineligible.append(("食物繊維高含有 / 食物繊維含有", f"{fiber:.1f}g fibre/serving — need ≥1.5g for fibre source, ≥3g for high-fibre"))
    else:
        unknown.append("Fibre claims — fibre not extracted")

    for claim, reason in eligible:
        passes.append(f"\u2713 CAN claim: {claim} — {reason}")

    for claim, reason in ineligible:
        issues.append(ComplianceIssue(
            severity="info",
            code="JAS-CLM-001",
            message=f"CANNOT claim: {claim} — {reason} ({serving_note})"
        ))

    for note in unknown:
        issues.append(ComplianceIssue(
            severity="info",
            code="JAS-CLM-002",
            message=f"UNKNOWN: {note} — add this value to assess claim eligibility"
        ))

    if not passes and not eligible:
        passes.append("No nutrient content claims currently available based on extracted values")

    n_eligible   = len(eligible)
    n_ineligible = len(ineligible)
    passed = True  # this check is informational — never blocks market readiness

    if n_eligible > 0:
        summary = (
            f"{n_eligible} nutrient claim(s) available to use on label. "
            f"{n_ineligible} claim(s) not currently achievable."
        )
    else:
        summary = (
            f"No nutrient content claims currently available. "
            f"{n_ineligible} claim threshold(s) not met."
        )

    return ComplianceReport(
        standard="Nutrient Claim Thresholds (栄養強調表示)",
        passed=passed,
        score=100 if n_eligible > 0 else 70,
        summary=summary,
        issues=issues,
        passed_checks=passes,
        not_applicable=False,
    )


# Functional food eligibility: 機能性表示食品 (CAA notification, 2015) and FOSHU (CAA pre-approval, 1991)
_FUNCTIONAL_CRITERIA = [
    {
        "category":    "Gut Health (腸内環境改善)",
        "claim_jp":    "お腹の調子を整える / 腸内環境を良好にする",
        "standard":    "機能性表示食品",
        "key":         "fiber",
        "threshold":   3.0,
        "unit":        "g",
        "direction":   ">=",
        "evidence":    "Dietary fibre ≥3g/serving supports gut microbiome health claims under CAA functional food guidelines.",
        "how":         "Submit a systematic review (SR) on dietary fibre and gut health to CAA. SR must cover ≥1 RCT with Japanese or similar population. Notification review period: 60 days.",
    },
    {
        "category":    "Protein Muscle Support (筋肉・体づくり)",
        "claim_jp":    "筋肉の維持に役立つ / たんぱく質を含む",
        "standard":    "機能性表示食品",
        "key":         "protein",
        "threshold":   10.0,
        "unit":        "g",
        "direction":   ">=",
        "evidence":    "≥10g protein/serving enables muscle maintenance claims. High-protein positioning is the #1 functional claim in Japanese convenience stores.",
        "how":         "File 機能性表示食品 notification with CAA. Required documents: SR on protein and muscle maintenance, safety data, label design. Cost: ¥500K–2M for SR preparation.",
    },
    {
        "category":    "Bone Health (骨の健康)",
        "claim_jp":    "骨の健康維持に役立つ",
        "standard":    "機能性表示食品 / FOSHU",
        "key":         "calcium",
        "threshold":   210.0,   # 210mg = 30% of Japanese DRV (700mg)
        "unit":        "mg",
        "direction":   ">=",
        "evidence":    "≥210mg calcium/serving (30% DRV) meets CAA bone health claim criteria when combined with vitamin D.",
        "how":         "Pair with ≥1.65μg vitamin D/serving for FOSHU bone health pathway. FOSHU approval: 12–24 months, ¥5–20M. 機能性表示食品 pathway faster but requires SR.",
    },
    {
        "category":    "Low Calorie Weight Management (体重管理)",
        "claim_jp":    "体脂肪が気になる方に / カロリーコントロールに役立つ",
        "standard":    "機能性表示食品",
        "key":         "calories",
        "threshold":   100.0,
        "unit":        "kcal",
        "direction":   "<=",
        "evidence":    "≤100 kcal/serving supports weight management positioning claims.",
        "how":         "Low-calorie products can use the 機能性表示食品 route with a SR on energy restriction and body weight. Also eligible for 低カロリー label claim under Article 7.",
    },
    {
        "category":    "Blood Pressure Support (血圧)",
        "claim_jp":    "血圧が高めの方に適した食品",
        "standard":    "FOSHU (特定保健用食品)",
        "key":         "sodium",
        "threshold":   0.6,     # <0.6g 食塩相当量 = <240mg sodium
        "unit":        "g",
        "direction":   "<=",
        "evidence":    "Low sodium (食塩相当量 <0.6g/serving) is a prerequisite for blood pressure FOSHU claims alongside a functional ingredient (GABA, peptides etc.).",
        "how":         "FOSHU blood pressure pathway requires a functional ingredient (e.g. 3mg GABA, casein dodecapeptide) + low sodium formulation + clinical trial. Budget ¥20–50M for full FOSHU.",
    },
]


def _check_functional_eligibility(nd: dict) -> ComplianceReport:
    """Report which 機能性表示食品 / FOSHU categories the nutrient profile could qualify for.
    Informational only; never a legal blocker."""
    issues: List[ComplianceIssue] = []
    passes: List[str] = []

    eligible_categories   = []
    ineligible_categories = []

    for crit in _FUNCTIONAL_CRITERIA:
        val = _get(nd, crit["key"])
        if val is None:
            issues.append(ComplianceIssue(
                severity="info",
                code="JAS-FNC-000",
                message=(
                    f"{crit['category']} — cannot assess: '{crit['key']}' not extracted. "
                    f"Add this value to determine eligibility."
                )
            ))
            continue

        threshold = crit["threshold"]
        direction = crit["direction"]
        met = (val >= threshold) if direction == ">=" else (val <= threshold)

        if met:
            eligible_categories.append(crit["category"])
            passes.append(
                f"\u2713 ELIGIBLE: {crit['category']} ({crit['standard']}) — "
                f"{crit['key']} = {val} {'≥' if direction == '>=' else '≤'} {threshold}{crit['unit']}. "
                f"Claim: \"{crit['claim_jp']}\". "
                f"Next step: {crit['how']}"
            )
        else:
            ineligible_categories.append(crit["category"])
            issues.append(ComplianceIssue(
                severity="info",
                code="JAS-FNC-001",
                message=(
                    f"NOT YET ELIGIBLE: {crit['category']} — "
                    f"current {crit['key']} = {val}, "
                    f"need {'≥' if direction == '>=' else '≤'} {threshold}{crit['unit']}. "
                    f"Gap: {abs(val - threshold):.2f}{crit['unit']}. "
                    f"Reformulation path: {crit['evidence']}"
                )
            ))

    passed = True
    n_eligible = len(eligible_categories)

    if n_eligible == 0:
        summary = (
            "No functional claim categories currently eligible. "
            "Reformulation opportunities identified above."
        )
        score = 60
    elif n_eligible == 1:
        summary = (
            f"1 functional claim category eligible: {eligible_categories[0]}. "
            f"Could unlock significant price premium."
        )
        score = 80
    else:
        summary = (
            f"{n_eligible} functional claim categories eligible: "
            f"{', '.join(eligible_categories[:3])}. "
            f"Strong basis for 機能性表示食品 registration."
        )
        score = 95

    return ComplianceReport(
        standard="Functional Food Eligibility (機能性表示食品/FOSHU)",
        passed=passed,
        score=score,
        summary=summary,
        issues=issues,
        passed_checks=passes,
        not_applicable=False,
    )


# Mandatory fields reminder (食品表示基準 Article 3). OCR presence checks are unreliable,
# so missing fields are INFO reminders for manual verification, never errors.

# (code, japanese_markers, english_name, law_ref, why_it_matters)
_MANDATORY_FIELD_REMINDERS = [
    (
        "JAS-MND-001",
        ["名称", "品名"],
        "Legal product category name (名称)",
        "食品表示基準 Article 3, Item 1",
        "Required on all processed foods. Must be the legal category name "
        "(e.g. スナック菓子), NOT the brand name. "
        "Missing this is a top CAA inspection finding for imported products."
    ),
    (
        "JAS-MND-002",
        ["原材料名", "原料名"],
        "Ingredients list (原材料名)",
        "食品表示基準 Article 3, Item 2",
        "All ingredients must be listed in descending order by weight. "
        "Additives must appear after a slash (/) separator. "
        "Each ingredient must use its legally designated Japanese name."
    ),
    (
        "JAS-MND-003",
        ["添加物", "乳化剤", "保存料", "着色料", "香料", "甘味料",
         "調味料", "酸化防止剤", "膨張剤", "増粘剤"],
        "Additive declaration (添加物)",
        "食品表示基準 Article 3, Item 2 (additive sub-column)",
        "Each additive must be listed by its function category name "
        "(e.g. 保存料(ソルビン酸)) and separated from main ingredients by '/'. "
        "Generic terms like '合成着色料' are no longer permitted — specific names required."
    ),
    (
        "JAS-MND-004",
        ["内容量"],
        "Net weight / content quantity (内容量)",
        "食品表示基準 Article 3, Item 3",
        "Must be stated in metric units only — g, ml, kg, or L. "
        "oz, lb, fl oz alone are illegal. "
        "For multi-unit packs, state both individual and total weight."
    ),
    (
        "JAS-MND-005",
        ["賞味期限", "消費期限"],
        "Best-before or use-by date (賞味期限 / 消費期限)",
        "食品表示基準 Article 3, Item 5",
        "Use 賞味期限 for stable goods (quality guarantee date) or "
        "消費期限 for perishables (safety deadline). "
        "Required on all products — even those with <30cm² label surface."
    ),
    (
        "JAS-MND-006",
        ["保存方法"],
        "Storage method (保存方法)",
        "食品表示基準 Article 3, Item 6",
        "Must specify storage conditions (temperature, light, humidity). "
        "Common format: 直射日光・高温多湿の場所を避けて保存してください. "
        "Frozen products must state 冷凍保存."
    ),
    (
        "JAS-MND-007",
        ["製造者", "販売者", "輸入者", "製造元", "販売元", "発売元"],
        "Manufacturer / distributor / importer (製造者/販売者)",
        "食品表示基準 Article 3, Item 7",
        "Must include full legal company name AND address. "
        "Imported products must show the Japanese importer (輸入者) — "
        "the foreign manufacturer's address alone is insufficient. "
        "P.O. Box addresses are not accepted."
    ),
    (
        "JAS-MND-008",
        ["原産地", "原材料原産地", "原産国", "国産", "北海道産", "都産",
         "県産", "道産", "府産", "産地", "made in"],
        "Country / region of origin (原産地)",
        "食品表示基準 Article 3, Item 8 (revised March 2022)",
        "Mandatory for the primary ingredient (重量割合最大の原材料) across "
        "22 food groups since March 2022 full enforcement. "
        "Acceptable formats: 国産, 〇〇県産, or country name for imports. "
        "If origin varies, the collective declaration format (又は表記) is permitted."
    ),
    (
        "JAS-MND-009",
        ["栄養成分", "エネルギー", "熱量"],
        "Nutrition facts table — 5 mandatory rows (栄養成分表示)",
        "食品表示基準 Article 3, Item 9",
        "All 5 rows are legally required: エネルギー(kcal), たんぱく質(g), "
        "脂質(g), 炭水化物(g), 食塩相当量(g). "
        "Values must be per serving AND the serving size must be stated. "
        "Voluntary nutrients (vitamins, minerals) may be added but are not required."
    ),
    (
        "JAS-MND-010",
        ["アレルギー", "アレルゲン", "特定原材料", "アレルギー物質"],
        "Allergen declaration section (アレルゲン表示)",
        "食品表示基準 Article 3, Schedule 4",
        "8 mandatory allergens (特定原材料) must be declared: "
        "小麦, そば, 卵, 乳, 落花生, えび, かに, くるみ. "
        "Declaration can be embedded in the ingredient list or in a separate "
        "allergen table — both formats are legally accepted."
    ),
    (
        "JAS-MND-011",
        ["/", "／"],
        "Additive slash separator (/) in ingredients (添加物区分)",
        "食品表示基準 Article 3, Item 2 — Additive labelling rules",
        "A slash (/) must separate main ingredients from additives in 原材料名. "
        "Example: 小麦粉、砂糖、バター／乳化剤(大豆由来)、香料. "
        "The ・ (middle dot) is NOT a legal separator for this purpose."
    ),
    (
        "JAS-MND-012",
        ["g", "ml", "kg", "ｇ", "ｍｌ", "ｋｇ"],
        "Metric units for net weight (メートル法単位)",
        "計量法 (Measurement Act) Article 8 / 食品表示基準 Article 3",
        "Net weight must be expressed in metric units (g, ml, kg, L). "
        "Imperial units (oz, lb, fl oz) cannot appear alone — metric must "
        "always be present. This applies even if the product is imported."
    ),
]

# Additive function terms, used to tell whether additives are present (JAS-MND-003, -011)
_ADDITIVE_FUNCTION_TERMS = [
    "乳化剤", "増粘剤", "着色料", "保存料", "酸化防止剤",
    "甘味料", "香料", "膨張剤", "調味料", "酸味料",
    "ph調整剤", "漂白剤", "発色剤", "防カビ剤", "光沢剤",
    "強化剤", "消泡剤", "製造用剤", "苦味料", "軟化剤",
    "アミノ酸", "ビタミンc", "ビタミンe",
]

_SLASH_LEGAL = ["/", "／"]   # only these are legally valid separators
_IMPERIAL_UNITS = ["oz", "lb", "fl oz", "ounce", "pound"]

_MANDATORY_NUTRITION_ROWS = [
    ("エネルギー", ["エネルギー", "熱量"],          "Energy (エネルギー)"),
    ("たんぱく質", ["たんぱく質", "タンパク質"],    "Protein (たんぱく質)"),
    ("脂質",       ["脂質", "脂肪"],                "Fat (脂質)"),
    ("炭水化物",   ["炭水化物"],                    "Carbohydrates (炭水化物)"),
    ("食塩相当量", ["食塩相当量", "ナトリウム", "塩分"], "Salt equivalent (食塩相当量)"),
]


def _check_mandatory_fields(nd: dict) -> ComplianceReport:
    """Emit INFO reminders for each mandatory label field OCR did not find (never an error).
    Also checks the 5 nutrition rows, metric units and the additive slash separator."""
    issues:  List[ComplianceIssue] = []
    passes:  List[str] = []

    raw_text    = str(nd.get("raw_text",    "") or "")
    ingredients = str(nd.get("ingredients", "") or "")
    ing_lower   = ingredients.lower()

    if not raw_text and not ingredients:
        return ComplianceReport(
            standard="Mandatory Label Fields (必須表示事項)",
            passed=True,
            score=None,
            summary="No label text available — mandatory fields reminder skipped.",
            issues=[],
            passed_checks=["No OCR text extracted; field verification could not run."],
            not_applicable=True,
        )

    searchable = raw_text
    if ingredients:
        searchable += "\n原材料名 " + ingredients

    confirmed   = 0
    unconfirmed = 0

    # Domestic products (国内製造 / 国産) need no origin check
    is_domestic = "国内製造" in searchable or "国産" in searchable

    for code, markers, field_name, law_ref, why in _MANDATORY_FIELD_REMINDERS:

        if code == "JAS-MND-008" and is_domestic:
            confirmed += 1
            passes.append(
                f"[{code}] {field_name} — 国内製造 detected: domestic product, "
                f"origin declaration satisfied ✓"
            )
            continue

        detected = any(m in searchable for m in markers)

        if detected:
            confirmed += 1
            passes.append(
                f"[{code}] {field_name} — detected by OCR ✓  |  {why}"
            )
        else:
            unconfirmed += 1
            issues.append(ComplianceIssue(
                severity="info",
                code=code,
                message=(
                    f"[VERIFY] {field_name} — not detected by OCR. "
                    f"Please confirm this field is present on the physical label. "
                    f"Why it matters: {why}  "
                    f"(Ref: {law_ref})"
                )
            ))

    # A row counts as present if the structured value exists or its Japanese label is in raw_text
    if not nd.get("no_nutrition_table", False):
        missing_rows = []
        row_check = [
            ("calories",      ["エネルギー", "熱量"],                  "Energy (エネルギー)"),
            ("protein",       ["たんぱく質", "タンパク質"],             "Protein (たんぱく質)"),
            ("fat",           ["脂質", "脂肪"],                        "Fat (脂質)"),
            ("carbohydrates", ["炭水化物"],                            "Carbohydrates (炭水化物)"),
            ("sodium",        ["食塩相当量", "ナトリウム", "塩分"],     "Salt equivalent (食塩相当量)"),
        ]
        for key, jp_labels, en_label in row_check:
            val = nd.get(key)
            found_in_raw = any(lbl in raw_text for lbl in jp_labels)
            if val is not None or found_in_raw:
                passes.append(f"Nutrition row {en_label} — present ✓")
            else:
                missing_rows.append(en_label)
        if missing_rows:
            for row in missing_rows:
                issues.append(ComplianceIssue(
                    severity="warning",
                    code="JAS-MND-009a",
                    message=(
                        f"Mandatory nutrition row '{row}' not detected. "
                        f"All 5 rows (energy, protein, fat, carbohydrates, "
                        f"salt equivalent) are required under "
                        f"食品表示基準 Article 3, Item 9. "
                        f"Fix: Add the missing row to the 栄養成分表示 table."
                    )
                ))

    raw_lower = raw_text.lower()
    imperial_found = [u for u in _IMPERIAL_UNITS if u in raw_lower]
    metric_found   = any(u in raw_text for u in ["g", "ml", "kg", "ｇ", "ｍｌ"])
    if imperial_found and not metric_found:
        issues.append(ComplianceIssue(
            severity="warning",
            code="JAS-MND-012a",
            message=(
                f"Imperial unit(s) detected ({', '.join(imperial_found)}) "
                f"but no metric unit (g/ml/kg) found. "
                f"Japan's Measurement Act requires net weight in metric units. "
                f"Imperial units may appear alongside metric but cannot appear alone."
            )
        ))
    elif not imperial_found:
        passes.append("Net weight units — metric units present, no imperial-only values ✓")

    if ingredients:
        additive_terms_found = [t for t in _ADDITIVE_FUNCTION_TERMS if t in ing_lower]
        has_legal_slash      = any(s in ingredients for s in _SLASH_LEGAL)

        if additive_terms_found and not has_legal_slash:
            issues.append(ComplianceIssue(
                severity="warning",
                code="JAS-MND-011a",
                message=(
                    f"Additive function terms detected "
                    f"({', '.join(additive_terms_found[:4])}"
                    f"{'…' if len(additive_terms_found) > 4 else ''}) "
                    f"but no legal slash separator (/ or ／) found in ingredient list. "
                    f"食品表示基準 requires additives after '/' — "
                    f"e.g. 小麦粉、砂糖／乳化剤、香料. "
                    f"Note: ・ (middle dot) is NOT a valid separator."
                )
            ))
        elif additive_terms_found and has_legal_slash:
            passes.append(
                f"Additive slash separator — {len(additive_terms_found)} "
                f"additive term(s) correctly separated by / ✓"
            )

    n_warn  = sum(1 for i in issues if i.severity == "warning")
    n_info  = sum(1 for i in issues if i.severity == "info")
    total   = confirmed + unconfirmed
    confirm_rate = int(confirmed / total * 100) if total else 100

    score = max(0, confirm_rate - (n_warn * 8))

    passed = n_warn == 0   # warnings = real issues; info = reminders only

    if n_warn == 0 and n_info == 0:
        summary = f"All {confirmed} mandatory fields confirmed by OCR."
    elif n_warn > 0:
        summary = (
            f"{n_warn} mandatory field issue(s) detected. "
            f"{confirmed}/{total} fields confirmed by OCR — "
            f"{unconfirmed} field(s) need manual verification."
        )
    else:
        summary = (
            f"{confirmed}/{total} mandatory fields confirmed by OCR. "
            f"{unconfirmed} field(s) could not be confirmed — please verify manually."
        )

    return ComplianceReport(
        standard="Mandatory Label Fields (必須表示事項)",
        passed=passed,
        score=score,
        summary=summary,
        issues=issues,
        passed_checks=passes,
        not_applicable=False,
    )


def run_compliance_check(nutrition_dict: dict) -> Dict[str, ComplianceReport]:
    """Run every compliance check on nutrition_dict.
    Returns a dict of check key -> ComplianceReport."""
    return {
        "food_labelling":       _check_food_labelling_standards(nutrition_dict),
        "nutrient_function":    _check_jas_nutrient_function(nutrition_dict),
        "organic":              _check_jas_organic(nutrition_dict),
        "allergen":             _check_allergen_labelling(nutrition_dict),
        "mandatory_fields":     _check_mandatory_fields(nutrition_dict),
        "ingredients":          _check_ingredients_compliance(nutrition_dict),
        "plausibility":         _check_ocr_plausibility(nutrition_dict),
        "nutrient_claims":      _check_nutrient_claims(nutrition_dict),
        "functional":           _check_functional_eligibility(nutrition_dict),
    }


# Ingredient scanner: aspartame needs the L-フェニルアラニン (PKU) warning (食品表示基準 Schedule 4);
# GMO-regulated crops need a declaration; some additives are restricted or must be named

_ASPARTAME_TERMS = [
    "アスパルテーム",
    "aspartame",
    "アスパルテーム・アセスルファムk",
    "アスパルテーム・アセスルファムカリウム",
]

_PKU_WARNING_TERMS = [
    "l-フェニルアラニン化合物を含む",
    "l-フェニルアラニン化合物",
    "フェニルアラニン",
    "phenylketonurics",
    "phenylalanine",
]

# GMO-regulated crops: (Japanese term, English name)
_GMO_REGULATED_CROPS = [
    ("大豆",         "soybean (大豆)"),
    ("とうもろこし", "corn (とうもろこし)"),
    ("ばれいしょ",   "potato (ばれいしょ)"),
    ("じゃがいも",   "potato (じゃがいも)"),
    ("なたね",       "rapeseed (なたね)"),
    ("綿実",         "cottonseed (綿実)"),
    ("アルファルファ","alfalfa (アルファルファ)"),
    ("てん菜",       "sugar beet (てん菜)"),
    ("パパイヤ",     "papaya (パパイヤ)"),
]

# Terms indicating a GMO declaration is already present
_GMO_DECLARATION_TERMS = [
    "遺伝子組換えでない",
    "遺伝子組換え不分別",
    "遺伝子組換え",
    "非遺伝子組換え",
    "gmo",
    "non-gmo",
    "遺伝子組み換えでない",
    "遺伝子組み換え",
    # Identity Preserved (IP) management counts as non-GMO
    "分別生産流通管理",
    "ip管理",
    "identity preserved",
]

# Derived ingredients exempt from GMO labelling (DNA/protein undetectable after refining)
_GMO_EXEMPT_DERIVED = [
    "大豆油", "大豆レシチン", "しょうゆ", "醤油", "味噌", "みそ",
    "コーンスターチ", "コーンシロップ", "水飴", "異性化液糖",
    "なたね油", "菜種油", "綿実油", "コーン油",
]

# Restricted additives: (terms, code, severity, message)
_RESTRICTED_ADDITIVES = [
    (
        ["安息香酸na", "安息香酸ナトリウム", "benzoic acid"],
        "JAS-ING-003",
        "warning",
        "安息香酸Na (sodium benzoate) detected. This preservative is prohibited "
        "in many food categories in Japan including fish paste products, soy sauce, "
        "and most beverages other than cola/fruit syrups "
        "(食品添加物公定書 / MHLW positive list). "
        "Verify that this additive is permitted for your specific product category."
    ),
    (
        ["合成着色料"],
        "JAS-ING-004",
        "warning",
        "'合成着色料' (synthetic food coloring) used as a generic blanket term. "
        "Since 2009, Japanese labelling standards require each synthetic dye to be "
        "named individually (e.g. 赤色40号, 黄色4号). The generic term '合成着色料' "
        "alone is non-compliant. Fix: Replace with the specific dye name(s)."
    ),
    (
        ["タール色素"],
        "JAS-ING-005",
        "warning",
        "'タール色素' (coal tar dye) used as a generic term. "
        "Each coal tar dye must be identified by its individual name "
        "(e.g. 赤色2号, 青色1号) under 食品表示基準. "
        "Fix: Replace with the specific approved dye name(s)."
    ),
]