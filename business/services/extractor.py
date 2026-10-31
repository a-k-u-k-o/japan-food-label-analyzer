"""Extracts nutrition facts, ingredients and allergens from Japanese label OCR text.
Only values actually found are returned; missing nutrients stay None."""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

# All 28 Japanese allergens (7 mandatory + 21 recommended): Japanese keyword -> English name
ALLERGEN_MAP = {
    # 7 Mandatory (特定原材料) — legally required on label
    "小麦":           "Wheat",
    "そば":           "Buckwheat",
    "卵":             "Egg",
    "たまご":         "Egg",
    "乳":             "Milk",
    "乳成分":         "Milk",
    "牛乳":           "Milk",
    "落花生":         "Peanut",
    "ピーナッツ":     "Peanut",
    "えび":           "Shrimp",
    "海老":           "Shrimp",
    "かに":           "Crab",
    "蟹":             "Crab",
    # 21 Recommended (特定原材料に準ずるもの)
    "アーモンド":     "Almond",
    "あわび":         "Abalone",
    "いか":           "Squid",
    "いくら":         "Salmon roe",
    "オレンジ":       "Orange",
    "カシューナッツ": "Cashew",
    "キウイフルーツ": "Kiwi",
    "キウイ":         "Kiwi",
    "牛肉":           "Beef",
    "くるみ":         "Walnut",
    "ごま":           "Sesame",
    "さけ":           "Salmon",
    "さば":           "Mackerel",
    "大豆":           "Soy",
    "鶏肉":           "Chicken",
    "バナナ":         "Banana",
    "豚肉":           "Pork",
    "まつたけ":       "Matsutake mushroom",
    "もも":           "Peach",
    "やまいも":       "Yam",
    "りんご":         "Apple",
    "ゼラチン":       "Gelatin",
    # Ingredient-embedded variants
    "小麦粉":         "Wheat",
    "小麦でん粉":     "Wheat",
    "小麦たんぱく":   "Wheat",
    "大豆由来":       "Soy",
    "大豆油":         "Soy",
    "大豆たんぱく":   "Soy",
    "卵黄":           "Egg",
    "卵白":           "Egg",
    "全卵":           "Egg",
    "乾燥卵":         "Egg",
    "脱脂粉乳":       "Milk",
    "全粉乳":         "Milk",
    "バター":         "Milk",
    "チーズ":         "Milk",
    "生クリーム":     "Milk",
    "練乳":           "Milk",
    "ホエイ":         "Milk",
    "カゼイン":       "Milk",
    "落花生油":       "Peanut",
    "ごま油":         "Sesame",
    "いか墨":         "Squid",
}

# The 7 mandatory allergens in English, so the ingredient scan doesn't re-add
# what the ○印 grid already declared (recommended 21品目 are still added)
MANDATORY_ALLERGENS_EN = {
    "Wheat", "Buckwheat", "Egg", "Milk", "Peanut", "Shrimp", "Crab"
}

NUTRIENT_PATTERNS = [
    # (field_name, japanese_labels, regex_for_value)
    ("calories",      ["エネルギー", "熱量"],
                      r"(\d+(?:\.\d+)?)\s*kcal"),
    ("protein",       ["たんぱく質", "タンパク質", "蛋白質"],
                      r"(\d+(?:\.\d+)?)\s*g"),
    ("fat",           ["脂質", "脂肪", "総脂肪", "脂肪分"],
                      r"(\d+(?:\.\d+)?)\s*g"),
    ("carbohydrates", ["炭水化物"],
                      r"(\d+(?:\.\d+)?)\s*g"),
    ("sodium",        ["食塩相当量", "塩分"],
                      r"(\d+(?:\.\d+)?)\s*g"),
    ("sodium_mg",     ["ナトリウム"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("sugar",         ["糖類", "糖質"],
                      r"(\d+(?:\.\d+)?)\s*g"),
    ("fiber",         ["食物繊維"],
                      r"(\d+(?:\.\d+)?)\s*g"),
    ("calcium",       ["カルシウム"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("iron",          ["鉄"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("vitamin_c",     ["ビタミンC", "ビタミンｃ", "VC"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("vitamin_d",     ["ビタミンD"],
                      r"(\d+(?:\.\d+)?)\s*[μµ]g"),
    ("potassium",     ["カリウム"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("magnesium",     ["マグネシウム"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("zinc",          ["亜鉛"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("selenium",      ["セレン"],
                      r"(\d+(?:\.\d+)?)\s*[μµ]g"),
    ("chromium",      ["クロム"],
                      r"(\d+(?:\.\d+)?)\s*[μµ]g"),
    ("manganese",     ["マンガン"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("vitamin_a",     ["ビタミンA"],
                      r"(\d+(?:\.\d+)?)\s*[μµ]g"),
    ("vitamin_b1",    ["ビタミンB1", "ビタミンB₁"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("vitamin_b2",    ["ビタミンB2", "ビタミンB₂"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("vitamin_b6",    ["ビタミンB6", "ビタミンB₆"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("vitamin_b12",   ["ビタミンB12", "ビタミンB₁₂"],
                      r"(\d+(?:\.\d+)?)\s*[μµ]g"),
    ("vitamin_e",     ["ビタミンE"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("niacin",        ["ナイアシン"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
    ("pantothenic",   ["パントテン酸"],
                      r"(\d+(?:\.\d+)?)\s*mg"),
]

SERVING_PATTERNS = [
    # Per-serving patterns only; 内容量 (package total) is excluded on purpose.
    # Order matters: most specific first.
    r"基準量\s*[：:]\s*\S*?[\(（](\d+(?:\.\d+)?)\s*g[\)）]",   # 基準量: 1粒(4.3g) / 基準量: 1箱(53g)
    r"基準量\s*[：:]\s*(\d+(?:\.\d+)?)\s*g\b",                  # 基準量: 53g (bare)
    r"\d+\s*粒\s*[\(（]\s*標準\s*(\d+(?:\.\d+)?)\s*g\s*[\)？]", # 1粒(標準4.3g) — space-tolerant
    r"\d+\s*[個袋食本枚缶瓶杯粒切皿膳包]\s*分?\s*[\(（][^\)）]{0,20}?(\d+(?:\.\d+)?)\s*g[\)）]",  # 1個分(製品23g) / 1袋(45g)
    r"\d+\s*箱\s*分?\s*[\(（][^\)）]{0,20}?(\d+(?:\.\d+)?)\s*g[\)）]",  # 1箱分(53g) / 1箱(53g)
    r"製品\s*(\d+(?:\.\d+)?)\s*g",                              # 製品23g standalone
    r"[\(（]\s*(\d+(?:\.\d+)?)\s*g\s*[\)）]\s*[当あ]た[りリ]",   # (53g)当たり
    r"約\s*(\d+(?:\.\d+)?)\s*g\s*[当あ]た[りリ]",
    r"[\(（]約\s*(\d+(?:\.\d+)?)\s*g[\)）]",
    r"コップ\s*\d+\s*杯\s*[\(（](\d+(?:\.\d+)?)\s*ml[\)？]",
    r"(\d+(?:\.\d+)?)\s*g\s*[当あ]た[りリ]",
    r"(\d+(?:\.\d+)?)\s*ml\s*[当あ]た[りリ]",
]

INGREDIENT_MARKERS = ["原材料名", "原料名", "原料", "材料"]
ALLERGEN_MARKERS   = ["アレルギー物質", "アレルゲン", "特定原材料", "アレルギー"]
FACTORY_MARKERS    = ["製造工場", "工場では", "を含む製品を生産",
                       "共通の設備で製造", "同じ設備で製造",
                       "乳を含む製品と",
                       "を含む製品と共通",
                       "を使用した製品と共通",   # Chocobi style
                       "特定原材料中",            # Koikeya style
                       "を生産しています"]

# markers where an ingredient/allergen section ends
SECTION_ENDS = [
    "栄養成分", "賞味期限", "保存方法", "内容量",
    "品名", "名称", "製造者", "販売者", "製造所",
    "電子レンジ", "開封後", "取扱", "注意", "※", "＊"
]


@dataclass
class NutritionData:
    """Parsed nutrition facts, ingredients and allergens for one label."""
    calories:      Optional[float] = None
    protein:       Optional[float] = None
    fat:           Optional[float] = None
    carbohydrates: Optional[float] = None
    sodium:        Optional[float] = None
    sugar:         Optional[float] = None
    fiber:         Optional[float] = None
    calcium:       Optional[float] = None
    iron:          Optional[float] = None
    vitamin_c:     Optional[float] = None
    vitamin_d:     Optional[float] = None
    potassium:     Optional[float] = None
    magnesium:     Optional[float] = None
    zinc:          Optional[float] = None
    selenium:      Optional[float] = None
    chromium:      Optional[float] = None
    manganese:     Optional[float] = None
    vitamin_a:     Optional[float] = None
    vitamin_b1:    Optional[float] = None
    vitamin_b2:    Optional[float] = None
    vitamin_b6:    Optional[float] = None
    vitamin_b12:   Optional[float] = None
    vitamin_e:     Optional[float] = None
    niacin:        Optional[float] = None
    pantothenic:   Optional[float] = None
    serving_size:  Optional[str]   = None
    ingredients:   Optional[str]   = None
    allergens_in_product: list = field(default_factory=list)
    no_nutrition_table:   bool = False  # True when label has no nutrition table
    daily_value_pct:      dict = field(default_factory=dict)  # nutrient → % DV
    flavor:        Optional[str]   = None   # e.g. あっさりサラダ味, グレープ
    variant:       Optional[str]   = None   # e.g. 5食入, アソート


def normalize_text(text: str) -> str:
    """NFKC-normalise text and remove spaces OCR inserts between Japanese characters ('脂　質' -> '脂質')."""
    text = unicodedata.normalize("NFKC", text)
    jp_char = "[\u3040-\u9fff]"
    pat = re.compile("(" + jp_char + ")[ \t\u3000]+(" + jp_char + ")")
    for _ in range(8):
        text = pat.sub(r"\1\2", text)
    return text


def _clip_section(text: str, start_idx: int, window: int = 500) -> str:
    """Return text from start_idx, cut at the next section boundary."""
    snippet = text[start_idx: start_idx + window]
    for marker in SECTION_ENDS:
        pos = snippet.find(marker)
        if pos > 15:
            snippet = snippet[:pos]
    return snippet


# Words containing an allergen term that must not count (e.g. 乳 inside 乳化剤, emulsifier)
ALLERGEN_EXCLUSIONS = {
    "乳":    ["乳化剤", "乳酸", "乳酸菌", "乳酸カルシウム", "乳酸ナトリウム",
              "離乳食",               # baby food
              "乳清",                 # whey
              "乳等を主要原料とする食品",  # dairy-based processed food
              "乳等",                 # short form
              "乳糖",                 # lactose additive
              "脱脂粉乳",             # skim milk powder ingredient
              "未焼成Ca",             # calcium compound
              ],
    # dairy ingredients that appear as ingredient names, not allergen declarations
    "バター":   ["バターパウダー", "バター風味料", "バターフレーバー",
                 "バターオイル", "バタービスケット"],
    "ホエイ":   ["ホエイパウダー", "ホエイたんぱく", "ホエイプロテイン"],
    "カゼイン": ["カゼインNa", "カゼインナトリウム"],
    "練乳":     ["加糖れん乳"],  # sweetened condensed milk — ingredient name
    "卵":    ["卵形", "卵巣"],
    "もも":  ["もも肉"],   # もも = peach but もも肉 = thigh meat
    "いか":  ["いかり"],
    "さけ":  ["さけび", "さけぶ"],  # cry/shout in Japanese
    "大豆":  [],  # no exclusions — 大豆 always means soy
}

# Marketing phrases meaning "no allergens"; lines containing them are ignored
NO_ALLERGEN_CLAIM_PHRASES = [
    "アレルギーに配慮",
    "アレルギーや添加物に配慮",
    "アレルゲン不使用",
    "アレルギー物質不使用",
    "アレルギー対応",
]

def _scan_allergens(text: str) -> set:
    """Return the set of English allergen names found in text."""
    found = set()
    for jp, en in ALLERGEN_MAP.items():
        if jp not in text:
            continue
        # drop excluded compound words, then see if the term still appears
        exclusions = ALLERGEN_EXCLUSIONS.get(jp, [])
        temp = text
        for excl in exclusions:
            temp = temp.replace(excl, "")
        if jp in temp:
            found.add(en)
    return found


def _ocr_has_real_nutrition_table(text: str) -> bool:
    """Check that OCR text is a real nutrition label, not a recipe page.
    Three or more recipe words before the nutrition marker count as a recipe."""
    has_calorie_label = any(k in text for k in ["エネルギー", "熱量"])
    has_kcal_unit     = "kcal" in text
    has_protein_label = any(k in text for k in ["たんぱく質", "蛋白質", "タンパク質"])

    if not (has_calorie_label and has_kcal_unit and has_protein_label):
        return False

    nutrition_start = len(text)  # default: end of text
    for marker in ["栄養成分表示", "栄養成分"]:
        idx = text.find(marker)
        if idx != -1:
            nutrition_start = min(nutrition_start, idx)

    pre_nutrition = text[:nutrition_start]
    recipe_signals = ["作り方", "いためる", "煮込む", "ルウを入れる",
                      "材料", "炊き方", "召し上がり方", "人分"]
    recipe_count = sum(1 for s in recipe_signals if s in pre_nutrition)

    if recipe_count >= 3:
        return False

    return True


def extract_nutrition(text: str) -> NutritionData:
    """Parse serving size, nutrients, ingredients, allergens, % daily values, flavor and variant from OCR text."""
    text = normalize_text(text)
    data = NutritionData()

    # Qwen sometimes splits numbers in dense table headers; collapse them first
    # "5 3 g" -> "53g"  |  "4 . 3 g" -> "4.3g"
    _serving_text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)          # "5 3" → "53"
    _serving_text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', _serving_text)  # "4 . 3" → "4.3"

    _NUTRITION_MARKERS = ["栄養成分表示", "栄養成分", "栄養価"]
    _nutr_start = len(_serving_text)
    for _m in _NUTRITION_MARKERS:
        _idx = _serving_text.find(_m)
        if _idx != -1:
            _nutr_start = min(_nutr_start, _idx)

    # 600-char window covers labels with long header text
    _nutr_scope = _serving_text[_nutr_start: _nutr_start + 600] if _nutr_start < len(_serving_text) else ""

    TIER1_PATTERNS = SERVING_PATTERNS

    def _try_patterns(search_text, patterns):
        for pat in patterns:
            m = re.search(pat, search_text)
            if m:
                captured = m.group(1) if m.lastindex else m.group(0)
                try:
                    g = float(re.sub(r'[^\d.]', '', captured))
                    if 1 <= g <= 500:   # serving sizes >500g are package totals
                        return m.group(0).strip()
                except (ValueError, TypeError):
                    pass
        return None

    # nutrition section first, then full text (safe because 内容量 is excluded from the patterns)
    data.serving_size = _try_patterns(_nutr_scope, TIER1_PATTERNS)

    if data.serving_size is None:
        data.serving_size = _try_patterns(_serving_text, TIER1_PATTERNS)

    # Isolate the nutrition section so nutrient names inside ingredients don't match
    # (e.g. 水溶性食物繊維 in ingredients matching 食物繊維)
    nutrition_section_markers = ["栄養成分表示", "栄養成分", "栄養価"]
    nutrition_text = None

    for marker in nutrition_section_markers:
        idx = text.find(marker)
        if idx == -1:
            continue
        candidate = text[idx:]

        # Qwen signals no table with NOT PRESENT
        if "NOT PRESENT" in candidate[:50]:
            data.no_nutrition_table = True
            break

        # Must have kcal AND at least one nutrient name to be a real table
        has_kcal     = bool(re.search(r"\d+\s*kcal", candidate[:300]))
        has_nutrient = any(n in candidate[:300] for n in
                           ["たんぱく質", "脂質", "炭水化物", "エネルギー", "熱量", "食塩相当量"])
        if has_kcal and has_nutrient:
            nutrition_text = candidate
            break

    # Even with a section marker, require calories + unit + protein in the full text (catches Qwen hallucinations)
    if nutrition_text is not None and not _ocr_has_real_nutrition_table(text):
        nutrition_text = None

    if nutrition_text is None and not data.no_nutrition_table:
        data.no_nutrition_table = True

    sodium_mg_val = None
    if nutrition_text is None:
        nutrition_text = ""   # empty string — nothing will match
    for field_name, labels, value_pat in NUTRIENT_PATTERNS:
        for label in labels:
            idx = nutrition_text.find(label)
            if idx == -1:
                continue
            # Search window: 80 chars to handle values on next line
            snippet = nutrition_text[idx: idx + 80]
            m = re.search(value_pat, snippet)
            if m:
                try:
                    val = float(m.group(1))
                    if field_name == "sodium_mg":
                        sodium_mg_val = val
                    else:
                        if getattr(data, field_name) is None:
                            setattr(data, field_name, val)
                    break
                except ValueError:
                    pass

    # Convert ナトリウム(mg) → 食塩相当量(g) if 食塩相当量 not directly found
    # Japanese law formula: NaCl(g) = Na(mg) × 2.54 / 1000
    if data.sodium is None and sodium_mg_val is not None:
        data.sodium = round(sodium_mg_val * 2.54 / 1000, 3)

    for marker in INGREDIENT_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            snippet = _clip_section(text, idx + len(marker), window=800)
            for stop in ["アレルギー", "アレルゲン", "製造工場", "販売者調べ",
                         "殺菌方法", "殺菌", "内容量", "賞味期限"]:
                pos = snippet.find(stop)
                if pos > 10:
                    snippet = snippet[:pos]
            snippet = re.sub(r"[：:]", "", snippet)
            snippet = re.sub(r"<[^>]+>", " ", snippet)   # remove <flavor tags>
            snippet = re.sub(r"\n+", " ", snippet)
            snippet = re.sub(r"\s{2,}", " ", snippet)
            snippet = snippet.strip(" 　、。")
            # Must contain Japanese characters to be valid
            if snippet and re.search(r"[\u3040-\u9fff]", snippet):
                data.ingredients = snippet[:800]
                break

    # Allergens come from the allergen section plus the ingredient list;
    # cross-contact / factory warnings (shared equipment) are excluded.
    product_allergens = set()

    # 1. dedicated allergen section
    for marker in ALLERGEN_MARKERS:
        idx = text.find(marker)
        if idx == -1:
            continue

        snippet = _clip_section(text, idx, window=400)

        # Cut at factory warning markers — don't let cross-contact bleed in
        for fm in FACTORY_MARKERS:
            pos = snippet.find(fm)
            if pos > 0:
                snippet = snippet[:pos]
                break

        clean_lines = [
            line for line in snippet.splitlines()
            if not any(fm in line for fm in FACTORY_MARKERS)
            and not any(p in line for p in NO_ALLERGEN_CLAIM_PHRASES)
        ]
        snippet = "\n".join(clean_lines)

        # "卵・小麦・落花生・えび・かに" is the shared-equipment list, not a declaration
        if re.search(r'卵[・]小麦[・]落花生[・]えび[・]かに', snippet):
            break

        found = _scan_allergens(snippet)
        if not found:
            break

        # If all 7 are marked the model probably read the grid header as marks;
        # keep only what the ingredients confirm.
        ALL_7 = {"Egg", "Milk", "Wheat", "Buckwheat", "Peanut", "Shrimp", "Crab"}
        if found >= ALL_7 and data.ingredients:
            ing_confirmed = _scan_allergens(data.ingredients)
            confirmed = found & ing_confirmed
            product_allergens |= confirmed if confirmed else found
        else:
            product_allergens |= found
        break

    # 2. ingredient list
    if data.ingredients:
        product_allergens |= _scan_allergens(data.ingredients)

    data.allergens_in_product = sorted(product_allergens)

    # % daily value (摂取目安に対する割合)
    DV_LABEL_MAP = {
        "エネルギー": "calories",   "熱量": "calories",
        "たんぱく質": "protein",    "タンパク質": "protein",
        "脂質": "fat",              "脂肪": "fat",
        "炭水化物": "carbohydrates",
        "食塩相当量": "sodium",     "塩分": "sodium",
        "糖類": "sugar",            "糖質": "sugar",
        "食物繊維": "fiber",
        "カルシウム": "calcium",    "鉄": "iron",
        "ビタミンC": "vitamin_c",   "ビタミンD": "vitamin_d",
        "ビタミンE": "vitamin_e",   "ビタミンA": "vitamin_a",
        "ビタミンB1": "vitamin_b1", "ビタミンB2": "vitamin_b2",
        "ビタミンB6": "vitamin_b6", "ビタミンB12": "vitamin_b12",
        "ナイアシン": "niacin",     "パントテン酸": "pantothenic",
        "カリウム": "potassium",    "マグネシウム": "magnesium",
        "亜鉛": "zinc",
    }
    if nutrition_text:
        dv_pat = re.compile(
            r'(' + '|'.join(re.escape(k) for k in DV_LABEL_MAP) + r')'
            r'.{0,30}?(\d+(?:\.\d+)?)\s*%'
        )
        for m in dv_pat.finditer(nutrition_text):
            fn = DV_LABEL_MAP.get(m.group(1))
            if fn and fn not in data.daily_value_pct:
                try:
                    data.daily_value_pct[fn] = float(m.group(2))
                except ValueError:
                    pass

    # Pattern 1: <グレープ> tags in ingredients
    flavor_tags = re.findall(r'<([^\d>]{1,15})>', text)
    jp_char = re.compile(r'[\u3040-\u9fff]')
    flavor_tags = [t for t in flavor_tags if jp_char.search(t)]
    if flavor_tags:
        data.flavor = '・'.join(flavor_tags[:4])

    # Pattern 2: 〇〇味 / フレーバー near product info
    if not data.flavor:
        taste_m = re.search(
            r'([^\s　\n]{2,10}(?:味|フレーバー|フレイバー))(?:\s|$|、|。)', text
        )
        if taste_m:
            data.flavor = taste_m.group(1).strip()

    variant_m = re.search(
        r'(\d+\s*(?:食入|本入|個入|袋入|枚入|粒入|缶入|食分|本組))', text
    )
    if variant_m:
        data.variant = variant_m.group(1).strip()

    return data


def nutrition_to_dict(data: NutritionData) -> dict:
    """Convert NutritionData to a dict (nutrients not found stay None)."""
    result = {}
    fields = [
        "calories", "protein", "fat", "carbohydrates", "sodium",
        "sugar", "fiber", "calcium", "iron", "zinc",
        "magnesium", "potassium", "manganese", "selenium", "chromium",
        "vitamin_a", "vitamin_b1", "vitamin_b2", "vitamin_b6",
        "vitamin_b12", "vitamin_c", "vitamin_d", "vitamin_e",
        "niacin", "pantothenic",
        "serving_size", "ingredients", "allergens_in_product",
        "no_nutrition_table", "daily_value_pct", "flavor", "variant",
    ]
    for f in fields:
        result[f] = getattr(data, f)
    return result