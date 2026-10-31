"""Translates common Japanese food ingredient terms to English with a lookup dictionary.
No external API is used."""

# Japanese ingredient -> English
INGREDIENT_DICT = {
    # Grains & Starches
    "小麦粉":       "wheat flour",
    "強力粉":       "bread flour",
    "薄力粉":       "cake flour",
    "米粉":         "rice flour",
    "でんぷん":     "starch",
    "コーンスターチ": "corn starch",
    "タピオカ":     "tapioca",
    "玄米":         "brown rice",
    "白米":         "white rice",
    "もち米":       "glutinous rice",
    "オートミール": "oatmeal",
    "大麦":         "barley",

    # Dairy
    "牛乳":         "milk",
    "全粉乳":       "whole milk powder",
    "脱脂粉乳":     "skim milk powder",
    "バター":       "butter",
    "チーズ":       "cheese",
    "クリーム":     "cream",
    "生クリーム":   "fresh cream",
    "ヨーグルト":   "yogurt",
    "練乳":         "condensed milk",

    # Eggs
    "卵":           "egg",
    "全卵":         "whole egg",
    "卵黄":         "egg yolk",
    "卵白":         "egg white",
    "乾燥卵":       "dried egg",

    # Sweeteners
    "砂糖":         "sugar",
    "上白糖":       "refined white sugar",
    "グラニュー糖": "granulated sugar",
    "黒砂糖":       "brown sugar",
    "はちみつ":     "honey",
    "水飴":         "starch syrup",
    "ぶどう糖":     "glucose",
    "果糖":         "fructose",
    "乳糖":         "lactose",
    "ステビア":     "stevia",
    "アスパルテーム": "aspartame",
    "キシリトール": "xylitol",
    "ソルビトール": "sorbitol",

    # Fats & Oils
    "植物油脂":     "vegetable oil",
    "植物油":       "vegetable oil",
    "パーム油":     "palm oil",
    "大豆油":       "soybean oil",
    "菜種油":       "rapeseed oil",
    "オリーブ油":   "olive oil",
    "ごま油":       "sesame oil",
    "ラード":       "lard",
    "ショートニング": "shortening",
    "マーガリン":   "margarine",

    # Proteins
    "大豆":         "soybean",
    "豆腐":         "tofu",
    "味噌":         "miso",
    "醤油":         "soy sauce",
    "鶏肉":         "chicken",
    "豚肉":         "pork",
    "牛肉":         "beef",
    "魚":           "fish",
    "かつお":       "bonito",
    "まぐろ":       "tuna",
    "鮭":           "salmon",
    "えび":         "shrimp",
    "かに":         "crab",
    "いか":         "squid",

    # Vegetables
    "にんじん":     "carrot",
    "たまねぎ":     "onion",
    "じゃがいも":   "potato",
    "トマト":       "tomato",
    "キャベツ":     "cabbage",
    "ほうれん草":   "spinach",
    "にんにく":     "garlic",
    "しょうが":     "ginger",
    "とうもろこし": "corn",

    # Seasonings & Additives
    "食塩":         "salt",
    "塩":           "salt",
    "酢":           "vinegar",
    "みりん":       "mirin (sweet rice wine)",
    "酒":           "sake",
    "だし":         "dashi stock",
    "こんぶ":       "kelp",
    "かつおぶし":   "bonito flakes",
    "MSG":          "monosodium glutamate",
    "グルタミン酸ナトリウム": "monosodium glutamate (MSG)",

    # Additives & Preservatives
    "乳化剤":       "emulsifier",
    "増粘剤":       "thickener",
    "着色料":       "food coloring",
    "保存料":       "preservative",
    "酸化防止剤":   "antioxidant",
    "甘味料":       "sweetener",
    "香料":         "flavoring",
    "膨張剤":       "leavening agent",
    "ベーキングパウダー": "baking powder",
    "イースト":     "yeast",
    "ゼラチン":     "gelatin",
    "寒天":         "agar",
    "ペクチン":     "pectin",
    "カラギナン":   "carrageenan",
    "レシチン":     "lecithin",
    "ビタミンC":    "vitamin C",
    "ビタミンE":    "vitamin E",
    "カロテン":     "carotene",
    "リン酸塩":     "phosphate",

    # Nuts & Seeds
    "アーモンド":   "almond",
    "くるみ":       "walnut",
    "カシューナッツ": "cashew",
    "ピーナッツ":   "peanut",
    "落花生":       "peanut",
    "ごま":         "sesame",
    "亜麻仁":       "flaxseed",
    "チアシード":   "chia seed",

    # Fruits
    "りんご":       "apple",
    "バナナ":       "banana",
    "オレンジ":     "orange",
    "レモン":       "lemon",
    "いちご":       "strawberry",
    "ぶどう":       "grape",
    "もも":         "peach",
    "なし":         "pear",

    # Processed/prepared foods
    "じゃがいも":   "potato",
    "肉じゃが":     "meat and potato stew",
    "豚肉":         "pork",
    "牛肉":         "beef",
    "でん粉":       "starch",
    "海藻粉末":     "seaweed powder",
    "水溶性食物繊維": "soluble dietary fiber",
    "発酵調味料":   "fermented seasoning",
    "しょうゆ":     "soy sauce",
    "醤油":         "soy sauce",
    "ビーフエキス": "beef extract",
    "増粘剤":       "thickener",
    "キサンタン":   "xanthan gum",
    "こんにゃく":   "konjac",
    "米こんにゃく": "rice konjac",
    "カラギナン":   "carrageenan",
    "難消化性デキストリン": "indigestible dextrin",
    "デキストリン": "dextrin",
    "カツオエキス": "bonito extract",
    "粉末油脂":     "powdered fat",
    "香辛料":       "spices",
    "たまねぎ":     "onion",
    "にんじん":     "carrot",
    "調味料":       "seasoning",
    "アミノ酸":     "amino acids",
    "ペプチド":     "peptides",
    "コラーゲン":   "collagen",
    "ヒアルロン酸": "hyaluronic acid",
    "コエンザイムQ10": "coenzyme Q10",
    "カフェイン":   "caffeine",
    "乳蛋白":       "milk protein",
    "大豆蛋白":     "soy protein",
    # Fruit concentrates & candy extras
    "濃縮グレープ果汁":      "concentrated grape juice",
    "濃縮ストロベリー果汁":  "concentrated strawberry juice",
    "濃縮アップル果汁":      "concentrated apple juice",
    "濃縮ライム果汁":        "concentrated lime juice",
    "ミックスジャム":        "mixed jam",
    "乳酸菌飲料":            "lactic acid bacteria drink",
    "乳成分":                "milk solids",
    "乳酸菌":                "lactic acid bacteria",
    # Candy / confectionery
    "水あめ":         "starch syrup",
    "グリセリン":     "glycerin",
    "酸味料":         "acidulant",
    "アカキャベツ色素": "red cabbage pigment",
    "ベニバナ黄色素":  "safflower yellow pigment",
    "クチナシ青色素":  "gardenia blue pigment",
    "濃縮りんご果汁":  "concentrated apple juice",
    "濃縮いちご果汁":  "concentrated strawberry juice",
    "濃縮ライム果汁":  "concentrated lime juice",
    "果汁":           "fruit juice",
    "色素":           "food coloring",
    "乳化剤":         "emulsifier",
    "香料":           "flavoring",
    "酸味料":         "acidulant",
    "増粘多糖類":     "thickening polysaccharides",
    # Other common
    "水":           "water",
    "食物繊維":     "dietary fiber",
    "コラーゲン":   "collagen",
    "カルシウム":   "calcium",
    "鉄":           "iron",
    "亜鉛":         "zinc",
    "マグネシウム": "magnesium",
}


def translate_ingredients(japanese_text: str) -> str:
    """Replace known Japanese ingredient terms in the text with English; unknown terms are left as-is."""
    if not japanese_text:
        return ""

    result = japanese_text
    # Sort by length descending to match longer phrases first
    for jp, en in sorted(INGREDIENT_DICT.items(), key=lambda x: -len(x[0])):
        result = result.replace(jp, en)

    return result


def get_ingredient_list(japanese_text: str) -> list:
    """Return the translated ingredients as a list, split on common delimiters."""
    if not japanese_text:
        return []
    translated = translate_ingredients(japanese_text)
    import re
    items = re.split(r"[、，,・/／\s]+", translated)
    return [i.strip() for i in items if i.strip()]


if __name__ == "__main__":
    test = "小麦粉、砂糖、バター、卵、食塩、乳化剤、香料"
    print("Original:", test)
    print("Translated:", translate_ingredients(test))
    print("As list:", get_ingredient_list(test))
