"""Three-stage OCR pipeline for Japanese food labels using Qwen2.5-VL: ingredients, nutrition, then translation."""

import gc
import os
import re
import json
import time
import torch
import streamlit as st
from PIL import Image
from transformers import BitsAndBytesConfig
from typing import TYPE_CHECKING, Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

if TYPE_CHECKING:
    from transformers import (  # type: ignore[import]
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
    )

_model:     Any = None
_processor: Any = None
_device:    Any = None

_MAX_PIXELS    = int(os.environ.get("OCR_MAX_PIXELS",    str(1280 * 720)))
_MAX_TOKENS_S1 = int(os.environ.get("OCR_MAX_TOKENS_S1", "640"))
_MAX_TOKENS_S2 = int(os.environ.get("OCR_MAX_TOKENS_S2", "512"))

_OCR_MAX_SIZE = 1024

JP_ALLERGEN_MAP = {
    "小麦": "wheat",       "乳": "milk",           "卵": "egg",
    "そば": "buckwheat",   "落花生": "peanut",      "えび": "shrimp",
    "かに": "crab",        "くるみ": "walnut",      "大豆": "soy",
    "ごま": "sesame",      "アーモンド": "almond",  "カシューナッツ": "cashew nut",
    "牛肉": "beef",        "豚肉": "pork",          "鶏肉": "chicken",
    "さけ": "salmon",      "さば": "mackerel",      "いか": "squid",
    "あわび": "abalone",   "いくら": "salmon roe",  "オレンジ": "orange",
    "バナナ": "banana",    "もも": "peach",         "りんご": "apple",
    "ゼラチン": "gelatin", "まつたけ": "matsutake", "やまいも": "yam",
    "マカダミアナッツ": "macadamia nut",
}


def _load_model():
    """Load the 4-bit Qwen2.5-VL model and processor once and cache them in module globals."""
    global _model, _processor, _device
    if _model is not None:
        return

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    _device   = "cuda" if torch.cuda.is_available() else "cpu"
    model_id  = "Qwen/Qwen2.5-VL-7B-Instruct"

    _processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=200 * 200,
        max_pixels=_MAX_PIXELS,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    _model.eval()
    st.session_state["_consumer_model_ref"] = _model


SYSTEM_OCR = (
    "You are a strict Japanese food label OCR engine.\n"
    "Copy every character EXACTLY as printed. Do not correct, normalise, or infer.\n"
    "Critical katakana pairs: ゲ(ge)≠グ(gu)  ペ(pe)≠ベ(be)  ソ≠ン  シ≠ツ  ー≠一  ヴ≠ウ\n"
    "Output ONLY valid JSON. No markdown, no prose."
)

# Stage 1a: ingredients and allergens
PROMPT_INGREDIENTS = (
    'IMPORTANT: If this any other image instead of a Japanese food label, return ONLY {"error": "not_a_label"}\n'
    'IMPORTANT: If the language of the text is not english or japanese, return ONLY {"error": "not_a_label"}\n'
    'IMPORTANT: If the image has no food label textual content in English or Japanese, return ONLY {"error": "not_a_label"}\n'
    "\n"
    "Scan the ENTIRE image left to right, top to bottom.\n"
    "Find the ingredients section (原材料名) and allergen section (アレルギー物質 / 一部に).\n"
    "Return ONLY this JSON — null for anything not found:\n"
    "{\n"
    '  "product_name": null,\n'
    '  "ingredients": { "raw_text": null, "items": [] },\n'
    '  "allergens":   { "style": null,    "items": [] }\n'
    "}\n"
    "\n"
    "Rules:\n"
    "- product_name: printed product name only — NOT a slogan or description\n"
    "- ingredients.raw_text: copy ONLY the content after 原材料名 label — verbatim\n"
    "- ingredients.items: split raw_text on 、(Japanese comma) only; "
    "keep （）bracket groups attached to the item before them\n"
    "- allergens.items: names from （）brackets after ingredient items AND "
    "any collective list after 一部に or （N品目中）\n"
    "- allergens.style: individual | collective | both | none\n"
    "- Do NOT include slogans, handling instructions, or storage text\n"
)

# Stage 1b: nutrition table only
PROMPT_NUTRITION = (
    "Scan the ENTIRE image. Find the nutrition facts table (栄養成分表示).\n"
    "It is often in a box on the RIGHT side of the label.\n"
    "The serving basis line (e.g. 'コップ1杯(200ml)当たり') is printed ABOVE or INSIDE the table.\n"
    "\n"
    "Return ONLY this JSON (null for any row not present in the table):\n"
    "{\n"
    '  "nutrition": {\n'
    '    "basis":         null,\n'
    '    "calories":      null,\n'
    '    "protein":       null,\n'
    '    "fat":           null,\n'
    '    "saturated_fat": null,\n'
    '    "carbohydrate":  null,\n'
    '    "sugar":         null,\n'
    '    "fibre":         null,\n'
    '    "salt":          null\n'
    "  }\n"
    "}\n"
    "\n"
    "STRICT field mapping — copy the printed value (number + unit) EXACTLY:\n"
    "  calories      ← エネルギー  row ONLY  (e.g. 137kcal)\n"
    "  protein       ← たんぱく質  row ONLY  (e.g. 6.8g)\n"
    "  fat           ← 脂　質      row ONLY  (e.g. 7.8g)\n"
    "  saturated_fat ← 飽和脂肪酸  row ONLY  — null if this row is absent.\n"
    "                  DO NOT use 食塩相当量 or any other row for this field.\n"
    "  carbohydrate  ← 炭水化物    row ONLY  (e.g. 9.9g)\n"
    "  sugar         ← 糖質 OR うち糖類 row — null if absent\n"
    "  fibre         ← 食物繊維    row ONLY  — null if absent\n"
    "  salt          ← 食塩相当量  row ONLY  (e.g. 0.21g)\n"
    "                  DO NOT use ナトリウム or any other row for this field.\n"
    "\n"
    "NULL vs ZERO rules (critical):\n"
    "  - null  = the row does NOT appear anywhere in the nutrition table\n"
    "  - '0g'  = the row IS printed and its value is zero\n"
    "  - Never guess or infer a value; null when absent\n"
    "\n"
    "Extra rules:\n"
    "  - basis: copy the serving line exactly as printed (e.g. 'コップ1杯(200ml)当たり')\n"
    "  - If two columns exist (per piece + per 100g), use the per-piece/per-serving column\n"
    "  - Copy values with units (137kcal, 7.8g, 0.21g) — Stage 2 strips units\n"
    "Output ONLY the JSON. No markdown, no prose."
)

SYSTEM_TRANSLATE = (
    "You are a Japanese food label translation specialist.\n"
    "Translate accurately using standard food industry English terms.\n"
    "Output ONLY valid JSON. No markdown, no prose."
)

# Stage 2: translation
PROMPT_TRANSLATE_TEMPLATE = (
    "Translate this Japanese food label JSON to English. Return ONLY valid JSON, no prose.\n"
    "\n"
    "INPUT:\n"
    "{ocr_json}\n"
    "\n"
    'OUTPUT — {"english":{...}} with these fields translated:\n'
    "- product_name: English name\n"
    "- ingredients.items: translate each item; keep (allergen) annotation in English\n"
    "- allergens.style: copy unchanged\n"
    "- allergens.items: translate each allergen name to English\n"
    "- nutrition.basis: translate serving note "
    "(コップ1杯(200ml)当たり→per cup (200ml), 1袋(51g)当たり→per bag (51g), 100gあたり→per 100g)\n"
    "- nutrition values: number only, no units (137kcal→137, 7.8g→7.8, 0.21g→0.21)\n"
    "- nutrition.sugar: strip うち prefix, number only (うち28.0g→28.0)\n"
    "- nutrition.saturated_fat: translate 飽和脂肪酸 row value ONLY.\n"
    "  If input saturated_fat is null, output null. "
    "  NEVER substitute the salt value here.\n"
    "- nutrition.salt: translate 食塩相当量 row value ONLY.\n"
    "  If input salt is null, output null. "
    "  NEVER use ナトリウム or any other field for salt.\n"
    "- nutrition.fibre: translate 食物繊維 row value ONLY — null if input is null\n"
    "- 0 means the label printed zero; null means the row was absent — preserve this distinction\n"
    "- 乳脂肪分 / 無脂乳固形分 are composition % specs — set fat to null if only % present\n"
    "- ingredients: ingredient names only — NO slogans or handling text\n"
    "\n"
    "Allergens: 卵=egg 乳=milk 小麦=wheat そば=buckwheat 落花生=peanut えび=shrimp かに=crab "
    "くるみ=walnut アーモンド=almond カシューナッツ=cashew nut キウイフルーツ=kiwi fruit "
    "牛肉=beef 豚肉=pork 鶏肉=chicken さけ=salmon さば=mackerel いか=squid "
    "あわび=abalone いくら=salmon roe オレンジ=orange バナナ=banana もも=peach "
    "りんご=apple ゼラチン=gelatin 大豆=soy ごま=sesame まつたけ=matsutake "
    "やまいも=yam マカダミアナッツ=macadamia nut\n"
    "Food: 水あめ=starch syrup 砂糖=sugar でん粉=starch 植物油脂=vegetable oil "
    "濃縮もも果汁=concentrated peach juice 濃縮果汁=concentrated juice "
    "酸味料=acidulant ゲル化剤=gelling agent ペクチン=pectin "
    "光沢剤=glazing agent 香料=flavoring 乳化剤=emulsifier "
    "増粘剤=thickener 着色料=food coloring 保存料=preservative "
    "甘味料=sweetener 酸化防止剤=antioxidant 調味料=seasoning "
    "膨張剤=leavening agent 加工でん粉=modified starch "
    "果糖ぶどう糖液糖=high-fructose corn syrup "
    "ベニバナ色素=safflower pigment 紅花色素=safflower pigment "
    "カロテン=carotene アントシアニン=anthocyanin "
    "ビタミンC=vitamin C もち米=glutinous rice 味噌=miso 醤油=soy sauce "
    "みりん=mirin 酢=vinegar 食塩=salt 食用油=edible oil チキン=chicken "
    "でん粉糖=starch sugar 乳糖=lactose "
    "食塩相当量=salt エネルギー=calories たんぱく質=protein 脂質=fat "
    "炭水化物=carbohydrate 糖質=sugar 食物繊維=fibre\n"
)


def _safe_parse(raw: str):
    """Parse JSON from model output, tolerating code fences and unclosed braces.
    Returns (parsed, error)."""
    if not raw or not raw.strip():
        return None, "empty model output"
    candidates = []
    for m in re.findall(r"```(?:json)?\s*([\s\S]+?)```", raw):
        candidates.append(m.strip())
    candidates.append(raw.strip())
    m2 = re.search(r"(\{[\s\S]+\})", raw)
    if m2:
        candidates.append(m2.group(1))
    for c in list(candidates):
        diff = c.count("{") - c.count("}")
        if diff > 0:
            candidates.append(c + "}" * diff)
    err = "no valid JSON found"
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        try:
            return json.loads(c), None
        except json.JSONDecodeError as e:
            err = str(e)
    return None, err


def _free_cache():
    """Run garbage collection and release cached CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _resize_for_ocr(image: Image.Image, max_size: int = _OCR_MAX_SIZE) -> Image.Image:
    """Downscale so the longest side is at most max_size, keeping aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _generate(messages: list, max_new_tokens: int = 512) -> tuple[str, float]:
    """Run greedy generation on chat messages; on CUDA OOM retry once at half the pixel budget.
    Returns (text, latency_s)."""
    from qwen_vl_utils import process_vision_info

    assert _processor is not None
    assert _model     is not None
    assert _device    is not None

    def _run():
        pvi = process_vision_info(messages)
        image_inputs, video_inputs = pvi[0], pvi[1]

        text_in = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _processor(
            text=[text_in],
            images=image_inputs  if image_inputs  else None,
            videos=video_inputs  if video_inputs  else None,
            padding=True,
            return_tensors="pt",
        ).to(_device)

        t0 = time.time()
        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.05,
            )
        latency = round(time.time() - t0, 2)

        input_len = inputs["input_ids"].shape[1]
        sequences = out.sequences if hasattr(out, "sequences") else out
        raw = _processor.batch_decode(
            sequences[:, input_len:], skip_special_tokens=True
        )[0].strip()

        del inputs, out
        _free_cache()
        return raw, latency

    try:
        return _run()
    except torch.OutOfMemoryError:
        _free_cache()

    orig_max = _processor.image_processor.max_pixels
    try:
        _processor.image_processor.max_pixels = orig_max // 2
        result = _run()
        return result
    except torch.OutOfMemoryError as e:
        raise RuntimeError(
            f"CUDA OOM even at reduced pixel budget ({orig_max // 2} px). "
            f"Original error: {e}"
        ) from e
    finally:
        _processor.image_processor.max_pixels = orig_max


def _smart_split(text: str) -> list[str]:
    """Split on the Japanese comma 、 only outside brackets."""
    parts, current, depth = [], [], 0
    for ch in text:
        if ch in "（(":   depth += 1
        elif ch in "）)": depth = max(0, depth - 1)
        if ch == "、" and depth == 0:
            tok = "".join(current).strip()
            if tok:
                parts.append(tok)
            current = []
        else:
            current.append(ch)
    tok = "".join(current).strip()
    if tok:
        parts.append(tok)
    return parts


_NOT_A_LABEL = {
    "japanese":    None,
    "english":     None,
    "ingredients": "",
    "latency_s":   0,
    "error":       "not_a_label",
    "s1a_raw":     "",
    "s1b_raw":     "",
    "s2_raw":      "",
}


def extract_label(image_path: str) -> dict:
    """Run OCR and translation on a label image.
    Returns Japanese and English fields, flat ingredients, latency, raw outputs and any error."""
    _load_model()
    _free_cache()

    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {**_NOT_A_LABEL, "error": f"Cannot open image: {e}"}

    ocr_image = _resize_for_ocr(pil_image)

    # stage 1a: ingredients and allergens
    try:
        s1a_raw, s1a_lat = _generate(
            [
                {"role": "system", "content": SYSTEM_OCR},
                {"role": "user", "content": [
                    {"type": "image", "image": ocr_image},
                    {"type": "text",  "text": PROMPT_INGREDIENTS},
                ]},
            ],
            max_new_tokens=_MAX_TOKENS_S1,
        )
    except RuntimeError as e:
        return {**_NOT_A_LABEL, "error": str(e), "s1a_raw": "", "s1b_raw": "", "s2_raw": ""}

    ingr_parsed, ingr_err = _safe_parse(s1a_raw)

    if isinstance(ingr_parsed, dict) and ingr_parsed.get("error") == "not_a_label":
        return _NOT_A_LABEL.copy()

    if ingr_parsed is None:
        return {
            **_NOT_A_LABEL,
            "latency_s": s1a_lat,
            "error": f"Ingredients OCR parse failed: {ingr_err}",
            "s1a_raw": s1a_raw,
        }

    _free_cache()

    # stage 1b: nutrition
    s1b_raw, s1b_lat, nut_parsed = "", 0.0, {}
    try:
        s1b_raw, s1b_lat = _generate(
            [
                {"role": "system", "content": SYSTEM_OCR},
                {"role": "user", "content": [
                    {"type": "image", "image": ocr_image},
                    {"type": "text",  "text": PROMPT_NUTRITION},
                ]},
            ],
            max_new_tokens=_MAX_TOKENS_S1,
        )
        nut_result, _ = _safe_parse(s1b_raw)
        if isinstance(nut_result, dict):
            nut_parsed = nut_result.get("nutrition") or nut_result
    except RuntimeError:
        nut_parsed = {}

    _free_cache()

    jp_parsed: dict = {
        "product_name": ingr_parsed.get("product_name"),
        "ingredients":  ingr_parsed.get("ingredients") or {},
        "allergens":    ingr_parsed.get("allergens")   or {},
        "nutrition":    nut_parsed if isinstance(nut_parsed, dict) else {},
    }

    # stage 2: translate
    ocr_str = json.dumps(jp_parsed, ensure_ascii=False, indent=2)
    prompt  = PROMPT_TRANSLATE_TEMPLATE.replace("{ocr_json}", ocr_str)
    try:
        s2_raw, s2_lat = _generate(
            [
                {"role": "system", "content": SYSTEM_TRANSLATE},
                {"role": "user",   "content": [{"type": "text", "text": prompt}]},
            ],
            max_new_tokens=_MAX_TOKENS_S2,
        )
    except RuntimeError as e:
        return {
            "japanese": jp_parsed, "english": None, "ingredients": "",
            "latency_s": round(s1a_lat + s1b_lat, 2),
            "error": f"Translation OOM: {e}",
            "s1a_raw": s1a_raw, "s1b_raw": s1b_raw, "s2_raw": "",
        }

    en_parsed, en_err = _safe_parse(s2_raw)
    english = (en_parsed or {}).get("english") or en_parsed or {}

    # post-processing
    en_alg = english.get("allergens") or {}
    if isinstance(en_alg, dict):
        en_alg["items"] = [
            JP_ALLERGEN_MAP.get(str(i).strip(), str(i))
            for i in (en_alg.get("items") or [])
        ]
        english["allergens"] = en_alg
    jp_alg_items  = (jp_parsed.get("allergens") or {}).get("items") or []
    translated_jp = [JP_ALLERGEN_MAP.get(str(i).strip(), str(i)) for i in jp_alg_items]

    raw_items = english.get("ingredients") or []
    if isinstance(raw_items, list):
        split_items = [t for item in raw_items for t in _smart_split(str(item))]
    elif isinstance(raw_items, dict):
        split_items = [
            t for item in (raw_items.get("items") or [])
            for t in _smart_split(str(item))
        ]
    else:
        split_items = _smart_split(str(raw_items))
    ingredients_flat = ", ".join(split_items)

    def _is_english(s: str) -> bool:
        return bool(str(s).strip()) and all(ord(c) < 0x3000 for c in str(s))

    existing_lower = {t.strip().lower() for t in ingredients_flat.split(",") if t.strip()}
    new_terms = [
        t for t in dict.fromkeys((en_alg.get("items") or []) + translated_jp)
        if _is_english(t) and t.lower() not in existing_lower
    ]
    if new_terms:
        ingredients_flat = (ingredients_flat + ", " + ", ".join(new_terms)).strip(", ")

    def _fix_slash(token: str) -> str:
        if "/" not in token or re.search(r"#\d", token):
            return token
        parts = token.split("/", 1)
        return f"{parts[0].strip()} ({parts[1].strip()})"

    ingredients_flat = ", ".join(
        _fix_slash(t) for t in ingredients_flat.split(", ") if t.strip()
    )

    seen: set = set()
    deduped: list = []
    for tok in ingredients_flat.split(", "):
        key = tok.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(tok.strip())
    ingredients_flat = ", ".join(deduped)

    nut_en = english.get("nutrition") or {}
    for k, v in nut_en.items():
        if isinstance(v, str):
            nut_en[k] = re.sub(r"\s*\(.*?\)", "", v).strip()
    english["nutrition"] = nut_en

    jp_nut = jp_parsed.get("nutrition") or {}
    en_nut_has_data = any(
        v is not None and str(v).strip() not in ("", "null")
        for k, v in nut_en.items()
        if k != "basis"
    )
    if not en_nut_has_data and jp_nut:
        fallback: dict = {}
        for field in ["calories", "protein", "fat", "saturated_fat",
                      "carbohydrate", "sugar", "fibre", "salt"]:
            jp_val = jp_nut.get(field)
            if jp_val is None:
                fallback[field] = None
                continue
            s = str(jp_val).strip()
            s = re.sub(r"^うち\s*", "", s)
            s = re.sub(r"\s*\(.*?\)", "", s)
            m = re.search(r"-?\d+\.?\d*", s)
            fallback[field] = float(m.group()) if m else None
        fallback["basis"] = jp_nut.get("basis") or ""
        english["nutrition"] = fallback
        nut_en = fallback

    basis = nut_en.get("basis") or ""
    if basis and any(ord(c) > 0x3000 for c in str(basis)):
        b = re.sub(r"当[たり]+|当り|あたり", "", basis)
        b = re.sub(r"コップ\s*(\d+)\s*杯\s*\(?(\d+[mMlL]+)\)?", r"per cup (\2)", b)
        b = re.sub(r"(\d+)\s*袋",           r"\1 bag",     b)
        b = re.sub(r"(\d+)\s*本",           r"\1 piece",   b)
        b = re.sub(r"(\d+)\s*枚",           r"\1 slice",   b)
        b = re.sub(r"(\d+)\s*個",           r"\1 piece",   b)
        b = re.sub(r"(\d+)\s*粒",           r"\1 piece",   b)
        b = re.sub(r"(\d+)\s*食",           r"\1 serving", b)
        b = re.sub(r"標準\s*",              "",             b)
        b = re.sub(r"100\s*g",              "per 100g",    b)
        b = re.sub(r"[\u3000-\u9fff\uff00-\uffef]+", " ", b).strip()
        b = re.sub(r"\s+", " ", b).strip(" ·-")
        if b:
            nut_en["basis"] = b
            english["nutrition"] = nut_en

    for field in ["sugar", "saturated_fat", "fibre"]:
        v = nut_en.get(field)
        if isinstance(v, str) and "うち" in v:
            nut_en[field] = re.sub(r"^うち\s*", "", v).strip()

    # drop saturated_fat values that are physically implausible
    def _to_num(v) -> float | None:
        if v is None:
            return None
        try:
            m = re.search(r"\d+\.?\d*", str(v))
            return float(m.group()) if m else None
        except (AttributeError, ValueError, TypeError):
            return None

    fat_v  = _to_num(nut_en.get("fat"))
    satf_v = _to_num(nut_en.get("saturated_fat"))
    salt_v = _to_num(nut_en.get("salt"))

    if satf_v is not None:
        bad = any([
            # saturated fat cannot exceed total fat
            fat_v is not None and satf_v > fat_v,
            # fat is zero but saturated fat is positive — impossible
            fat_v == 0.0 and satf_v > 0,
            # saturated fat is implausibly tiny fraction of fat (< 1%)
            fat_v is not None and fat_v > 1.0 and satf_v < fat_v * 0.01,
            # equals salt: the model likely copied the salt row
            salt_v is not None and abs(satf_v - salt_v) < 0.001,
            # more than 5g saturated fat with under 1g total fat is impossible
            fat_v is not None and fat_v < 1.0 and satf_v > 5.0,
        ])
        if bad:
            nut_en["saturated_fat"] = None
            english["nutrition"] = nut_en

    return {
        "japanese":    jp_parsed,
        "english":     english,
        "ingredients": ingredients_flat,
        "latency_s":   round(s1a_lat + s1b_lat + s2_lat, 2),
        "error":       en_err if en_parsed is None else None,
        "s1a_raw":     s1a_raw,
        "s1b_raw":     s1b_raw,
        "s2_raw":      s2_raw,
    }