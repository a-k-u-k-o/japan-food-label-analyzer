"""Qwen2.5-VL-7B OCR pipeline for the business role: full-label OCR, a focused second pass and JSON extraction.
The model is loaded lazily on the first business upload and cached."""

import re
import gc
import json
import math
from typing import Optional

import torch
from PIL import Image
import streamlit as st
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# re-exported so pages.py can import it from here
from business.services.label_field_extractor import extract_label_fields  # noqa: F401


@st.cache_resource
def load_qwen_model():
    """Load Qwen2.5-VL-7B-Instruct with the best available quantization; cached across runs."""
    import os as _os

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Cache to Google Drive when running in Colab
    if _os.path.isdir("/content/drive/MyDrive"):
        cache_dir = "/content/drive/MyDrive/hf_cache"
        _os.makedirs(cache_dir, exist_ok=True)
        _os.environ["HF_HOME"] = cache_dir
        print(f"[Qwen OCR] HF_HOME → {cache_dir}")

    print(f"[Qwen OCR] Loading {model_id}...")
    model = None

    # Try 4-bit NF4 with Flash Attention 2 first (A100/H100)
    try:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        print("[Qwen OCR] Loaded: 4-bit NF4 + Flash Attention 2")
    except Exception as e:
        print(f"[Qwen OCR] Flash Attention failed ({e}), trying standard 4-bit...")

    # Then 4-bit NF4 without Flash Attention (T4 and similar)
    if model is None:
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, quantization_config=bnb, device_map="auto"
            )
            print("[Qwen OCR] Loaded: 4-bit NF4 (no Flash Attention)")
        except Exception as e:
            print(f"[Qwen OCR] 4-bit failed ({e}), falling back to float16...")

    # Last resort: float16 (CPU or any GPU, slow)
    if model is None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        print("[Qwen OCR] Loaded: float16 fallback")

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 256,
        max_pixels=1280 * 28 * 28,
    )
    model.eval()
    print("[Qwen OCR] Model ready.")
    return model, processor


def preprocess_label_image(image: Image.Image) -> Image.Image:
    """Upscale, deskew, denoise, CLAHE and sharpen a label image; returns the original on any error."""
    import cv2
    import numpy as np

    try:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        MIN_SIZE = 1200
        if max(h, w) < MIN_SIZE:
            scale = MIN_SIZE / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180,
                                threshold=100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    if -15 < angle < 15:
                        angles.append(angle)
            if angles:
                median_angle = float(np.median(angles))
                if abs(median_angle) > 0.4:
                    cx, cy = w // 2, h // 2
                    M = cv2.getRotationMatrix2D((cx, cy), median_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)

        img = cv2.fastNlMeansDenoisingColored(img, None, h=6, hColor=6,
                                              templateWindowSize=7,
                                              searchWindowSize=21)

        lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l       = clahe.apply(l)
        img     = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        img  = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

        MAX_FINAL = 1600
        h, w = img.shape[:2]
        if max(h, w) > MAX_FINAL:
            scale = MAX_FINAL / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    except Exception:
        return image


def run_ocr(image: Image.Image, model, processor) -> str:
    """Run full-label OCR with Qwen2.5-VL and return the cleaned text, grouped by section."""
    MAX_SIZE = 1024
    w, h = image.size
    if max(w, h) > MAX_SIZE:
        scale = MAX_SIZE / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prompt_text = (
        "You are a strict OCR assistant for Japanese food product labels. "
        "Extract ALL text from this label. Follow these rules exactly:\n\n"

        "RULES:\n"
        "- Extract ONLY text from the food label itself\n"
        "- Do NOT add explanations, translations, or comments\n"
        "- Preserve exact Japanese characters, numbers, and units\n"
        "- If text is rotated or mirrored, correct it before extracting\n"
        "- Extract both Japanese AND English text if present\n"
        "- Write 未記載 ONLY when a field is truly absent\n\n"

        "EXTRACT THESE SECTIONS IN ORDER:\n\n"

        "## 栄養成分表示\n"
        "FIRST LINE: Copy the serving size exactly. Write as: 基準量: [serving size]\n"
        "If no serving size visible, write: 基準量: 不明\n"
        "THEN copy every nutrient row as: [nutrient name] [value][unit]\n\n"

        "## 原材料名\n"
        "Copy the COMPLETE ingredients list exactly as printed.\n\n"

        "## アレルギー物質\n"
        "Look at the allergen grid: 卵  乳  小麦  そば  落花生  えび  かに\n"
        "ONLY mark allergens where ○ is clearly printed below the name.\n"
        "Output format: ○印: [only the allergens with ○]\n"
        "Ignore factory warning sentences — those go in ## 製造工場注意\n\n"

        "## 製造工場注意\n"
        "Copy any factory allergen warning text exactly.\n\n"

        "## 商品情報\n"
        "品名: [value]\n"
        "内容量: [value]\n"
        "賞味期限: [value]\n"
        "保存方法: [value]\n"
        "製造者: [company name only]\n"
        "販売者: [company name only]\n"
        "製造所: [factory address only]\n"
        "輸入者: [importer if shown, else skip]\n"
        "原産国: [country — skip for domestic Japanese]\n"
        "お問い合わせ: [phone number only]\n\n"

        "## マーケティング情報\n"
        "健康強調: [claim]\n"
        "品質表示: [claim]\n"
        "英語表示: [English text]\n"
        "警告文: [warning]\n"
        "Skip lines where value is absent.\n\n"

        "## リサイクルマーク\n"
        "Look at ALL corners for recycling symbols.\n"
        "Each on its OWN LINE: [part]: [material]\n"
        "e.g. 外箱: 紙 / 菓子袋: PP / ボトル: PET\n"
        "If none visible: write なし\n\n"

        "## 調理方法\n"
        "Copy cooking instructions if present.\n\n"

        "Output only the extracted text under each section header. Nothing else."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt_text},
        ]
    }]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    device = next(model.parameters()).device
    inputs = processor(
        text=[text_input], images=image_inputs,
        videos=video_inputs, padding=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.05,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return _clean_ocr_output(raw)


def focused_pass(image: Image.Image, existing_ocr: str,
                 model, processor) -> str:
    """Second pass for recycling marks and company names; skipped if the first pass already got both."""
    recycling_ok = bool(re.search(
        r"## リサイクルマーク\n(?!未記載)(.+)", existing_ocr
    ))
    COMPANY_KW = ["株式会社", "有限会社", "合同会社", "（株）", "(株)"]
    mfr_line = re.search(r"製造者[：:\s]+([^\n]+)", existing_ocr)
    company_ok = mfr_line and any(kw in mfr_line.group(1) for kw in COMPANY_KW)

    if recycling_ok and company_ok:
        return existing_ocr

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    MAX_SIZE = 1024
    w, h = image.size
    if max(w, h) > MAX_SIZE:
        scale = MAX_SIZE / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    parts = []
    if not recycling_ok:
        parts.append(
            "## リサイクルマーク\n"
            "Look carefully at ALL recycling symbols on this label.\n"
            "Output each on its OWN LINE as: [part]: [material]\n"
            "Examples: 外箱: 紙 / 菓子袋: PP / ボトル: PET\n"
            "If none visible: なし\n"
        )
    if not company_ok:
        parts.append(
            "## 製造者・販売者\n"
            "Find the company that MADE this product (製造者) and the company "
            "that SELLS it (販売者). They are often different.\n"
            "製造者: [exact name with 株式会社 etc]\n"
            "販売者: [exact name with 株式会社 etc]\n"
            "製造所: [factory address]\n"
            "Copy EXACTLY as printed.\n"
        )

    if not parts:
        return existing_ocr

    prompt = (
        "You are a strict Japanese OCR assistant. "
        "Read ONLY what is visibly printed. Copy EXACTLY.\n\n"
        + "\n".join(parts)
        + "\nOutput only the section text. Nothing else."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ]
    }]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    device = next(model.parameters()).device
    inputs = processor(
        text=[text_input], images=image_inputs,
        videos=video_inputs, padding=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=256,
            do_sample=False, use_cache=True, repetition_penalty=1.05,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    focused = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    if not focused:
        return existing_ocr

    if not recycling_ok and "リサイクルマーク" in focused:
        rm = re.search(r"## リサイクルマーク\n([\s\S]+?)(?=##|$)", focused)
        if rm and rm.group(1).strip() not in ("なし", "未記載", ""):
            if "## リサイクルマーク" in existing_ocr:
                existing_ocr = re.sub(
                    r"(## リサイクルマーク\n)[\s\S]*?(?=##|$)",
                    r"\1" + rm.group(1), existing_ocr,
                )
            else:
                existing_ocr += "\n\n## リサイクルマーク\n" + rm.group(1)

    if not company_ok:
        for field in ["製造者", "販売者", "製造所"]:
            fm = re.search(field + r"[：:\s]+([^\n]+)", focused)
            if fm:
                val = fm.group(1).strip()
                valid = (
                    any(kw in val for kw in COMPANY_KW)
                    if field in ("製造者", "販売者")
                    else re.search(r"[都道府県市区町村]", val)
                )
                if valid:
                    existing_ocr = re.sub(
                        field + r"[：:\s]+[^\n]+",
                        field + ": " + val, existing_ocr,
                    )
    return existing_ocr


_JSON_PROMPT_COMPANY = (
    "You are a strict Japanese food label OCR assistant. "
    "Your ONLY job is to read text that is PHYSICALLY PRINTED on the label in this image.\n"
    "Output ONLY valid JSON. No prose, no markdown fences.\n\n"
    "CRITICAL: If this image does NOT show a Japanese food label, output ALL fields as null.\n"
    "CRITICAL: Do NOT use your training memory. Do NOT copy from the example below.\n"
    "The example shows the FORMAT only — replace every value with what you actually see.\n\n"
    "FORMAT EXAMPLE (do not copy these values):\n"
    "{\n"
    '  "product_name": null,\n'
    '  "flavor": null,\n'
    '  "variant": null,\n'
    '  "net_weight": null,\n'
    '  "expiry": null,\n'
    '  "storage": null,\n'
    '  "manufacturer": null,\n'
    '  "seller": null,\n'
    '  "factory": null,\n'
    '  "importer": null,\n'
    '  "origin_country": null,\n'
    '  "contact": null,\n'
    '  "age_restriction": null,\n'
    '  "trademark": null\n'
    "}\n\n"
    "RULES:\n"
    "- manufacturer = company that MADE it (製造者) — must be visible on label\n"
    "- seller = company that SELLS it (販売者) — must be visible on label\n"
    "- factory = factory address (製造所) — must contain prefecture/city text\n"
    "- origin_country: null for domestic Japanese products\n"
    "- Set ANY field to null if it is not clearly visible on THIS image\n"
    "- If the image is not a food label, return ALL fields as null\n"
    "- NEVER invent, guess, or copy from memory"
)

_JSON_PROMPT_RECYCLING = (
    "Find ALL recycling marks on this Japanese food label.\n"
    "Output ONLY valid JSON. No prose, no markdown fences.\n\n"
    '{"found": true, "marks": [{"part": "外袋", "material": "PP"}]}\n\n'
    "RULES:\n"
    "- Look at ALL corners — marks are often small\n"
    "- part: 外袋/外箱/菓子袋/ボトル/キャップ/個装袋\n"
    "- material: PP/PE/PS/PET/PVC/紙/缶/アルミ\n"
    "- ポリプロピレン=PP, ポリエチレン=PE\n"
    '- If no marks: {"found": false, "marks": []}'
)

_JSON_PROMPT_CLAIMS = (
    "Extract marketing claims and certifications from this Japanese label.\n"
    "Output ONLY valid JSON. No prose, no markdown fences.\n\n"
    '{"claims_jp":[],"claims_en":[],"warnings":[],"certifications":[],'
    '"spice_level":null,"cooking_instructions":null}\n\n'
    "RULES:\n"
    "- claims_jp: Japanese health/quality claims printed on THIS label\n"
    "- certifications: only if cert mark/text is clearly visible\n"
    "- spice_level: integer or null\n"
    "- Use [] for missing sections — never invent values"
)

_JSON_PROMPT_ALLERGENS = (
    "Extract allergen information from this Japanese food label.\n"
    "Output ONLY valid JSON. No prose, no markdown fences.\n\n"
    '{"found":true,"style":"grid","in_product":["milk"],'
    '"cross_contact":["egg","wheat"]}\n\n'
    "RULES:\n"
    "- GRID: 卵 乳 小麦 そば 落花生 えび かに — only ○ cells → in_product\n"
    "- Cross-contact warnings (shared equipment) → cross_contact only\n"
    "- TEXT: '本品に含まれるアレルゲン: 小麦・卵' → in_product, style=text\n"
    "- Names: 卵=egg 乳=milk 小麦=wheat そば=buckwheat 落花生=peanut "
    "えび=shrimp かに=crab\n"
    '- If none: {"found":false,"style":null,"in_product":[],"cross_contact":[]}'
)


# alias; pages.py imports either name
_focused_pass = focused_pass


def _safe_json_parse(raw: str) -> Optional[dict]:
    """Parse JSON from model output, tolerating code fences and missing closing braces."""
    if not raw:
        return None
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
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        try:
            return json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _run_single_json_ocr(image: Image.Image, prompt: str,
                          model, processor,
                          max_tokens: int = 400) -> Optional[dict]:
    """Run one JSON-prompt OCR pass and return the parsed dict, or None."""
    MAX_SIZE = 1024
    w, h = image.size
    if max(w, h) > MAX_SIZE:
        scale = MAX_SIZE / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ]
    }]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    device = next(model.parameters()).device
    inputs = processor(
        text=[text_input], images=image_inputs,
        videos=video_inputs, padding=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, use_cache=True, repetition_penalty=1.05,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return _safe_json_parse(_clean_ocr_output(raw))


def run_json_ocr(image: Image.Image, model, processor) -> dict:
    """Run the four JSON OCR prompts one at a time (to save VRAM); keys: company, recycling, claims, allergens."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "company":   _run_single_json_ocr(
            image, _JSON_PROMPT_COMPANY,   model, processor, max_tokens=350),
        "recycling": _run_single_json_ocr(
            image, _JSON_PROMPT_RECYCLING, model, processor, max_tokens=200),
        "claims":    _run_single_json_ocr(
            image, _JSON_PROMPT_CLAIMS,    model, processor, max_tokens=300),
        "allergens": _run_single_json_ocr(
            image, _JSON_PROMPT_ALLERGENS, model, processor, max_tokens=200),
    }


def _clean_ocr_output(text: str) -> str:
    """Remove model artifacts from raw OCR output."""
    if not text:
        return text
    text = re.sub(r"\s*#{1,6}\s*$",   "",    text, flags=re.MULTILINE)
    text = re.sub(r"\s*#{1,6}(\s|$)", " ",   text)
    text = re.sub(r"\s+$",            "",    text, flags=re.MULTILINE)
    text = re.sub(r"[、。]{3,}",       "、",  text)
    text = re.sub(r"[・]{3,}",         "・",  text)
    text = re.sub(r"\.{4,}",          "...", text)
    text = re.sub(r"\*{2,}",          "",    text)
    text = re.sub(r"`{1,3}",          "",    text)
    text = re.sub(r"\n{3,}",          "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def merge_json_into_fields(label_fields: dict, json_results: dict):
    """Merge JSON OCR results into label_fields in place; JSON values override text-OCR values."""
    COMPANY_KW = ["株式会社", "有限会社", "合同会社", "（株）", "(株)"]

    comp = json_results.get("company") or {}
    _FIELD_MAP = {
        "product_name":    "product_name",
        "flavor":          "flavor",
        "variant":         "variant",
        "net_weight":      "net_weight",
        "expiry":          "expiry_info",
        "storage":         "storage",
        "contact":         "contact",
        "age_restriction": "age_restriction",
        "trademark":       "trademark",
        "importer":        "importer",
        "origin_country":  "origin_country",
    }
    for json_key, field_key in _FIELD_MAP.items():
        v = comp.get(json_key)
        if v and str(v).strip() not in ("null", "None", ""):
            label_fields[field_key] = str(v).strip()

    mfr = comp.get("manufacturer")
    if mfr and any(kw in str(mfr) for kw in COMPANY_KW):
        label_fields["manufacturer"] = str(mfr).strip()
    elif mfr and str(mfr).strip() not in ("null", "None", ""):
        if not label_fields.get("manufacturer"):
            label_fields["manufacturer"] = str(mfr).strip()

    seller = comp.get("seller")
    if seller and any(kw in str(seller) for kw in COMPANY_KW):
        label_fields["seller"] = str(seller).strip()
    elif seller and str(seller).strip() not in ("null", "None", ""):
        if not label_fields.get("seller"):
            label_fields["seller"] = str(seller).strip()

    factory = comp.get("factory")
    if factory and re.search(r'[都道府県市区町村\d]', str(factory)):
        label_fields["factory"] = str(factory).strip()

    claims  = json_results.get("claims") or {}
    jp      = claims.get("claims_jp", [])
    en      = claims.get("claims_en", [])
    warn    = claims.get("warnings", [])
    certs   = claims.get("certifications", [])
    cooking = claims.get("cooking_instructions")
    spice   = claims.get("spice_level")

    if jp:
        existing = set(label_fields.get("claims_jp") or [])
        for c in jp:
            if c and c not in existing:
                existing.add(c)
                label_fields.setdefault("claims_jp", []).append(c)
    if en:
        existing = set(label_fields.get("claims_en") or [])
        for c in en:
            if c and c not in existing:
                existing.add(c)
                label_fields.setdefault("claims_en", []).append(c)
    if warn:
        existing_warns = list(label_fields.get("warnings") or [])
        for w in warn:
            if not w:
                continue
            w_stripped = w.strip()
            if w_stripped in existing_warns:
                continue
            if any(w_stripped in ex for ex in existing_warns):
                continue
            shorter = [ex for ex in existing_warns if ex in w_stripped]
            for s in shorter:
                existing_warns.remove(s)
            existing_warns.append(w_stripped)
        label_fields["warnings"] = existing_warns
    if certs:
        label_fields.setdefault("certifications_json", []).extend(certs)
    if cooking and not label_fields.get("cooking_instructions"):
        label_fields["cooking_instructions"] = cooking
    if spice is not None and not label_fields.get("spice_level"):
        label_fields["spice_level"] = spice

    alg = json_results.get("allergens") or {}
    if alg.get("found") and alg.get("style"):
        style_map = {
            "grid": "特定原材料7品目",
            "text": "特定原材料等",
            "box":  "アレルゲン強調表示",
        }
        label_fields["allergen_declaration_style"] = style_map.get(
            alg["style"], alg["style"]
        )

    cross = alg.get("cross_contact", []) or []
    if cross:
        cross_str = "共通設備使用: " + ", ".join(cross)
        existing_warns = label_fields.get("warnings") or []
        if cross_str not in existing_warns:
            label_fields.setdefault("warnings", []).append(cross_str)


def _validate_certifications(label_fields: dict):
    """Strip hallucinated certifications matching known few-shot example values."""
    FAKE_PATTERNS = ["C081641", "1165364", "<copy"]
    certs = label_fields.get("certifications_json") or []
    label_fields["certifications_json"] = [
        c for c in certs
        if not any(fp in str(c) for fp in FAKE_PATTERNS)
    ]
    tm = label_fields.get("trademark")
    if tm and any(fp in str(tm) for fp in FAKE_PATTERNS):
        label_fields["trademark"] = None


def _validate_company_fields(label_fields: dict):
    """Null out company fields holding old few-shot example values or null-like strings."""
    # values from the old few-shot example; if the model outputs these it copied them from memory
    HALLUCINATED_VALUES = {
        "株式会社東ハト",
        "株式会社バンダイ",
        "埼玉県入間郡三芳町上富1332",
        "0120-40-3102",
        "この箱の底面に記載",
        "直射日光・高温多湿をおさけください",
        "スナック菓子(チョコビ)",
        "チョコレート味",
        "25g",
    }
    # strings the model outputs instead of JSON null
    NULL_STRINGS = {
        "null", "NULL", "None", "none", "未記載", "不明",
        "N/A", "n/a", "-", "—", "", "なし",
    }

    COMPANY_FIELDS = [
        "manufacturer", "seller", "factory", "contact",
        "product_name", "net_weight", "expiry", "storage",
        "importer", "origin_country", "trademark", "age_restriction",
    ]

    for field in COMPANY_FIELDS:
        val = label_fields.get(field)
        if val is None:
            continue
        val_str = str(val).strip()
        if val_str in HALLUCINATED_VALUES:
            label_fields[field] = None
        elif val_str in NULL_STRINGS:
            label_fields[field] = None


def merge_json_recycling(symbol_data: dict, json_results: dict):
    """Add recycling marks from JSON OCR to symbol_data, accepting only known parts and materials."""
    rec = json_results.get("recycling") or {}
    if not rec.get("found") or not rec.get("marks"):
        return

    from business.services.symbol_detector import RECYCLING_TEXT_MAP

    VALID_PARTS_JP = {
        "外袋", "内袋", "個装袋", "外箱", "菓子袋",
        "ボトル", "キャップ", "ラベル", "カップ", "ふた",
        "箱", "容器", "トレー",
    }
    VALID_MATS = {
        "PP", "PE", "PS", "PET", "PVC", "LDPE", "HDPE",
        "プラ", "紙", "缶", "アルミ", "スチール", "びん",
        "ポリプロピレン", "ポリエチレン", "ポリスチレン",
        "塩化ビニル", "ペット",
    }

    existing_sigs = set()
    for r in symbol_data.get("recycling", []):
        existing_sigs.add(f"{r.get('type', '')}|{r.get('part', '')}")

    for mark in rec.get("marks", []):
        part = (mark.get("part") or "").strip()
        mat  = (mark.get("material") or "").strip()

        if not mat or mat not in VALID_MATS:
            continue
        if part and part not in VALID_PARTS_JP:
            continue

        mat_norm = {
            "ポリプロピレン": "PP", "ポリエチレン": "PE",
            "ポリスチレン":   "PS", "塩化ビニル":   "PVC",
            "ペット":         "PET",
        }.get(mat, mat)

        mat_en  = RECYCLING_TEXT_MAP.get(mat_norm, mat_norm)
        part_en = RECYCLING_TEXT_MAP.get(part, part) if part else None

        sig = f"{mat_en}|{part_en or ''}"
        if sig not in existing_sigs:
            existing_sigs.add(sig)
            symbol_data["recycling"].append({
                "type":       mat_en,
                "part":       part_en,
                "raw_text":   f"{part}:{mat}" if part else mat,
                "confidence": "high",
                "source":     "json_ocr",
            })