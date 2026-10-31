"""Detects symbols on Japanese food labels (recycling marks, certifications, warnings, trademarks).
Combines an OCR text scan, OpenCV candidate regions and Qwen identification of each crop."""

import cv2
import numpy as np
import os
import re
import json
import tempfile
from PIL import Image
from typing import Optional


# recycling material/part text -> English label
RECYCLING_TEXT_MAP = {
    "プラ": "Plastic (プラ)", "PP": "Polypropylene (PP)",
    "ポリプロピレン": "Polypropylene (PP)",  # full name alias
    "PE": "Polyethylene (PE)", "ポリエチレン": "Polyethylene (PE)",
    "PS": "Polystyrene (PS)", "ポリスチレン": "Polystyrene (PS)",
    "PET": "PET", "ペット": "PET",
    "PVC": "PVC", "塩化ビニル": "PVC",
    "LDPE": "LDPE", "HDPE": "HDPE",
    "紙": "Paper (紙)", "缶": "Can (缶)",
    "アルミ": "Aluminium", "スチール": "Steel", "びん": "Glass Bottle (びん)",
    "外袋": "Outer bag", "内袋": "Inner bag", "個装袋": "Individual wrapper",
    "外箱": "Outer box", "菓子袋": "Snack bag", "ボトル": "Bottle",
    "キャップ": "Cap", "ラベル": "Label", "カップ": "Cup",
    "ふた": "Lid", "箱": "Box", "容器": "Container",
    "トレー": "Tray",
}


def _preprocess_for_symbols(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to an inverted adaptive-threshold binary image for contour finding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=11, C=2
    )
    return thresh


def _find_symbol_candidates(img: np.ndarray) -> list:
    """Find candidate symbol regions by contour (small, roughly square or round); at most 15."""
    h, w = img.shape[:2]
    thresh = _preprocess_for_symbols(img)

    # connect broken contours in printed symbols
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    total_area = h * w

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        # skip noise and large text blocks
        if not (total_area * 0.0003 < area < total_area * 0.04):
            continue
        if not (20 < cw < 300 and 20 < ch < 300):
            continue

        # symbols are roughly square or round
        aspect = cw / ch if ch > 0 else 0
        if not (0.4 < aspect < 2.5):
            continue

        # recycling marks are rounded
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.25:
            continue

        # pad the crop for better Qwen recognition
        pad = max(5, int(min(cw, ch) * 0.15))
        x1 = max(0, x - pad);  y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad);  y2 = min(h, y + ch + pad)

        candidates.append({
            "crop":        img[y1:y2, x1:x2],
            "position":    (x1, y1, x2, y2),
            "area":        area,
            "circularity": round(circularity, 3),
        })

    # largest first; drop regions overlapping a kept one by more than 30% IoU
    candidates.sort(key=lambda c: c["area"], reverse=True)
    deduped = []
    for cand in candidates:
        x1, y1, x2, y2 = cand["position"]
        overlap = False
        for kept in deduped:
            kx1, ky1, kx2, ky2 = kept["position"]
            ix1 = max(x1, kx1); iy1 = max(y1, ky1)
            ix2 = min(x2, kx2); iy2 = min(y2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / (cand["area"] + kept["area"] - inter) > 0.3:
                    overlap = True
                    break
        if not overlap:
            deduped.append(cand)

    return deduped[:15]


def _identify_with_qwen(crop: np.ndarray, model, processor, device: str) -> Optional[dict]:
    """Send a symbol crop to the already-loaded Qwen model for identification."""
    import torch
    from qwen_vl_utils import process_vision_info

    tmp_path = os.path.join(tempfile.gettempdir(), f"sym_{id(crop)}.jpg")
    cv2.imwrite(tmp_path, crop)

    try:
        pil_img = Image.open(tmp_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": (
                    "You are a Japanese food packaging symbol identifier.\n"
                    "Look at this small image crop from a Japanese food label.\n\n"
                    "Identify if it contains any of these:\n"
                    "- Recycling marks: プラ, PP, PE, PS, PET, 紙, 缶, びん\n"
                    "- Part labels: 外袋, 内袋, 個装袋, 外箱, ボトル, キャップ\n"
                    "- Certifications: FSC, JAS有機, ハラール, YERLİ ÜRETİM\n"
                    "- Safety/Warning: P-マーク, ST安全マーク, 移り香注意\n"
                    "- Other: 商標登録, QRコード, バーコード\n\n"
                    "If NOT a symbol (text block, food photo, background), reply: null\n\n"
                    'Reply ONLY with JSON: {"symbol":"...","category":"...","part":"...","confidence":"high/medium/low"}\n'
                    "category: recycling | certification | warning | quality | barcode | unknown\n"
                    "part: packaging part if shown (外袋/外箱/etc) or null"
                )}
            ]
        }]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input], images=image_inputs,
            videos=video_inputs, padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=80,
                do_sample=False, use_cache=True,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        if raw.lower() == "null" or not raw:
            return None

        m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                if result.get("symbol"):
                    return result
            except json.JSONDecodeError:
                pass
        return None

    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _scan_text_for_symbols(ocr_text: str) -> dict:
    """Detect text-embedded symbols (recycling, certifications, warnings, quality, campaigns) in OCR text."""
    results = {
        "recycling": [], "certifications": [],
        "warnings": [], "quality": [], "campaigns": [],
    }
    if not ocr_text:
        return results

    text = ocr_text

    # The OCR prompt emits one "part: material" line per mark in the リサイクルマーク section
    # (e.g. 外箱: 紙); the legacy inline format (紙 外箱 プラ 菓子袋:PP,PET) is also handled.

    # longer tokens first to prevent partial matches (ポリプロピレン before PP, PET before PE)
    MAT_ORDER = [
        "ポリプロピレン", "ポリエチレン", "ポリスチレン", "塩化ビニル", "ペット",
        "LDPE", "HDPE", "PET", "PVC", "PP", "PE", "PS",
        "プラ", "紙", "缶", "アルミ", "スチール", "びん"
    ]

    PART_WORDS = (
        "外袋|内袋|個装袋|外箱|菓子袋|ボトル|キャップ|ラベル|カップ|ふた|箱|容器|トレー"
    )

    def _split_materials(raw_mat: str) -> list:
        """Split 'PP,PET' → ['PP','PET'] using ordered matching to avoid PE←PET."""
        tokens = []
        remaining = raw_mat.strip()
        while remaining:
            remaining = remaining.lstrip(",、・/ \t")
            if not remaining:
                break
            matched = False
            for mat in MAT_ORDER:
                if remaining.startswith(mat):
                    after = remaining[len(mat):]
                    if not after or not after[0].isalpha():
                        tokens.append(mat)
                        remaining = after
                        matched = True
                        break
            if not matched:
                sep_m = re.search(r'[,、・/\s]', remaining)
                if sep_m:
                    tokens.append(remaining[:sep_m.start()])
                    remaining = remaining[sep_m.start():]
                else:
                    tokens.append(remaining)
                    break
        return [t for t in tokens if t]

    def _add_recycling(mat_tok: str, part: str, raw: str):
        """Add a recycling entry if not already present."""
        mat_en = RECYCLING_TEXT_MAP.get(mat_tok, mat_tok)
        part_en = RECYCLING_TEXT_MAP.get(part, part) if part else None
        sig = f"{mat_tok}|{part}"
        if sig not in [f"{r.get('_mat','')  }|{r.get('_part','')}" for r in results["recycling"]]:
            results["recycling"].append({
                "type": mat_en, "part": part_en,
                "raw_text": raw, "confidence": "high", "source": "ocr_text",
                "_mat": mat_tok, "_part": part or "",
            })

    # Pattern 1 (primary): line-by-line format from the OCR section
    #   外箱: 紙  /  菓子袋: PP  /  外袋: PP, PET  /  外袋: ポリプロピレン
    _MAT_PAT = (
        r'ポリプロピレン|ポリエチレン|ポリスチレン|塩化ビニル|ペット'
        r'|LDPE|HDPE|PET|PVC|PP|PE|PS|プラ|紙|缶|アルミ|スチール|びん'
    )
    pat_line = re.compile(
        r'^(' + PART_WORDS + r')[：:\s]+'
        r'((?:' + _MAT_PAT + r')'
        r'(?:[,、・\s]+(?:' + _MAT_PAT + r'))*)',
        re.MULTILINE
    )
    for m in pat_line.finditer(text):
        part = m.group(1)
        for tok in _split_materials(m.group(2)):
            _add_recycling(tok, part, f"{part}:{tok}")

    # Pattern 1b: なし means no recycling marks; nothing to add
    if re.search(r'^(?:外袋|菓子袋|外箱)[：:\s]*なし', text, re.MULTILINE):
        pass

    # Pattern 2: 紙 外箱  (paper + part — space separated, no colon, same line)
    pat_paper = re.compile(r'紙[ \t]*(' + PART_WORDS + r')')
    for m in pat_paper.finditer(text):
        _add_recycling("紙", m.group(1), m.group(0).strip())

    # Pattern 3: プラ 外袋  (plastic + part — space separated, same line)
    pat_pla = re.compile(r'プラ[ \t]*(' + PART_WORDS + r')')
    for m in pat_pla.finditer(text):
        _add_recycling("プラ", m.group(1), m.group(0).strip())

    # Pattern 4: Fallback standalone symbols
    if not results["recycling"]:
        if re.search(r'プラ', text):
            results["recycling"].append({
                "type": "Plastic (プラ)", "part": None,
                "raw_text": "プラ", "confidence": "medium", "source": "ocr_text",
                "_mat": "プラ", "_part": "",
            })
        if re.search(r'(?<![^\s、。\n])(紙)(?=\s|$|[、。\n])', text):
            results["recycling"].append({
                "type": "Paper (紙)", "part": None,
                "raw_text": "紙", "confidence": "medium", "source": "ocr_text",
                "_mat": "紙", "_part": "",
            })

    # drop the internal dedup keys
    for r in results["recycling"]:
        r.pop("_mat", None)
        r.pop("_part", None)

    # FSC: require the full code (letter + 5+ digits, e.g. C081641); bare "FSC" is too ambiguous
    fsc_m = re.search(r'FSC[®©]?\s*([A-Z]\d{5,})', text)
    if fsc_m:
        results["certifications"].append({
            "type": "FSC® Forest Stewardship Council",
            "code": fsc_m.group(1),
            "raw_text": fsc_m.group(0),
            "confidence": "high", "source": "ocr_text",
        })

    if re.search(r'YERL[İI]|YERLI', text, re.IGNORECASE):
        results["certifications"].append({
            "type": "YERLİ ÜRETİM (Turkish Local Production)",
            "confidence": "high", "source": "ocr_text",
        })

    if re.search(r'販売社ライセンス|玩具安全|STマーク|Pマーク|米認許', text):
        results["certifications"].append({
            "type": "P-Mark / ST Safety Mark (Toy Safety)",
            "confidence": "medium", "source": "ocr_text",
        })

    if re.search(r'有機JAS|JAS有機|有機認証', text):
        results["certifications"].append({
            "type": "JAS Organic Certified (有機JAS)",
            "confidence": "high", "source": "ocr_text",
        })

    if re.search(r'移り香注意|移り香', text):
        results["warnings"].append({
            "type": "Odor Transfer Warning (移り香注意)",
            "confidence": "high", "source": "ocr_text",
        })

    if re.search(r'ごみ.*区分|廃棄.*自治体|自治体.*区分', text):
        results["warnings"].append({
            "type": "Follow Local Waste Separation Guidelines",
            "confidence": "high", "source": "ocr_text",
        })

    # 商標登録: require a registration number (7+ digits; Japanese TM numbers are 7-10)
    # so stray 商標登録 text without a number doesn't match
    tm_m = re.search(r'商標登録\s*第?\s*(\d{7,})\s*号?', text)
    if tm_m:
        results["quality"].append({
            "type": "Registered Trademark (商標登録)",
            "number": tm_m.group(1),
            "raw_text": tm_m.group(0),
            "confidence": "high", "source": "ocr_text",
        })

    if "折りパケ" in text:
        results["campaigns"].append({
            "type": "折りパケ運動 (Fold & Compact Packaging Campaign)",
            "source": "ocr_text",
        })

    # Deduplicate each category
    for key in results:
        seen, deduped = [], []
        for item in results[key]:
            sig = item.get("type", "") + item.get("raw_text", "")
            if sig not in seen:
                seen.append(sig)
                deduped.append(item)
        results[key] = deduped

    return results


def detect_all_symbols(
    image_path: str,
    ocr_text: str = "",
    model=None,
    processor=None,
    device: str = "cpu",
    use_qwen: bool = True,
) -> dict:
    """Detect symbols using the OCR text scan, OpenCV regions and Qwen crops.
    Text-scan results take priority; Qwen only adds validated entries not already found."""
    # text scan always runs first (no GPU needed)
    results = _scan_text_for_symbols(ocr_text)
    results["raw_candidates"] = 0

    # allowed values for Qwen results
    VALID_RECYCLING_TYPES = {
        "PP", "PE", "PS", "PET", "PVC", "LDPE", "HDPE",
        "プラ", "紙", "缶", "アルミ", "スチール", "びん",
        "Polypropylene (PP)", "Polyethylene (PE)", "Polystyrene (PS)",
        "Plastic (プラ)", "Paper (紙)", "Can (缶)",
        "Aluminium", "Steel", "Glass Bottle (びん)",
    }
    VALID_PART_NAMES = {
        "Outer bag", "Inner bag", "Individual wrapper", "Outer box",
        "Snack bag", "Bottle", "Cap", "Label", "Cup", "Lid",
        "Box", "Container", "Tray",
        "外袋", "内袋", "個装袋", "外箱", "菓子袋", "ボトル",
        "キャップ", "ラベル", "カップ", "ふた", "箱", "容器", "トレー",
    }
    VALID_CATEGORIES = {"recycling", "certification", "warning", "quality", "barcode"}

    try:
        img = cv2.imread(image_path)
        if img is None:
            pil = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        if img is not None:
            candidates = _find_symbol_candidates(img)
            results["raw_candidates"] = len(candidates)

            if use_qwen and model is not None and processor is not None:
                # the text scan is more reliable for recycling; vision only fills gaps
                ocr_found_recycling = len(results["recycling"]) > 0

                for cand in candidates:
                    qr = _identify_with_qwen(cand["crop"], model, processor, device)
                    if not qr or not qr.get("symbol"):
                        continue

                    # only accept high confidence
                    if qr.get("confidence", "low").lower() != "high":
                        continue

                    category = qr.get("category", "unknown")
                    sym  = qr.get("symbol", "").strip()
                    part = (qr.get("part") or "").strip()

                    if category not in VALID_CATEGORIES:
                        continue

                    if category == "recycling":
                        if ocr_found_recycling:
                            continue
                        mat_valid = any(
                            valid.lower() in sym.lower()
                            for valid in VALID_RECYCLING_TYPES
                        )
                        if not mat_valid:
                            continue
                        if part and part not in VALID_PART_NAMES:
                            continue

                    entry = {
                        "type":       sym,
                        "part":       part if part else None,
                        "position":   cand["position"],
                        "confidence": "high",
                        "source":     "qwen_vision",
                    }

                    # skip if the text scan already found it
                    target   = results.get(category, [])
                    existing = [r.get("type", "").lower() for r in target]
                    if not any(sym.lower() in et for et in existing):
                        if category in results:
                            results[category].append(entry)

    except Exception as e:
        results["opencv_error"] = str(e)

    return results


def format_symbols_html(symbol_data: dict) -> str:
    """Return HTML for Streamlit (unsafe_allow_html=True): symbols grouped by category as pill badges."""
    sections = [
        ("recycling",      "♻️ Recycling",      "#e8f5e9", "#1b5e20", "#a5d6a7"),
        ("certifications", "🏅 Certifications", "#e3f2fd", "#0d47a1", "#90caf9"),
        ("warnings",       "⚠️ Warnings",       "#fff8e1", "#e65100", "#ffcc02"),
        ("quality",        "✅ Quality",         "#f3e5f5", "#4a148c", "#ce93d8"),
        ("campaigns",      "📦 Campaigns",      "#fce4ec", "#880e4f", "#f48fb1"),
    ]

    html = ""
    for key, title, bg, text_color, border in sections:
        items = symbol_data.get(key, [])
        if not items:
            continue
        html += f'<div style="margin-bottom:.6rem;"><span style="font-size:.75rem;font-weight:700;color:#374151;">{title}</span><br style="margin:.2rem 0;">'
        for item in items:
            label = item.get("type", "")
            part  = item.get("part")
            code  = item.get("code", "")
            num   = item.get("number", "")
            badge = label
            if part and part not in ("null", "None", "unknown"):
                badge += f" [{part}]"
            if code:
                badge += f" ({code})"
            if num:
                badge += f" #{num}"
            html += (
                f'<span style="display:inline-block;background:{bg};border:1px solid {border};'
                f'color:{text_color};font-size:.68rem;padding:.2rem .6rem;border-radius:20px;'
                f'margin:.15rem .1rem;font-family:IBM Plex Mono,monospace;">{badge}</span>'
            )
        html += "</div>"

    return html