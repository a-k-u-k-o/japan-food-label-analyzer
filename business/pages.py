"""Business-role pages: label upload with the OCR pipeline, and the market readiness results.
Entry point is route(), called from the root app.py."""

import re
import os
import tempfile

import torch
import streamlit as st
from PIL import Image

from business.services.ocr import (
    load_qwen_model,
    preprocess_label_image,
    run_ocr,
    run_json_ocr,
    extract_label_fields,
    merge_json_into_fields,
    merge_json_recycling,
    _validate_certifications,
    _validate_company_fields,
    _focused_pass,
)
from business.services.extractor    import extract_nutrition, nutrition_to_dict
from business.services.translator   import translate_ingredients
from business.services.classifier   import classify_health
from business.services.compliance   import run_compliance_check
from business.services.benchmarking import benchmark, detect_category
from business.services.suggestions  import generate_suggestions
from business.services.readiness    import compute_readiness
from business.services.symbol_detector import detect_all_symbols, format_symbols_html


def _render_nutrition_table(nutrition: dict, levels: dict, missing: list):
    """Render the nutrition facts table with a colour-coded level per nutrient."""
    display = [
        ("エネルギー / Calories",     "calories",      "kcal", True),
        ("たんぱく質 / Protein",       "protein",       "g",    False),
        ("脂質・脂肪 / Fat",           "fat",           "g",    False),
        ("炭水化物 / Carbohydrates",   "carbohydrates", "g",    False),
        ("糖類 / Sugars",             "sugar",         "g",    False),
        ("食物繊維 / Fiber",           "fiber",         "g",    False),
        ("食塩相当量 / Sodium",        "sodium",        "g",    False),
        ("カルシウム / Calcium",       "calcium",       "mg",   False),
        ("鉄 / Iron",                 "iron",          "mg",   False),
        ("亜鉛 / Zinc",               "zinc",          "mg",   False),
        ("マグネシウム / Magnesium",   "magnesium",     "mg",   False),
        ("カリウム / Potassium",       "potassium",     "mg",   False),
        ("ビタミンC / Vitamin C",      "vitamin_c",     "mg",   False),
        ("ビタミンD / Vitamin D",      "vitamin_d",     "μg",   False),
        ("ビタミンE / Vitamin E",      "vitamin_e",     "mg",   False),
    ]
    lc = {"low": "#27ae60", "medium": "#e67e22", "high": "#c0392b", "unknown": "#aaa"}
    rows = ""
    found_any = False
    for label, key, unit, is_cal in display:
        val = nutrition.get(key)
        if val is None:
            continue
        found_any = True
        lvl     = levels.get(key, "unknown")
        color   = lc.get(lvl, "#aaa")
        val_str = (
            f"{val:.0f}" if is_cal or unit in ("mg", "μg")
            else f"{val:.2f}".rstrip("0").rstrip(".")
        )
        rows += (
            f"<tr><td>{label}</td>"
            f'<td class="nutr-value" style="color:{color}">{val_str} {unit}</td>'
            f'<td><span style="font-size:.65rem;text-transform:uppercase;color:{color}">'
            f'{lvl if lvl != "unknown" else "—"}</span></td></tr>'
        )
    if not found_any:
        rows = '<tr><td colspan="3" style="color:#aaa;font-style:italic;">No nutrition data extracted</td></tr>'

    st.markdown(
        f'<table class="nutr-table"><thead><tr>'
        f'<th>Nutrient</th><th>Amount</th><th>Level</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )
    if missing:
        st.markdown(
            f'<div style="font-size:.75rem;color:#aaa;margin-top:.5rem;">'
            f'Not found: {", ".join(missing)}</div>',
            unsafe_allow_html=True,
        )


def _render_allergens(in_product: list):
    """Render detected allergens as red pills, or a 'None detected' note."""
    if not in_product:
        st.markdown(
            '<span style="color:#8a8070;font-size:.85rem;">None detected</span>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        "".join(
            f'<span style="display:inline-block;background:#fee2e2;color:#991b1b;'
            f'border-radius:6px;padding:2px 10px;margin:2px;font-size:.82rem;'
            f'font-weight:600;">{a}</span>'
            for a in in_product
        ),
        unsafe_allow_html=True,
    )


def _render_label_info(fields: dict):
    """Render extracted label fields as a grid, followed by claim and warning pills."""
    _EMPTY = {"なし", "null", "None", "未記載", "不明", "N/A", "-", "—", ""}

    def _valid(v):
        return v and str(v).strip() not in _EMPTY

    items = [
        ("Product Type",    fields.get("product_name")),
        ("Net Weight",      fields.get("net_weight")),
        ("Expiry",          fields.get("expiry_info")),
        ("Storage",         fields.get("storage")),
        ("Manufacturer",    fields.get("manufacturer")),
        ("Seller",          fields.get("seller")),
        ("Importer",        fields.get("importer")),
        ("Factory",         fields.get("factory")),
        ("Origin",          fields.get("origin_country")),
        ("Contact",         fields.get("contact")),
        ("Allergen Style",  fields.get("allergen_declaration_style")),
        ("Trademark",       fields.get("trademark")),
        ("Age Restriction", fields.get("age_restriction")),
        ("Spice Level",     f"辛さ {fields['spice_level']}" if fields.get("spice_level") else None),
    ]
    visible = [(k, v) for k, v in items if _valid(v)]
    if not visible:
        return

    html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem .8rem;margin-bottom:.6rem;">'
    for key, val in visible:
        val_disp = str(val)[:80] + ("…" if len(str(val)) > 80 else "")
        html += (
            f'<div style="border-bottom:1px solid #f0f0f0;padding:.3rem 0;">'
            f'<div style="font-size:.62rem;color:#9ca3af;text-transform:uppercase;'
            f'letter-spacing:.12em;font-family:monospace;">{key}</div>'
            f'<div style="font-size:.82rem;color:#111;">{val_disp}</div>'
            f'</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    claims_jp = fields.get("claims_jp", [])
    claims_en = fields.get("claims_en", [])
    warnings  = fields.get("warnings", [])

    if claims_jp or claims_en:
        st.markdown(
            '<div style="font-size:.6rem;letter-spacing:.2em;color:#6b7280;'
            'text-transform:uppercase;margin:.5rem 0 .2rem;">Claims</div>',
            unsafe_allow_html=True,
        )
        pills = "".join(
            f'<span style="background:#e0f2fe;color:#0369a1;border-radius:4px;'
            f'padding:2px 8px;margin:2px;font-size:.75rem;display:inline-block;">{c}</span>'
            for c in claims_jp
        )
        pills += "".join(
            f'<span style="background:#f0fdf4;color:#166534;border-radius:4px;'
            f'padding:2px 8px;margin:2px;font-size:.75rem;display:inline-block;">{c}</span>'
            for c in claims_en
        )
        st.markdown(pills, unsafe_allow_html=True)

    if warnings:
        st.markdown(
            '<div style="font-size:.6rem;letter-spacing:.2em;color:#6b7280;'
            'text-transform:uppercase;margin:.5rem 0 .2rem;">Warnings</div>',
            unsafe_allow_html=True,
        )
        pills = "".join(
            f'<span style="background:#fef3c7;color:#92400e;border-radius:4px;'
            f'padding:2px 8px;margin:2px;font-size:.75rem;display:inline-block;">{w}</span>'
            for w in warnings
        )
        st.markdown(pills, unsafe_allow_html=True)


def _render_readiness_card(readiness, grade_colors: dict):
    """Render the Japan market readiness header card and its score breakdown."""
    grade_color = grade_colors.get(readiness.letter_grade, "#6b7280")

    def _comp_bar(c):
        is_pending = "Run the benchmark" in c.status
        bar_color  = "#374151" if is_pending else grade_colors.get(c.grade, "#6b7280")
        bar_w      = int(c.score / c.max_score * 100)
        grade_disp = "⏳" if is_pending else f"{c.grade} {c.score}/{c.max_score}"
        opacity    = "0.35" if is_pending else "1"
        return (
            f'<div style="margin-bottom:.5rem;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:.72rem;color:#6b7280;margin-bottom:.2rem;">'
            f'<span>{c.name.split("(")[0].strip()}</span>'
            f'<span style="opacity:{opacity};color:{bar_color};">{grade_disp}</span>'
            f'</div>'
            f'<div style="background:#e2e8f0;border-radius:99px;height:6px;overflow:hidden;">'
            f'<div style="width:{bar_w}%;height:6px;border-radius:99px;'
            f'background:{bar_color};opacity:{opacity};"></div>'
            f'</div></div>'
        )

    bars_html    = "".join(_comp_bar(c) for c in readiness.components)
    actions_html = "".join(
        f'<div class="readiness-action">→ {a}</div>'
        for a in readiness.top_actions
    )

    grade_cls = f"readiness-{readiness.letter_grade}"

    st.markdown(
        f'<div class="readiness-card {grade_cls}">'
        f'<div class="readiness-grade">{readiness.letter_grade}</div>'
        f'<div class="readiness-body">'
        f'<div class="readiness-headline">{readiness.headline}</div>'
        f'<div class="readiness-sub">{readiness.sub_headline}</div>'
        f'<div class="readiness-bar-wrap">'
        f'<div class="readiness-bar" style="width:{readiness.overall_score}%;"></div>'
        f'</div>'
        f'<div style="font-size:.75rem;color:#64748b;margin-bottom:.5rem;">'
        f'Score: <b style="color:#0f172a;">{readiness.overall_score}/100</b></div>'
        f'<div class="readiness-actions">{actions_html}</div>'
        f'</div>'
        f'<div style="min-width:200px;">{bars_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.expander("📋 Score Breakdown — click to expand"):
        bd_cols = st.columns(3)
        for idx, comp in enumerate(readiness.components):
            g          = comp.grade
            gc         = grade_colors.get(g, "#6b7280")
            pct        = int(comp.score / comp.max_score * 100)
            is_pending = "Run the benchmark" in comp.status
            with bd_cols[idx % 3]:
                detail_rows = "".join(
                    f'<div style="font-size:.72rem;line-height:1.75;color:'
                    f'{"#059669" if l.startswith("✓") else "#dc2626" if l.startswith("✗") else "#d97706" if l.startswith("⚠") else "#6b7280"}'
                    f';">{l}</div>'
                    for l in comp.breakdown
                )
                st.markdown(
                    f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                    f'border-radius:10px;padding:1rem;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:.4rem;">'
                    f'<span style="font-size:.8rem;font-weight:600;">'
                    f'{comp.name.split("(")[0].strip()}</span>'
                    f'<span style="font-size:1.1rem;font-weight:700;color:{gc};">'
                    f'{"⏳" if is_pending else g}</span>'
                    f'</div>'
                    f'<div style="background:#e2e8f0;border-radius:99px;height:5px;'
                    f'overflow:hidden;margin-bottom:.4rem;">'
                    f'<div style="width:{pct}%;height:5px;border-radius:99px;'
                    f'background:{gc};opacity:{"0.35" if is_pending else "1"};"></div>'
                    f'</div>'
                    f'<div style="font-size:.72rem;color:#6b7280;margin-bottom:.4rem;'
                    f'line-height:1.5;">{comp.status}</div>'
                    f'{detail_rows}</div>',
                    unsafe_allow_html=True,
                )


def page_biz_upload(nav_fn):
    """Upload page: take a label image, run the analysis pipeline, then go to results."""
    nav_fn()

    st.markdown(
        '<div class="page-title">🇯🇵 Japan Market Readiness</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-sub">Upload a Japanese food product label to run '
        'compliance validation, competitive benchmarking, and improvement analysis.</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1.7], gap="large")

    with col_left:
        st.markdown(
            '<div style="font-size:.62rem;letter-spacing:.2em;text-transform:uppercase;'
            'color:#6b7280;margin-bottom:.5rem;">Upload Label Image</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            image       = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded label", use_container_width=True)
            analyze_btn = st.button(
                "Analyze Label →", type="primary", use_container_width=True
            )
        else:
            image       = None
            analyze_btn = False
            st.markdown("""
            <div style="background:#f8fafc;border:2px dashed #e2e8f0;border-radius:12px;
                 padding:2.5rem 1rem;text-align:center;">
              <div style="font-size:2rem;margin-bottom:.6rem;">🍱</div>
              <div style="font-size:.72rem;color:#9ca3af;letter-spacing:.15em;
                   text-transform:uppercase;">Upload a Japanese food label<br>to start analysis</div>
            </div>""", unsafe_allow_html=True)

    # analysis pipeline
    with col_right:
        if uploaded and analyze_btn and image:

            with st.spinner("Step 1 — Preprocessing image…"):
                processed_image = preprocess_label_image(image)

            with st.expander("🔍 Image Preprocessing Preview", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Original")
                    st.image(image, use_container_width=True)
                with c2:
                    st.caption("✓ Preprocessed")
                    st.image(processed_image, use_container_width=True)

            with st.spinner("Step 1 — Running Qwen2.5-VL-7B full label OCR…"):
                ocr_model, ocr_processor = load_qwen_model()
                raw_text = run_ocr(processed_image, ocr_model, ocr_processor)

            with st.spinner("Step 1b — Focused pass: recycling marks & company names…"):
                raw_text = _focused_pass(
                    processed_image, raw_text, ocr_model, ocr_processor
                )

            with st.spinner("Step 1c — JSON OCR: company · recycling · claims · allergens…"):
                json_ocr_results = run_json_ocr(
                    processed_image, ocr_model, ocr_processor
                )

            with st.spinner("Step 2 — Extracting nutrition data…"):
                nutrition_data = extract_nutrition(raw_text)
                nutrition_dict = nutrition_to_dict(nutrition_data)

            with st.spinner("Step 3 — Extracting label fields…"):
                label_fields = extract_label_fields(raw_text)
                merge_json_into_fields(label_fields, json_ocr_results)
                _validate_certifications(label_fields)
                _validate_company_fields(label_fields)

            with st.spinner("Step 4 — Detecting symbols & certifications…"):
                _device = "cuda" if torch.cuda.is_available() else "cpu"
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image.save(tmp.name)
                    _tmp_path = tmp.name
                try:
                    symbol_data = detect_all_symbols(
                        image_path=_tmp_path,
                        ocr_text=raw_text,
                        model=ocr_model,
                        processor=ocr_processor,
                        device=_device,
                        use_qwen=True,
                    )
                    merge_json_recycling(symbol_data, json_ocr_results)
                    FAKE_CERT_PATTERNS = ["C081641", "1165364", "<copy"]
                    for cert in (json_ocr_results.get("claims") or {}).get(
                            "certifications", []):
                        if not cert:
                            continue
                        if any(fp in str(cert) for fp in FAKE_CERT_PATTERNS):
                            continue
                        fsc_m = re.search(r'FSC\s*([A-Z]\d{5,})', cert)
                        entry = (
                            {"type": "FSC® Forest Stewardship Council",
                             "code": fsc_m.group(1),
                             "confidence": "high", "source": "json_ocr"}
                            if fsc_m else
                            {"type": cert, "confidence": "high", "source": "json_ocr"}
                        )
                        existing = [
                            c.get("type", "") for c in
                            symbol_data.get("certifications", [])
                        ]
                        if entry["type"] not in existing:
                            symbol_data.setdefault("certifications", []).append(entry)
                finally:
                    if os.path.exists(_tmp_path):
                        os.remove(_tmp_path)

            with st.spinner("Step 5 — Translating ingredients…"):
                translated = translate_ingredients(
                    nutrition_data.ingredients or ""
                )

            with st.spinner("Step 6 — Classifying nutrients…"):
                result = classify_health(nutrition_dict)

            with st.spinner("Step 7 — Checking regulatory compliance…"):
                nutrition_dict["raw_text"]    = raw_text
                nutrition_dict["ingredients"] = nutrition_data.ingredients or ""
                compliance_reports = run_compliance_check(nutrition_dict)

            with st.spinner("Step 7 — Detecting product category…"):
                category = detect_category(
                    nutrition_dict, nutrition_data.ingredients or ""
                )
                st.session_state["bench_category"] = category
                st.session_state["bench_nd"]       = nutrition_dict
                st.session_state.pop("bench_report",  None)
                st.session_state.pop("bench_ran_cat", None)

            with st.spinner("Step 7 — Generating improvement suggestions…"):
                _quick_bench = benchmark(nutrition_dict, category)
                suggestions  = generate_suggestions(
                    nutrition_dict, {}, compliance_reports
                )

            readiness = compute_readiness(
                compliance_reports, _quick_bench,
                suggestions, bench_confirmed=False,
            )

            st.session_state["biz_results"] = {
                "nutrition_dict":    nutrition_dict,
                "nutrition_data":    nutrition_data,
                "compliance_reports": compliance_reports,
                "suggestions":       suggestions,
                "readiness":         readiness,
                "result":            result,
                "translated":        translated,
                "raw_text":          raw_text,
                "category":          category,
                "label_fields":      label_fields,
                "symbol_data":       symbol_data,
                "json_ocr_results":  json_ocr_results,
            }
            st.session_state.page = "biz_results"
            st.rerun()

        elif not uploaded:
            st.markdown("""
            <div style="padding:3rem 1rem;text-align:center;">
              <div style="font-size:1rem;color:#9ca3af;line-height:2.4;margin-bottom:1.5rem;">
                Upload a Japanese food label<br>to start the analysis pipeline
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;
                   text-align:left;max-width:480px;margin:auto;">
                <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1rem;">
                  <div style="font-size:1.2rem;margin-bottom:.3rem;">🏷️</div>
                  <div style="font-size:.6rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.12em;">Compliance</div>
                  <div style="font-size:.78rem;margin-top:.2rem;">Japanese food law checks</div>
                </div>
                <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1rem;">
                  <div style="font-size:1.2rem;margin-bottom:.3rem;">📊</div>
                  <div style="font-size:.6rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.12em;">Benchmarking</div>
                  <div style="font-size:.78rem;margin-top:.2rem;">vs. category competitors</div>
                </div>
                <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1rem;">
                  <div style="font-size:1.2rem;margin-bottom:.3rem;">🎨</div>
                  <div style="font-size:.6rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.12em;">Suggestions</div>
                  <div style="font-size:.78rem;margin-top:.2rem;">Prioritised improvements</div>
                </div>
                <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1rem;">
                  <div style="font-size:1.2rem;margin-bottom:.3rem;">🇯🇵</div>
                  <div style="font-size:.6rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.12em;">Readiness Score</div>
                  <div style="font-size:.78rem;margin-top:.2rem;">A–F market entry grade</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)


def page_biz_results(nav_fn):
    """Results page: readiness card, label details, and the compliance, benchmark and suggestion tabs."""
    nav_fn()

    if "biz_results" not in st.session_state:
        st.warning("No analysis results yet. Please upload a label first.")
        if st.button("← Back to Upload"):
            st.session_state.page = "biz_upload"
            st.rerun()
        return

    _r               = st.session_state["biz_results"]
    nutrition_dict   = _r["nutrition_dict"]
    nutrition_data   = _r["nutrition_data"]
    compliance_reports = _r["compliance_reports"]
    suggestions      = _r["suggestions"]
    readiness        = _r["readiness"]
    result           = _r["result"]
    translated       = _r["translated"]
    raw_text         = _r["raw_text"]
    category         = _r["category"]
    label_fields     = _r.get("label_fields", {})
    symbol_data      = _r.get("symbol_data", {})
    json_ocr_results = _r.get("json_ocr_results", {})

    levels  = result["nutrition_levels"]
    missing = result["nutrients_missing"]
    reasons = result["reasons"]

    no_table = nutrition_dict.get("no_nutrition_table", False)

    GRADE_COLORS = {
        "A": "#059669", "B": "#16a34a",
        "C": "#d97706", "D": "#dc2626", "F": "#7f1d1d",
    }

    # Serving size
    _sv_raw = nutrition_data.serving_size or ""
    _sv_g   = None
    if _sv_raw:
        _m    = re.search(r"(\d+(?:\.\d+)?)\s*g(?!\w)", _sv_raw)
        _m_ml = re.search(r"(\d+(?:\.\d+)?)\s*ml(?=[^a-zA-Z]|$)", _sv_raw) if not _m else None
        if _m:     _sv_g = f"{float(_m.group(1)):g}g"
        elif _m_ml: _sv_g = f"{float(_m_ml.group(1)):g}ml"
    _sv_display = f"Per serving: {_sv_g}" if _sv_g else "Per serving: Not specified"

    st.markdown(
        f'<div style="font-size:.72rem;color:#9ca3af;font-family:monospace;'
        f'margin-bottom:1rem;">{_sv_display}</div>',
        unsafe_allow_html=True,
    )

    if st.button("← Scan Another Label", key="biz_back"):
        st.session_state.pop("biz_results", None)
        st.session_state.pop("bench_report", None)
        st.session_state.page = "biz_upload"
        st.rerun()

    st.divider()

    if no_table:
        st.warning(
            "⚠ This label does not contain a 栄養成分表示 (nutrition facts table). "
            "Please upload the side of the package showing the nutrition table."
        )
        al_col, ing_col = st.columns(2)
        with al_col:
            st.markdown("**Allergens**")
            _render_allergens(nutrition_data.allergens_in_product)
        with ing_col:
            st.markdown("**Ingredients**")
            st.write(translated or nutrition_data.ingredients or "Not found")
        st.markdown("**Label Information**")
        _render_label_info(label_fields)
        symbols_html = format_symbols_html(symbol_data)
        if symbols_html:
            st.markdown("**Symbols & Certifications**")
            st.markdown(symbols_html, unsafe_allow_html=True)

    else:
        _render_readiness_card(readiness, GRADE_COLORS)

        st.divider()

        nt_col, al_col = st.columns([1.6, 1])
        with nt_col:
            st.markdown(
                '<div style="font-size:.62rem;letter-spacing:.2em;color:#6b7280;'
                'text-transform:uppercase;margin-bottom:.5rem;">Nutrition Facts</div>',
                unsafe_allow_html=True,
            )
            _render_nutrition_table(nutrition_dict, levels, missing)
        with al_col:
            st.markdown(
                '<div style="font-size:.62rem;letter-spacing:.2em;color:#6b7280;'
                'text-transform:uppercase;margin-bottom:.5rem;">Allergens</div>',
                unsafe_allow_html=True,
            )
            _render_allergens(nutrition_data.allergens_in_product)
            adecl = label_fields.get("allergen_declaration_style")
            if adecl:
                st.markdown(
                    f'<span style="font-size:.6rem;background:#fef9c3;'
                    f'border:1px solid #fde047;color:#713f12;padding:.2rem .6rem;'
                    f'border-radius:4px;">{adecl}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("---")
            st.markdown(
                '<div style="font-size:.62rem;letter-spacing:.2em;color:#6b7280;'
                'text-transform:uppercase;margin-bottom:.5rem;">Health Notes</div>',
                unsafe_allow_html=True,
            )
            for r in reasons:
                icon  = ("✓" if r.startswith("Positive") else
                         "⚠" if r.startswith("Caution") else
                         "✗" if r.startswith("Concern") else "·")
                color = ("#27ae60" if icon == "✓" else
                         "#e67e22" if icon == "⚠" else
                         "#c0392b" if icon == "✗" else "#888")
                st.markdown(
                    f'<div style="font-size:.82rem;line-height:1.7;margin-bottom:.4rem;">'
                    f'<span style="color:{color};font-weight:700;">{icon}</span> {r}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:.62rem;letter-spacing:.2em;color:#6b7280;'
            'text-transform:uppercase;margin-bottom:.4rem;">Ingredients (Translated)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:.85rem;line-height:1.9;color:#374151;">'
            f'{translated or nutrition_data.ingredients or "Not found on label"}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        lbl_col, sym_col = st.columns([1.4, 1])
        with lbl_col:
            st.markdown(
                '<div style="font-size:.62rem;letter-spacing:.2em;color:#6b7280;'
                'text-transform:uppercase;margin-bottom:.4rem;">Label Information</div>',
                unsafe_allow_html=True,
            )
            _render_label_info(label_fields)
            cooking = label_fields.get("cooking_instructions")
            if cooking:
                with st.expander("🍳 Cooking / Usage Instructions"):
                    st.markdown(
                        f'<div style="font-size:.82rem;line-height:1.9;'
                        f'white-space:pre-wrap;">{cooking}</div>',
                        unsafe_allow_html=True,
                    )
        with sym_col:
            st.markdown(
                '<div style="font-size:.62rem;letter-spacing:.2em;color:#6b7280;'
                'text-transform:uppercase;margin-bottom:.4rem;">Symbols & Certifications</div>',
                unsafe_allow_html=True,
            )
            symbols_html = format_symbols_html(symbol_data)
            if symbols_html:
                st.markdown(symbols_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<span style="color:#9ca3af;font-size:.8rem;font-style:italic;">'
                    'No symbols detected</span>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs([
            "🏷️ Compliance Validator",
            "📊 Competitor Benchmarking",
            "🎨 Improvement Suggestions",
        ])

        # compliance tab
        with tab1:
            st.markdown(
                '<div style="font-size:.62rem;letter-spacing:.2em;color:#6b7280;'
                'text-transform:uppercase;margin-bottom:1rem;">Regulatory Compliance</div>',
                unsafe_allow_html=True,
            )
            for std, report in compliance_reports.items():
                color = "#27ae60" if report.passed else "#c0392b"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:1rem;margin:.8rem 0;">'
                    f'<span class="score-badge" style="background:{color};color:#fff;">'
                    f'{report.score}</span>'
                    f'<div><strong style="font-size:.95rem;">{report.standard}</strong><br>'
                    f'<span style="font-size:.75rem;color:#888;">{report.summary}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                errors   = [i for i in report.issues if i.severity == "error"]
                warnings = [i for i in report.issues if i.severity == "warning"]
                infos    = [i for i in report.issues if i.severity == "info"]
                passes   = report.passed_checks

                for i in errors:
                    st.markdown(
                        f'<div class="issue-error"><strong>✗ ERROR [{i.code}]</strong>'
                        f' — {i.message}</div>',
                        unsafe_allow_html=True,
                    )
                for i in warnings:
                    st.markdown(
                        f'<div class="issue-warning">⚠ {i.message}</div>',
                        unsafe_allow_html=True,
                    )
                for i in infos:
                    st.markdown(
                        f'<div class="issue-info">ℹ {i.message}</div>',
                        unsafe_allow_html=True,
                    )
                if passes:
                    with st.expander(f"✓ {len(passes)} checks passed"):
                        for p in passes[:8]:
                            st.markdown(
                                f'<div style="font-size:.82rem;color:#27ae60;">{p}</div>',
                                unsafe_allow_html=True,
                            )
                st.markdown("---")

        # benchmarking tab
        with tab2:
            st.markdown(
                '<div style="font-size:.62rem;letter-spacing:.2em;color:#6b7280;'
                'text-transform:uppercase;margin-bottom:1rem;">Competitor Benchmarking</div>',
                unsafe_allow_html=True,
            )
            _bench_nd  = st.session_state.get("bench_nd", nutrition_dict)
            _auto_cat  = st.session_state.get("bench_category", category)

            _CAT_META = {
                "instant_noodles":   ("🍜", "Instant Noodles",     "Cup / bag ramen, udon, soba"),
                "potato_snacks":     ("🍟", "Potato & Corn Snacks", "Chips, puffs, corn snacks"),
                "chocolate":         ("🍫", "Chocolate",            "Bars, pocky, coated snacks"),
                "beverages":         ("🧃", "RTD Beverages",        "Cans, bottles, ready-to-drink"),
                "instant_beverages": ("☕", "Instant Drinks",       "Stick coffee, powder mixes"),
                "rice_crackers":     ("🍘", "Rice Crackers",        "Senbei, okaki, arare"),
                "packaged_meals":    ("🍱", "Packaged Meals",       "Retort curry, stew, rice"),
                "supplements":       ("💪", "Supplements",          "Protein bars, shakes, powder"),
                "soft_candy":        ("🍬", "Soft Candy & Gummies", "Hi-chew, gummies, chewy candy"),
                "soup":              ("🍵", "Soup & Miso",          "Instant miso, potage, consommé"),
            }

            _selected_cat = st.session_state.get("bench_selected_cat", _auto_cat)
            st.caption(
                f"Auto-detected: **{_CAT_META.get(_auto_cat, ('', 'Unknown', ''))[1]}** "
                f"— click any card to override"
            )

            _cat_keys = list(_CAT_META.keys())
            for row_keys in [_cat_keys[:5], _cat_keys[5:]]:
                cols = st.columns(len(row_keys))
                for col, cat_key in zip(cols, row_keys):
                    emoji, name, desc = _CAT_META[cat_key]
                    is_sel = (cat_key == _selected_cat)
                    with col:
                        border = "2px solid #3b82f6" if is_sel else "1px solid #e2e8f0"
                        bg     = "#eff6ff" if is_sel else "#fff"
                        st.markdown(
                            f'<div style="background:{bg};border:{border};border-radius:8px;'
                            f'padding:.6rem .4rem;text-align:center;margin-bottom:.3rem;">'
                            f'<div style="font-size:1.1rem;">{emoji}</div>'
                            f'<div style="font-size:.65rem;font-weight:600;color:#0f172a;">{name}</div>'
                            f'<div style="font-size:.58rem;color:#9ca3af;line-height:1.3;">{desc}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            "✓ Selected" if is_sel else "Select",
                            key=f"cat_btn_{cat_key}",
                            use_container_width=True,
                        ):
                            st.session_state["bench_selected_cat"] = cat_key
                            _new_bench = benchmark(_bench_nd, cat_key)
                            st.session_state["bench_report"] = _new_bench
                            _r2 = dict(st.session_state.get("biz_results", {}))
                            if _r2:
                                _new_suggestions = generate_suggestions(
                                    _r2["nutrition_dict"],
                                    _new_bench,
                                    _r2["compliance_reports"],
                                )
                                _r2["suggestions"] = _new_suggestions
                                _new_readiness = compute_readiness(
                                    _r2["compliance_reports"], _new_bench,
                                    _new_suggestions, bench_confirmed=True,
                                )
                                _r2["readiness"] = _new_readiness
                                st.session_state["biz_results"] = _r2
                            st.rerun()

            bench_report = st.session_state.get("bench_report")
            if bench_report:
                st.markdown("---")

                _peers = bench_report.get("peers", [])
                if _peers:
                    _peer_pills = "".join(
                        f'<span style="display:inline-flex;align-items:center;gap:4px;'
                        f'background:#f1f5f9;border:1px solid #e2e8f0;border-radius:6px;'
                        f'padding:3px 10px;margin:3px 3px;font-size:.72rem;color:#334155;'
                        f'font-family:monospace;white-space:nowrap;">'
                        f'<span style="color:#94a3b8;">vs</span> {p["name"]}</span>'
                        for p in _peers
                    )
                    st.markdown(
                        f'<div style="margin-bottom:.75rem;">'
                        f'<div style="font-size:.6rem;letter-spacing:.18em;text-transform:uppercase;'
                        f'color:#9ca3af;margin-bottom:.4rem;">Compared against {len(_peers)} products</div>'
                        f'<div style="display:flex;flex-wrap:wrap;">{_peer_pills}</div>'
                        f'<div style="font-size:.6rem;color:#cbd5e1;margin-top:.35rem;">'
                        f'Source: Open Food Facts barcodes + official JP brand pages · all values per-100g</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                better = sum(
                    1 for d in bench_report["comparisons"].values()
                    if d.get("better_than_avg")
                )
                worse = len(bench_report["comparisons"]) - better
                pills = ""
                if better:
                    pills += (
                        f'<span style="background:#dcfce7;color:#166534;border-radius:20px;'
                        f'padding:3px 12px;font-size:.8rem;font-weight:600;margin-right:6px;">'
                        f'✓ Better than avg: {better}</span>'
                    )
                if worse:
                    pills += (
                        f'<span style="background:#fee2e2;color:#991b1b;border-radius:20px;'
                        f'padding:3px 12px;font-size:.8rem;font-weight:600;">'
                        f'✗ Worse than avg: {worse}</span>'
                    )
                if pills:
                    st.markdown(pills, unsafe_allow_html=True)

                metric_labels = {
                    "calories":      "Calories (kcal/100g)",
                    "protein":       "Protein (g/100g)",
                    "fat":           "Fat (g/100g)",
                    "carbohydrates": "Carbs (g/100g)",
                    "sodium":        "Sodium (g/100g)",
                    "sugar":         "Sugar (g/100g)",
                    "fiber":         "Fiber (g/100g)",
                }
                rows = ""
                for m, data in bench_report["comparisons"].items():
                    lbl      = metric_labels.get(m, m)
                    our      = data["our_value"]
                    avg      = data["category_avg"]
                    best     = data["best_in_class"]
                    diff     = data["diff_vs_avg_pct"]
                    better_m = data["better_than_avg"]
                    n_p      = data.get("n_peers", "?")
                    inferred = data.get("inferred", False)
                    arrow    = "▼" if diff < 0 else "▲"
                    color    = "#059669" if better_m else "#dc2626"
                    our_disp = f"~{our}" if inferred else str(our)
                    rows += (
                        f'<tr><td style="font-size:.82rem;">{lbl}</td>'
                        f'<td><span style="font-weight:600;color:{color};">{our_disp}</span></td>'
                        f'<td style="color:#6b7280;">{avg}</td>'
                        f'<td style="color:#6b7280;">{best}</td>'
                        f'<td style="color:{color};">{arrow}{abs(diff):.1f}%</td>'
                        f'<td style="color:#9ca3af;font-size:.72rem;">{n_p}</td></tr>'
                    )
                st.markdown(
                    f'<div style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;margin-top:.5rem;">'
                    f'<table class="bench-table"><thead><tr>'
                    f'<th>Nutrient</th><th>Yours</th>'
                    f'<th>Category Avg</th><th>Best in Class</th>'
                    f'<th>vs Avg</th><th>Peers</th>'
                    f'</tr></thead><tbody>{rows}</tbody></table></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="pending-note">⏳ Select a category card above '
                    'to run the benchmark comparison.</div>',
                    unsafe_allow_html=True,
                )

        # suggestions tab
        with tab3:
            st.markdown(
                '<div style="font-size:.62rem;letter-spacing:.2em;color:#6b7280;'
                'text-transform:uppercase;margin-bottom:1rem;">'
                'Label & Product Improvement Suggestions</div>',
                unsafe_allow_html=True,
            )
            _bench_confirmed = st.session_state.get("bench_report") is not None
            if not _bench_confirmed:
                st.markdown(
                    '<div class="pending-note">⚠ <strong>Competitive suggestions pending</strong> — '
                    'go to <strong>Competitor Benchmarking</strong> and select a category to unlock.</div>',
                    unsafe_allow_html=True,
                )
            if not suggestions:
                st.success("No major improvements needed — product profile looks strong.")
            else:
                high   = [s for s in suggestions if s.priority == "high"]
                medium = [s for s in suggestions if s.priority == "medium"]
                low    = [s for s in suggestions if s.priority == "low"]

                for group, label in [
                    (high,   "🔴 High Priority"),
                    (medium, "🟡 Medium Priority"),
                    (low,    "🔵 Low Priority"),
                ]:
                    if not group:
                        continue
                    st.markdown(
                        f'<div style="font-size:.65rem;letter-spacing:.2em;color:#888;'
                        f'text-transform:uppercase;margin:.8rem 0 .4rem;">{label}</div>',
                        unsafe_allow_html=True,
                    )
                    for s in group:
                        st.markdown(
                            f'<div class="sug-card sug-card-{s.priority}">'
                            f'<div class="sug-header">'
                            f'<span class="sug-title">{s.title}</span>'
                            f'<span class="sug-cat-tag">{s.category}</span>'
                            f'</div>'
                            f'<div class="sug-detail">{s.detail}</div>'
                            f'<div class="sug-meta">'
                            f'<span class="sug-impact">💰 {s.impact}</span>'
                            f'<span class="sug-how">⚙ {s.how}</span>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )


def route(nav_fn):
    """Dispatch to the business results or upload page; nav_fn is the shared nav from app.py."""
    page = st.session_state.page

    if page == "biz_results":
        page_biz_results(nav_fn)
    else:
        st.session_state.page = "biz_upload"
        page_biz_upload(nav_fn)