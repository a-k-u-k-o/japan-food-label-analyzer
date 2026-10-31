"""Streamlit pages for the consumer role: label upload, results, smart pairing and feedback."""

import json
import re
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from consumer.services.detection     import analyze_ingredients, JP_MANDATORY
from consumer.services.diet          import classify_diet
from consumer.services.health_score  import compute_health_score
from consumer.services.smart_pairing import smart_pair
from shared.storage import save_image, save_record, save_behavior


@st.cache_resource
def _load_pairing_data():
    """Load the food DB, allergen lookups, additives table and diet blocklists once per session."""
    data_dir = Path(__file__).parent.parent / "data"
    food_db  = pd.read_csv(data_dir / "food_products_cleaned.csv")
    am       = pd.read_csv(data_dir / "allergen_master.csv")
    at       = pd.read_csv(data_dir / "allergen_tokens.csv")
    ins      = pd.read_csv(data_dir / "ingredient_sources.csv")
    adf      = pd.read_csv(data_dir / "japanese_food_additives.csv", on_bad_lines="skip")

    id2a: dict[int, str] = dict(
        zip(am["allergen_id"].astype(int),
            am["canonical_name"].str.lower().str.strip())
    )
    t2a: dict[str, int] = {}

    for _, row in at.iterrows():
        if bool(pd.notna(row["allergen_id"])) and bool(pd.notna(row["token"])):
            t2a[str(row["token"]).lower().strip()] = int(row["allergen_id"])
    for _, row in ins.iterrows():
        if bool(pd.notna(row["allergen_id"])) and bool(pd.notna(row["ingredient"])):
            k = str(row["ingredient"]).lower().strip()
            if k not in t2a:
                t2a[k] = int(row["allergen_id"])

    adf["name_normalized"] = adf["name"].str.lower().str.strip()

    with open(data_dir / "diet_blocklists.json", encoding="utf-8") as f:
        diet_lists = json.load(f)

    return food_db, t2a, id2a, adf, diet_lists


def _bmi_color(cls: str) -> str:
    """Badge colour for a BMI class."""
    return {
        "Underweight":   "#3B9AE1",
        "Normal weight": "#2EC4B6",
        "Overweight":    "#F4A261",
        "Obese":         "#E63946",
    }.get(cls, "#888")


def _strip_float(val):
    """Parse a nutrition value to (number, is_mg); percentages and unparsable values give None."""
    if val is None:
        return None, False
    s = str(val).strip()
    if re.fullmatch(r"0+\.?0*\s*%", s):
        return 0.0, False
    if "%" in s:
        return None, False
    is_mg = bool(re.search(r"mg", s, re.IGNORECASE))
    try:
        return float(s), is_mg
    except (ValueError, TypeError):
        m = re.search(r"-?\d+\.?\d*", s)
        if m:
            try:
                return float(m.group()), is_mg
            except ValueError:
                pass
    return None, is_mg


_NUT_LIMITS = {
    "Calories":      (9999, None),
    "Protein":       (200,  None),
    "Fat":           (200,  None),
    "Saturated Fat": (100,  None),
    "Carbohydrate":  (500,  None),
    "Sugar":         (300,  None),
    "Fibre":         (100,  None),
    "Salt":          (10,   100),
}

# Optional rows on JP labels: a "0" from OCR usually means the row was not printed
_ZERO_SUPPRESS = {"Sugar", "Fibre", "Saturated Fat"}


def _clamp(label: str, value: float, is_mg: bool):
    """Convert a value to its display unit (kcal or g), rescaling mg readings."""
    unit = "kcal" if label == "Calories" else "g"
    if label == "Calories":
        return value, unit
    if is_mg:
        value = value / 1000.0
    elif label in _NUT_LIMITS:
        _, mg_thresh = _NUT_LIMITS[label]
        if mg_thresh and value > mg_thresh:
            value = value / 1000.0
    return round(value, 3), unit


def parse_nutrition(english: dict):
    """Turn the English nutrition dict into display rows plus serving basis, or None if empty."""
    nut = (english or {}).get("nutrition") or {}
    if not nut:
        return None

    def _qn(v):
        if v is None:
            return None
        m = re.search(r"\d+\.?\d*", str(v))
        return float(m.group()) if m else None

    # OCR sometimes copies the carbohydrate value into sugar
    sv = _qn(nut.get("sugar"))
    cv = _qn(nut.get("carbohydrate"))
    if sv is not None and cv is not None and abs(sv - cv) < 0.1:
        nut["sugar"] = None

    def _g(n, *keys):
        for k in keys:
            v = n.get(k)
            if v is not None:
                return v
        return None

    # bare "sodium" comes last: it may be in mg, which would be mis-scaled
    field_map = [
        (["calories_kcal", "calories"],                           "Calories"),
        (["protein_g",     "protein"],                            "Protein"),
        (["fat_g",         "fat"],                                "Fat"),
        (["saturated_fat_g","saturated_fat"],                     "Saturated Fat"),
        (["carbohydrate_g", "carbohydrate", "carbs"],             "Carbohydrate"),
        (["sugar_g",        "sugar",        "sugars"],            "Sugar"),
        (["fibre_g",        "fibre",        "fiber", "dietary_fibre"], "Fibre"),
        (["salt_g",         "sodium_g",     "salt",  "sodium"],   "Salt"),
    ]

    rows = []
    for keys, label in field_map:
        val = _g(nut, *keys)
        if val is None:
            continue
        num, is_mg = _strip_float(val)
        if num is not None:
            if num == 0.0 and label in _ZERO_SUPPRESS:
                continue
            c, u = _clamp(label, num, is_mg)
            rows.append({"Nutrient": label, "Amount": c, "Unit": u})
        else:
            rows.append({
                "Nutrient": label,
                "Amount":   str(val),
                "Unit":     "kcal" if label == "Calories" else "g",
            })

    # a saturated fat value equal to salt is almost certainly a key collision, so drop it
    salt_amt = next(
        (r["Amount"] for r in rows if r["Nutrient"] == "Salt"
         and isinstance(r["Amount"], float)),
        None,
    )
    if salt_amt is not None:
        rows = [
            r for r in rows
            if not (
                r["Nutrient"] == "Saturated Fat"
                and isinstance(r["Amount"], float)
                and abs(r["Amount"] - salt_amt) < 0.001
            )
        ]

    return {"rows": rows, "basis": nut.get("basis")} if rows else None


def _nutrition_flat(result: dict) -> dict:
    """Flatten the scan's English nutrition into numeric values keyed for smart pairing."""
    english = (result.get("ocr") or {}).get("english") or {}
    nut = english.get("nutrition") or {}

    def _n(v):
        if v is None:
            return None
        m = re.search(r"\d+\.?\d*", str(v))
        return float(m.group()) if m else None

    def _salt(n: dict) -> float | None:
        for k in ("salt_g", "sodium_g", "salt", "sodium"):
            v = n.get(k)
            if v is not None:
                parsed = _n(v)
                if parsed is not None:
                    # some OCR outputs store sodium_g in mg
                    if k == "sodium_g" and parsed > 10:
                        return round(parsed / 1000, 3)
                    return parsed
        return None

    return {
        "calories":        _n(nut.get("calories")),
        "protein_g":       _n(nut.get("protein")),
        "fat_g":           _n(nut.get("fat")),
        "carbohydrate_g":  _n(nut.get("carbohydrate")),
        "sugar_g":         _n(nut.get("sugar")),
        "fibre_g":         _n(nut.get("fibre")),
        "salt_g":          _salt(nut),
        "saturated_fat_g": _n(nut.get("saturated_fat")),
    }


def _confidence(jp: dict, en: dict) -> dict:
    """Per-field and overall OCR completeness scores (0-100) from the Japanese extraction."""
    s: dict[str, int] = {}

    pname = (jp or {}).get("product_name")
    s["Product Name"] = 100 if pname and str(pname).strip() else 0

    ing   = (jp or {}).get("ingredients") or {}
    items = ing.get("items") or [] if isinstance(ing, dict) else []
    s["Ingredients"] = (
        100 if len(items) >= 5 else
        80  if len(items) >= 2 else
        70  if len(items) == 1 else 0
    )

    alg = (jp or {}).get("allergens")
    if alg is None:
        s["Allergens"] = 0
    elif isinstance(alg, dict):
        st2, ai = alg.get("style"), alg.get("items") or []
        s["Allergens"] = (
            100 if st2 and ai else
            80  if st2 == "none" else
            60  if st2 else 30
        )
    else:
        s["Allergens"] = 30

    nut = (jp or {}).get("nutrition") or {}
    s["Nutrition"] = round(
        sum(1 for k in ["calories", "protein", "fat", "carbohydrate", "salt"]
            if nut.get(k)) / 5 * 100
    )
    return {"fields": s, "overall": round(sum(s.values()) / len(s))}


def _conf_badge(score: int) -> str:
    """Format a confidence score with a green, yellow or red marker."""
    if score >= 80:
        return f"🟢 {score}%"
    if score >= 50:
        return f"🟡 {score}%"
    return f"🔴 {score}%"


def _pie(rows: list):
    """Macronutrient pie chart figure, or None if there are no macro values."""
    keys = {"Protein", "Fat", "Carbohydrate"}
    labels, values = [], []
    for r in rows:
        if r["Nutrient"] not in keys:
            continue
        try:
            v = float(r["Amount"])
        except (ValueError, TypeError):
            v = 0.0
        if v > 0:
            labels.append(r["Nutrient"])
            values.append(v)
    if not labels:
        return None
    colors = ["#4C9BE8", "#F4A261", "#2EC4B6"][: len(labels)]
    fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor="none")

    pie_result = ax.pie(
        values, labels=None, autopct="%1.1f%%", colors=colors,
        startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    if len(pie_result) >= 3:
        for at in pie_result[2]:
            at.set_fontsize(10)
            at.set_color("white")
            at.set_fontweight("bold")

    patches = [
        mpatches.Patch(color=c, label=f"{l} ({v:.1f}g)")
        for c, l, v in zip(colors, labels, values)
    ]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), ncol=len(labels),
              fontsize=8, frameon=False)
    ax.set_title("Macronutrients", fontsize=10, pad=6)
    fig.tight_layout()
    return fig


def _dri_bar(nut_rows: list, dri_raw: dict):
    """Bar chart of each nutrient as a percentage of the daily DRI, or None if nothing to compare."""
    pairs = [
        ("Calories",     "calories",       "kcal"),
        ("Protein",      "protein_g",      "g"),
        ("Fat",          "fat_max_g",      "g"),
        ("Carbohydrate", "carb_max_g",     "g"),
        ("Sugar",        "sugar_limit_g",  "g"),
        ("Salt",         "sodium_g",       "g"),
    ]
    lv_map = {
        r["Nutrient"]: r["Amount"]
        for r in nut_rows
        if isinstance(r.get("Amount"), float)
    }
    names, pcts = [], []
    for nutrient, dri_key, _ in pairs:
        lv = lv_map.get(nutrient)
        dv = dri_raw.get(dri_key)
        if lv is not None and dv:
            try:
                names.append(nutrient)
                pcts.append(round(float(lv) / float(dv) * 100, 1))
            except (ValueError, TypeError, ZeroDivisionError):
                pass
    if not names:
        return None
    colors = [
        "#2EC4B6" if p <= 25 else
        "#4C9BE8" if p <= 50 else
        "#F4A261" if p <= 80 else
        "#E63946"
        for p in pcts
    ]
    fig, ax = plt.subplots(figsize=(5, 3.2), facecolor="none")
    bars = ax.barh(names, pcts, color=colors, edgecolor="none", height=0.42)
    ax.axvline(100, color="#E63946", linestyle="--",
               linewidth=1.1, alpha=0.6, label="100% DRI")
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{pct}%", va="center", fontsize=9, color="#333",
        )
    ax.set_xlabel("% of Daily DRI", fontsize=9)
    ax.set_xlim(0, max(pcts + [100]) * 1.28)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def page_upload(nav_fn):
    """Upload page: take a label photo, run OCR and analysis, save the record and go to results."""
    nav_fn()
    user = st.session_state["user"]

    st.markdown('<div class="page-title">Scan a Food Label</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Upload a photo of a Japanese food product label '
        'to analyse its ingredients, allergens and nutrition.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Drop your label photo here",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="visible",
        )

        if uploaded is not None:
            st.image(uploaded, use_container_width=True)
            _sig = getattr(uploaded, "file_id", None) or (
                uploaded.name + str(uploaded.size)
            )
            st.session_state["_current_sig"] = _sig
        else:
            st.session_state["_current_sig"] = ""
            st.info("📂 Upload a Japanese food label image above.")

        can_scan = uploaded is not None
        if st.button(
            "🔍 Analyse Label", type="primary",
            disabled=not can_scan, use_container_width=True,
        ):
            if uploaded is None:
                st.error("Please upload an image first.")
                st.stop()

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded.name).suffix
                ) as tmp:
                    uploaded.seek(0)
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                with st.spinner("🔄 Running OCR + translation…"):
                    from consumer.services.ocr import extract_label
                    ocr_result = extract_label(tmp_path)

                if ocr_result.get("error") == "not_a_label":
                    st.error(
                        "⚠️ **Please enter the image again.**  "
                        "The uploaded image does not appear to be a Japanese "
                        "food label, or the label language is not Japanese or "
                        "English.  Please upload a clear photo of a Japanese "
                        "product label and try again."
                    )
                    st.stop()

                if ocr_result.get("error") and not ocr_result.get("ingredients"):
                    st.error(f"OCR failed: {ocr_result['error']}")
                else:
                    ingredients_text = ocr_result.get("ingredients", "")
                    english  = ocr_result.get("english") or {}
                    japanese = ocr_result.get("japanese") or {}

                    if not isinstance(english.get("nutrition"), dict):
                        english["nutrition"] = {}
                    raw_ing = english.get("ingredients")
                    if isinstance(raw_ing, str):
                        english["ingredients"] = [
                            i.strip() for i in raw_ing.split(",") if i.strip()
                        ]
                    elif not isinstance(raw_ing, list):
                        english["ingredients"] = []

                    with st.spinner("🔬 Detecting allergens & additives…"):
                        detection = analyze_ingredients(ingredients_text)

                    en_nutrition  = english.get("nutrition")
                    diet_result   = classify_diet(
                        ingredients_flat=ingredients_text,
                        nutrition=en_nutrition,
                        allergens=detection["allergens"],
                    )
                    health_result = compute_health_score(
                        ingredients_flat=ingredients_text,
                        nutrition=en_nutrition,
                        additives=detection["additives"],
                        dri_raw=(st.session_state["dri"] or {}).get("_raw"),
                    )

                    saved_img = save_image(tmp_path)

                    raw_pname = english.get("product_name") or ""
                    if isinstance(raw_pname, dict):
                        product_name = str(
                            raw_pname.get("English_name") or ""
                        ).strip()
                    else:
                        product_name = str(raw_pname).strip()

                    scan_id = save_record(
                        user_id=user["user_id"],
                        username=user["username"],
                        user_allergen=user.get("allergen", ""),
                        image_path=saved_img,
                        product_name=product_name,
                        ingredients=ingredients_text,
                        detected_allergens=detection["allergens"],
                        detected_additives=detection["additives"],
                        health_score=health_result.get("score", 0),
                        health_grade=health_result.get("grade", ""),
                        health_verdict=health_result.get("verdict", ""),
                        dri=st.session_state["dri"],
                    )

                    st.session_state["scan_result"] = {
                        "scan_id":     scan_id,
                        "ocr":         ocr_result,
                        "detection":   detection,
                        "diet":        diet_result,
                        "health":      health_result,
                        "ingredients": ingredients_text,
                        "confidence":  _confidence(japanese, english),
                    }
                    st.session_state["scan_id"]        = scan_id
                    st.session_state["_last_sig"]      = _sig
                    st.session_state["pairing_result"] = None
                    st.session_state["page"]           = "results"
                    st.rerun()

            finally:
                if tmp_path:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except OSError:
                        pass

    with right:
        st.markdown("##### Your Profile")
        dri = st.session_state["dri"]
        if dri:
            inp  = dri["inputs"]
            bclr = _bmi_color(dri["bmi_class"])
            st.markdown(f"""
            <div style="background:#f0f9ff;border-radius:12px;padding:16px 18px;border:1px solid #bae6fd;">
              <div style="font-size:13px;color:#0369a1;font-weight:600;margin-bottom:8px;">DRI Profile Active ✓</div>
              <div style="font-size:13px;color:#0f172a;">
                <b>{inp['sex'].title()}</b> · {inp['age']} yrs · {inp['height_cm']} cm · {inp['weight_kg']} kg<br>
                Activity: <b>{inp['activity']}</b><br>
                BMI: <b>{dri['bmi']}</b>
                <span style="background:{bclr};color:#fff;border-radius:8px;padding:1px 8px;font-size:11px;font-weight:600;margin-left:4px;">{dri['bmi_class']}</span><br>
                Daily calories: <b>{dri['eer']:,} kcal</b>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("💡 No DRI profile found. Update your body details in Account settings for personalised scoring.")

        allergen = user.get("allergen", "")
        if allergen:
            st.markdown(f"""
            <div style="background:#fff7ed;border-radius:12px;padding:14px 18px;border:1px solid #fed7aa;margin-top:12px;">
              <div style="font-size:13px;color:#9a3412;font-weight:600;">Your Allergen Alert 🚨</div>
              <div style="font-size:13px;color:#0f172a;margin-top:4px;">Products containing <b>{allergen.title()}</b> will be flagged.</div>
            </div>
            """, unsafe_allow_html=True)


def page_results(nav_fn):
    """Results page: ingredients, allergens, additives, health score, diet classes and nutrition vs DRI."""
    nav_fn()

    if not st.session_state["scan_result"]:
        st.warning("No scan results yet. Please upload and scan a label first.")
        if st.button("← Back to Upload"):
            st.session_state["page"] = "upload"
            st.rerun()
        return

    r         = st.session_state["scan_result"]
    detection = r["detection"]
    allergens = detection["allergens"]
    additives = detection["additives"]
    ocr       = r["ocr"]
    english   = ocr.get("english") or {}
    japanese  = ocr.get("japanese") or {}
    conf      = r.get("confidence") or _confidence(japanese, english)
    user      = st.session_state["user"]
    user_alg  = user.get("allergen", "")
    pname     = (english or {}).get("product_name") or "Unknown Product"

    st.markdown(f'<div class="page-title">{pname}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-sub">Scan ID: <code>{r["scan_id"]}</code> · '
        f'OCR {ocr.get("latency_s", "—")}s</div>',
        unsafe_allow_html=True,
    )

    conf_cols = st.columns(len(conf["fields"]) + 1)
    conf_cols[0].metric("Overall Confidence", _conf_badge(conf["overall"]))
    for col, (field, score) in zip(conf_cols[1:], conf["fields"].items()):
        col.metric(field, _conf_badge(score))

    st.divider()

    st.markdown('<p class="sec-hd">🧾 Ingredients (English)</p>',
                unsafe_allow_html=True)
    raw_items = english.get("ingredients") or []
    if isinstance(raw_items, list) and raw_items:
        chips = "".join(
            f'<span class="chip">{item}</span>' for item in raw_items
        )
        st.markdown(chips, unsafe_allow_html=True)
    elif r["ingredients"]:
        st.write(r["ingredients"])
    else:
        st.info("No ingredients extracted.")

    st.divider()

    st.markdown('<p class="sec-hd">⚠️ Allergen & Additive Alerts</p>',
                unsafe_allow_html=True)

    user_alg_found = (
        any((user_alg in det or det in user_alg) for det in allergens)
        if user_alg else False
    )
    if user_alg and user_alg_found:
        st.error(
            f"🚨 **Your allergen detected: {user_alg.title()}** — "
            f"this product contains it.", icon="🚨",
        )

    mandatory   = [a for a in allergens if a in JP_MANDATORY]
    other       = [a for a in allergens if a not in JP_MANDATORY]
    recommended = [
        a for a in other
        if not (user_alg and (user_alg in a or a in user_alg))
    ]

    acol1, acol2 = st.columns(2)
    with acol1:
        st.markdown("**🏷️ Allergens**")
        if mandatory:
            st.markdown("🔴 Mandatory (JP law)")
            st.markdown(
                "".join(f'<span class="pill pill-red">{a.title()}</span>'
                        for a in sorted(mandatory)),
                unsafe_allow_html=True,
            )
        if recommended:
            st.markdown("🟡 Recommended")
            st.markdown(
                "".join(f'<span class="pill pill-yellow">{a.title()}</span>'
                        for a in sorted(recommended)),
                unsafe_allow_html=True,
            )
        if not mandatory and not recommended and not user_alg_found:
            st.markdown('<span class="pill pill-green">✅ None detected</span>',
                        unsafe_allow_html=True)
    with acol2:
        st.markdown("**🧪 Additives**")
        if additives:
            st.markdown(
                "".join(
                    f'<span class="pill pill-orange">{aname}'
                    f'<span style="font-size:10px;opacity:.7"> #{aid}</span></span>'
                    for aid, aname in additives
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="pill pill-green">✅ None detected</span>',
                        unsafe_allow_html=True)

    st.divider()

    st.markdown('<p class="sec-hd">🏥 Health Score</p>', unsafe_allow_html=True)
    health  = r.get("health") or {}
    score   = health.get("score", 0)
    verdict = health.get("verdict", "Moderate")
    grade   = health.get("grade", "C")
    secs    = health.get("sections", {})
    vcls    = {
        "Healthy":   "health-healthy",
        "Moderate":  "health-moderate",
        "Unhealthy": "health-unhealthy",
    }.get(verdict, "health-moderate")
    tags_html = "".join(
        f'<span class="htag htag-neg">✗ {f}</span>'
        for f in health.get("top_flags", [])[:3]
    )
    tags_html += "".join(
        f'<span class="htag htag-pos">✓ {b}</span>'
        for b in health.get("top_boosts", [])[:2]
    )
    sec_lbl  = {"ingredients": "Ingredients", "macros": "Macros",
                "sodium": "Sodium", "additives": "Additives"}
    sec_html = "".join(
        f'<span class="sscore">{sec_lbl.get(sk, sk).title()} '
        f'{sv["score"]}/{sv["max"]}</span>'
        for sk, sv in secs.items()
    )
    st.markdown(f"""
    <div class="health-card {vcls}">
      <div class="health-grade">{grade}</div>
      <div class="health-body">
        <div class="health-verdict">{verdict}</div>
        <div class="health-score-line">Score: {score}/100</div>
        <div class="health-bar-wrap">
          <div class="health-bar" style="width:{score}%"></div>
        </div>
        <div class="health-tags">{tags_html}</div>
        <div class="section-scores">{sec_html}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Score Breakdown by Section", expanded=False):
        for sk, sv in secs.items():
            st.markdown(f"**{sec_lbl.get(sk, sk).title()}** — {sv['score']}/{sv['max']}")
            for f in sv.get("flags", []):
                st.markdown(f"  ✗ {f}", help="Penalty factor")
            for b in sv.get("boosts", []):
                st.markdown(f"  ✓ {b}", help="Positive factor")

    st.divider()

    st.markdown('<p class="sec-hd">🥗 Diet Classification</p>',
                unsafe_allow_html=True)
    diet = r.get("diet") or {}
    DIET_META = {
        "vegan":      ("🌱", "Vegan"),
        "vegetarian": ("🥚", "Vegetarian"),
        "keto":       ("🥑", "Keto"),
        "diabetic":   ("🩺", "Diabetic-Friendly"),
    }
    STATUS_CLASS = {
        "yes":       "diet-yes",
        "no":        "diet-no",
        "caution":   "diet-caution",
        "uncertain": "diet-uncertain",
    }
    dcols = st.columns(2)
    for idx, (key, (icon, title)) in enumerate(DIET_META.items()):
        d       = diet.get(key, {})
        status  = d.get("status", "uncertain")
        label   = d.get("label", "Unknown")
        css_cls = STATUS_CLASS.get(status, "diet-uncertain")
        rhtml   = ""
        for r_txt in d.get("reasons", [])[:2]:
            rhtml += f'<span class="neg">✗</span>{r_txt}<br>'
        for p_txt in d.get("positives", [])[:1]:
            rhtml += f'<span class="pos">✓</span>{p_txt}<br>'
        with dcols[idx % 2]:
            st.markdown(f"""
            <div class="diet-card {css_cls}">
              <span class="diet-icon">{icon}</span>
              <span class="diet-title">{title}</span>
              <span class="diet-label">{label}</span>
              <div class="diet-reason">{rhtml}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown('<p class="sec-hd">📊 Nutrition Information</p>',
                unsafe_allow_html=True)
    nut_data = parse_nutrition(english)
    if nut_data and nut_data["rows"]:
        if nut_data["basis"]:
            st.caption(f"Per serving: {nut_data['basis']}")
        nc1, nc2 = st.columns([1, 1])
        with nc1:
            df = pd.DataFrame(nut_data["rows"])

            def _fmt(row):
                amt, unit = row["Amount"], row["Unit"]
                if isinstance(amt, float):
                    n = int(amt) if amt == int(amt) else round(amt, 2)
                    return f"{n} {unit}"
                return str(amt)

            df["Value"] = df.apply(_fmt, axis=1)
            st.dataframe(
                df[["Nutrient", "Value"]].set_index("Nutrient"),
                use_container_width=True,
            )
        with nc2:
            fig = _pie(nut_data["rows"])
            if fig:
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("No macronutrient data for chart.")

        dri = st.session_state["dri"]
        if dri:
            st.markdown(
                '<p class="sec-hd">📈 This Product vs Your Daily DRI</p>',
                unsafe_allow_html=True,
            )
            st.caption(
                "What % of your estimated daily needs does this single product provide?"
            )
            dri_raw  = dri["_raw"]
            comp_fig = _dri_bar(nut_data["rows"], dri_raw)
            if comp_fig:
                st.pyplot(comp_fig, use_container_width=True)

            lv_map = {
                row["Nutrient"]: row["Amount"]
                for row in nut_data["rows"]
                if isinstance(row.get("Amount"), float)
            }
            sodium_g  = dri_raw.get("sodium_g")
            sugar_lim = dri_raw.get("sugar_limit_g")
            comp_rows = []
            for nutrient, dkey, dlabel in [
                ("Calories",     "calories",      f"{dri_raw.get('calories')} kcal"),
                ("Protein",      "protein_g",     f"{dri_raw.get('protein_g')} g"),
                ("Fat",          "fat_max_g",     f"{dri_raw.get('fat_max_g')} g max"),
                ("Carbohydrate", "carb_max_g",    f"{dri_raw.get('carb_max_g')} g max"),
                ("Sugar",        "sugar_limit_g", f"{sugar_lim} g (WHO 10%)" if sugar_lim else "—"),
                ("Salt",         "sodium_g",      f"{sodium_g} g" if sodium_g else "2.3 g"),
            ]:
                lv = lv_map.get(nutrient)
                dv = dri_raw.get(dkey)
                if lv is not None and dv:
                    try:
                        pct = round(float(lv) / float(dv) * 100, 1)
                        comp_rows.append({
                            "Nutrient":   nutrient,
                            "In Product": f"{lv}",
                            "Daily DRI":  dlabel,
                            "% of Need":  f"{pct}%",
                        })
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass
            if comp_rows:
                st.dataframe(
                    pd.DataFrame(comp_rows).set_index("Nutrient"),
                    use_container_width=True,
                )
        else:
            st.caption(
                "💡 Set up your body profile during registration "
                "to see personalised DRI comparisons."
            )
    else:
        st.info("ℹ️ No nutrition data extracted from this label.")

    with st.expander("🔬 Raw OCR Output", expanded=False):
        st.json(japanese or {})
        st.caption("Stage 1 (OCR — Japanese):")
        st.code((ocr.get("s1a_raw") or ocr.get("s1_raw") or "")[:3000] or "(empty)", language="json")
        st.caption("Stage 1b (Nutrition — Japanese):")
        st.code((ocr.get("s1b_raw") or "")[:3000] or "(empty)", language="json")
        st.caption("Stage 2 (Translation — English):")
        st.code((ocr.get("s2_raw") or "")[:3000] or "(empty)", language="json")

    st.divider()
    col_a, col_b, col_c = st.columns([2, 2, 2])
    with col_a:
        if st.button("← Scan Another", use_container_width=True):
            st.session_state["scan_result"] = None
            st.session_state["page"] = "upload"
            st.rerun()
    with col_b:
        if st.button("🍱 Smart Pairing →", type="primary", use_container_width=True):
            st.session_state["page"] = "pairing"
            st.rerun()
    with col_c:
        if st.button("Skip to Feedback →", use_container_width=True):
            st.session_state["page"] = "feedback"
            st.rerun()


def page_pairing(nav_fn):
    """Smart pairing page: suggest database products that fill the remaining per-meal nutrient gap."""
    nav_fn()

    if not st.session_state["scan_result"]:
        st.warning("No scan result found. Please scan a label first.")
        if st.button("← Back to Upload"):
            st.session_state["page"] = "upload"
            st.rerun()
        return
    if not st.session_state["dri"]:
        st.warning("DRI profile needed for pairing. Please complete your profile in Account settings.")
        if st.button("← Back to Results"):
            st.session_state["page"] = "results"
            st.rerun()
        return

    r        = st.session_state["scan_result"]
    dri_raw  = st.session_state["dri"]["_raw"]
    ocr      = r["ocr"]
    english  = ocr.get("english") or {}
    diet     = r.get("diet") or {}
    user     = st.session_state["user"]
    user_alg = user.get("allergen", "")
    pname    = english.get("product_name") or "Scanned product"

    st.markdown('<div class="page-title">🍱 Smart Pairing</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Find products that best complement your '
        'scanned item toward a balanced meal.</div>',
        unsafe_allow_html=True,
    )

    food_db, t2a, id2a, adf, diet_lists = _load_pairing_data()
    scanned_nutrition = _nutrition_flat(r)

    pc1, pc2, pc3 = st.columns([3, 2, 2])
    with pc1:
        st.markdown(f"**Scanned:** {pname}")
    with pc2:
        top_n = st.selectbox("Top suggestions", [3, 5, 10], index=1)
    with pc3:
        meals = st.selectbox("Meals per day", [2, 3, 4], index=1)

    user_allergens_input: str = st.text_input(
        "Exclude allergens (comma-separated)",
        value=user_alg,
        placeholder="e.g. milk, wheat, egg",
    ) or ""
    user_allergens = [
        a.strip().lower()
        for a in user_allergens_input.split(",")
        if a.strip()
    ]

    all_categories = sorted(food_db["Category"].dropna().unique().tolist())
    all_brands = sorted(
        set(str(n).split()[0]
            for n in food_db["Name-EN"].dropna()
            if str(n).strip())
    )

    with st.expander("🔎 Filter by Category & Brand (optional)", expanded=False):
        fc1, fc2 = st.columns(2)
        with fc1:
            selected_categories = st.multiselect(
                "Categories", options=all_categories,
                default=[], placeholder="All categories",
            )
        with fc2:
            selected_brands = st.multiselect(
                "Brands", options=all_brands,
                default=[], placeholder="All brands",
            )

    if st.button("🔍 Find Pairings", type="primary"):
        with st.spinner("Scoring database products…"):
            filtered_db: pd.DataFrame = food_db.copy()
            if selected_categories:
                filtered_db = pd.DataFrame(
                    filtered_db[filtered_db["Category"].isin(selected_categories)]
                )
            if selected_brands:
                filtered_db = pd.DataFrame(
                    filtered_db[
                        filtered_db["Name-EN"].apply(
                            lambda n: any(str(n).startswith(b) for b in selected_brands)
                        )
                    ]
                )

            if filtered_db.empty:
                st.warning("No products match the selected filters.")
                st.stop()

            pairing = smart_pair(
                scanned_nutrition=scanned_nutrition,
                scanned_diet=diet,
                dri_raw=dri_raw,
                food_db=filtered_db,
                token_to_allergen_id=t2a,
                id_to_allergen=id2a,
                additives_df=adf,
                diet_lists=diet_lists,
                user_allergens=user_allergens,
                top_n=30,
                meals=meals,
                product_name=pname,
            )

            pre_count      = len(pairing["top_pairings"])
            seen_brands_d: dict[str, int] = {}
            diverse        = []
            for c in pairing["top_pairings"]:
                brand = (
                    str(c.get("name_en", "")).split()[0]
                    if c.get("name_en") else ""
                )
                cnt = seen_brands_d.get(brand, 0)
                if cnt < 2:
                    diverse.append(c)
                    seen_brands_d[brand] = cnt + 1
                if len(diverse) >= top_n:
                    break

            pairing["top_pairings"] = diverse
            pairing["_meals_used"]  = meals
            pairing["_filter_note"] = (
                f"Categories: {', '.join(selected_categories) or 'All'}  |  "
                f"Brands: {', '.join(selected_brands) or 'All'}  |  "
                f"Showing {len(diverse)} "
                f"(diversity-filtered from {pre_count} scored)"
            )
            st.session_state["pairing_result"] = pairing

    pairing = st.session_state["pairing_result"]
    if pairing:
        if pairing.get("_filter_note"):
            st.caption(f"🔎 {pairing['_filter_note']}")

        st.markdown("---")
        st.markdown("#### 📊 Nutrient Gap After Scanned Product")
        st.caption(
            f"Per-meal target = daily DRI ÷ {pairing.get('_meals_used', meals)} meals"
        )

        gap    = pairing["gap_after_scan"]
        target = pairing["meal_dri_target"]
        icons  = {
            "calories":       "🔥",
            "protein_g":      "💪",
            "fat_g":          "🧈",
            "carbohydrate_g": "🌾",
            "salt_g":         "🧂",
        }
        gcols = st.columns(len(gap))
        for col, (k, v) in zip(gcols, gap.items()):
            tgt = target.get(k, 1)
            pct = round((1 - v / tgt) * 100) if tgt else 0
            col.metric(
                f"{icons.get(k, '')} {k.replace('_g','').replace('_',' ').title()}",
                f"{v:.1f}g" if "calorie" not in k else f"{v:.0f}kcal",
                f"{pct}% filled",
                delta_color="normal",
            )

        st.markdown(
            f"**{pairing['excluded_count']}** products excluded "
            f"(allergen conflict or excessive salt)."
        )
        st.markdown("#### 🏆 Top Pairing Suggestions")

        if not pairing["top_pairings"]:
            st.warning("No eligible pairing products found.")
        else:
            for rank, p in enumerate(pairing["top_pairings"], 1):
                sb        = p["score_breakdown"]
                composite = sb["composite"]
                tier      = p.get("pareto_tier", "?")
                pct_bar   = max(0, min(100, int(composite * 100)))
                alg_all   = sorted(
                    set(p["allergens"]["in_product"] + p["allergens"]["cross_contact"])
                )
                add_hi = [
                    a["name"] for a in p["additives"]
                    if str(a.get("risk_level", "")).lower() in ("high", "medium")
                ]
                gap_rows = ""
                for n in ["calories", "protein_g", "fat_g", "carbohydrate_g", "salt_g"]:
                    db_v   = p["nutrition"].get(n)
                    before = p["gap_before"].get(n, 0.0)
                    after  = p["gap_after"].get(n, 0.0)
                    filled = round(before - after, 2)
                    db_str = f"{db_v:.1f}" if db_v is not None else "—"
                    label  = n.replace("_g", "").replace("_", " ").title()
                    cls    = "gap-pos" if filled > 0 else "gap-neg"
                    gap_rows += (
                        f"<tr><td>{label}</td>"
                        f"<td>{db_str}</td>"
                        f"<td class='{cls}'>{filled:+.1f}</td></tr>"
                    )
                alg_str = ", ".join(a.title() for a in alg_all) if alg_all else "None"
                add_str = ", ".join(add_hi) if add_hi else "None flagged"
                st.markdown(f"""
                <div class="pair-card">
                  <div>
                    <span class="pair-rank">#{rank}</span>
                    <span class="pair-name">{p['name_en']}</span>
                    <span class="pair-cat">{p['category']}</span>
                  </div>
                  <div style="font-size:12px;color:#64748b;margin-top:4px;">
                    Pareto T{tier} · Composite: <b>{composite:.3f}</b> ·
                    L2={sb['l2_nutrition']:.3f} · Diet+={sb['diet_bonus']:.2f} ·
                    Add−={sb['additive_penalty']:.3f} · Salt−={sb['salt_penalty']:.3f}
                  </div>
                  <div class="pair-bar-wrap">
                    <div class="pair-bar" style="width:{pct_bar}%"></div>
                  </div>
                  <table class="gap-table">
                    <tr><th>Nutrient</th><th>DB value</th><th>Gap filled</th></tr>
                    {gap_rows}
                  </table>
                  <div style="margin-top:8px;font-size:12px;color:#475569;">
                    🏷️ <b>Allergens:</b> {alg_str} &nbsp;&nbsp;
                    🧪 <b>Flagged additives:</b> {add_str}
                  </div>
                </div>""", unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download pairing result (JSON)",
                data=json.dumps(pairing, ensure_ascii=False, indent=2),
                file_name="smart_pairing_result.json",
                mime="application/json",
            )

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← Back to Results", use_container_width=True):
            st.session_state["page"] = "results"
            st.rerun()
    with col_b:
        if st.button("Continue to Feedback →", type="primary",
                     use_container_width=True):
            st.session_state["page"] = "feedback"
            st.rerun()


def page_feedback(nav_fn):
    """Feedback page: record whether the user would buy the product and a suggested pairing."""
    nav_fn()

    if not st.session_state["scan_result"]:
        st.warning("No scan to give feedback on.")
        if st.button("← Start Over"):
            st.session_state["page"] = "upload"
            st.rerun()
        return

    r       = st.session_state["scan_result"]
    ocr     = r["ocr"]
    english = ocr.get("english") or {}
    pname   = english.get("product_name") or "this product"
    user    = st.session_state["user"]
    pairing = st.session_state["pairing_result"]

    st.markdown('<div class="page-title">Your Thoughts</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Help us understand your purchase behaviour — '
        'this data is saved anonymously for research.</div>',
        unsafe_allow_html=True,
    )

    health  = r.get("health") or {}
    score   = health.get("score", 0)
    grade   = health.get("grade", "?")
    verdict = health.get("verdict", "—")
    st.markdown(f"""
    <div style="background:#f8faff;border-radius:14px;padding:16px 20px;border:1.5px solid #c7d5f5;margin-bottom:24px;">
      <div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:4px;">{pname}</div>
      <div style="font-size:13px;color:#64748b;">
        Health Grade: <b style="font-size:18px;color:#0f172a;">{grade}</b> &nbsp;·&nbsp;
        Score: <b>{score}/100</b> &nbsp;·&nbsp; {verdict}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="buy-q">Would you buy this product?</div>',
                unsafe_allow_html=True)
    will_buy_choice = st.radio(
        "", ["✅ Yes, I would buy it", "❌ No, I would not buy it"],
        index=None, key="fb_will_buy", label_visibility="collapsed",
    )

    paired_product_selected = None
    chosen_pair_name        = ""
    chosen_pair_id          = ""

    if will_buy_choice == "✅ Yes, I would buy it":
        st.markdown("---")
        st.markdown(
            '<div class="buy-q" style="font-size:17px;">Would you also buy '
            'one of the recommended paired products?</div>',
            unsafe_allow_html=True,
        )
        has_pairings = pairing and pairing.get("top_pairings")
        if not has_pairings:
            st.info("💡 No pairing results found. Run Smart Pairing first.")
            buy_paired = st.radio(
                "", ["✅ Yes, I'd buy a pairing", "❌ No, just the scanned product"],
                index=None, key="fb_buy_paired", label_visibility="collapsed",
            )
            if buy_paired == "✅ Yes, I'd buy a pairing":
                st.info("Go back to Smart Pairing to see suggestions.")
                paired_product_selected = True
        else:
            pair_options = [
                p["name_en"] for p in pairing["top_pairings"] if p.get("name_en")
            ]
            buy_paired = st.radio(
                "", ["✅ Yes, I'd buy a pairing", "❌ No, just the scanned product"],
                index=None, key="fb_buy_paired", label_visibility="collapsed",
            )
            if buy_paired == "✅ Yes, I'd buy a pairing":
                paired_product_selected = True
                st.markdown("**Which paired product would you choose?**")
                chosen_pair_name = st.selectbox(
                    "Select a product",
                    options=["— pick one —"] + pair_options,
                    key="fb_pair_pick",
                )
                if chosen_pair_name != "— pick one —":
                    for p in pairing["top_pairings"]:
                        if p.get("name_en") == chosen_pair_name:
                            chosen_pair_id = str(p.get("product_id", ""))
                            break
                else:
                    chosen_pair_name = ""
            elif buy_paired == "❌ No, just the scanned product":
                paired_product_selected = False

    st.markdown("---")
    if st.button(
        "💾 Save My Response", type="primary",
        use_container_width=True, key="fb_submit",
    ):
        if will_buy_choice is None:
            st.error("Please answer whether you would buy this product.")
        else:
            will_buy = will_buy_choice.startswith("✅")
            if will_buy and paired_product_selected is None:
                st.error("Please answer the paired product question too.")
            else:
                save_behavior(
                    scan_id=r["scan_id"],
                    user_id=user["user_id"],
                    product_name=pname,
                    will_buy=will_buy,
                    paired_product_selected=paired_product_selected,
                    paired_product_name=(
                        chosen_pair_name
                        if chosen_pair_name != "— pick one —" else ""
                    ),
                    paired_product_id=chosen_pair_id,
                )
                st.success("✅ Response saved! Thank you.")
                st.balloons()
                st.markdown(f"""
                <div style="background:#f0fdf4;border-radius:12px;padding:16px 20px;border:1px solid #86efac;margin-top:16px;">
                  <div style="font-weight:700;color:#15803d;margin-bottom:6px;">Response Recorded</div>
                  <div style="font-size:13px;color:#0f172a;">
                    Product: <b>{pname}</b><br>
                    Would buy: <b>{"Yes" if will_buy else "No"}</b><br>
                    {"Would buy paired product: <b>" + ("Yes" if paired_product_selected else "No") + "</b><br>" if will_buy else ""}
                    {"Chosen pairing: <b>" + chosen_pair_name + "</b>" if chosen_pair_name and chosen_pair_name != "— pick one —" else ""}
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← Back to Pairing", use_container_width=True):
            st.session_state["page"] = "pairing"
            st.rerun()
    with col_b:
        if st.button(
            "🔄 Scan Another Product", type="primary",
            use_container_width=True,
        ):
            st.session_state["scan_result"]   = None
            st.session_state["pairing_result"] = None
            st.session_state["scan_id"]       = None
            st.session_state["page"]          = "upload"
            st.rerun()


def route(nav_fn):
    """Render the page named in session state, defaulting to upload."""
    page = st.session_state["page"]
    if page == "upload":
        page_upload(nav_fn)
    elif page == "results":
        page_results(nav_fn)
    elif page == "pairing":
        page_pairing(nav_fn)
    elif page == "feedback":
        page_feedback(nav_fn)
    else:
        st.session_state["page"] = "upload"
        st.rerun()