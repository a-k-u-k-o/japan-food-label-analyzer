# 🍱 Japan Food App

**AI-powered Japanese food label analysis — one Streamlit app, two audiences.**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)
![Model](https://img.shields.io/badge/VLM-Qwen2.5--VL--7B-6f42c1)
![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-orange)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20WSL-lightgrey)

Photograph a Japanese food label and get structured, actionable results:

| 🧑 Consumer | 🏢 Business |
|---|---|
| "Is this safe and healthy **for me**?" | "Is this label **ready for the Japanese market**?" |
| Allergen + additive detection, diet checks (vegan / vegetarian / keto / diabetic), a personalised health score, and complementary food pairings | JAS / CAA compliance check, benchmarking against 50 real competitor products, a Market Readiness Score, and AI-written improvement suggestions |

Both roles share one login system and one process (`streamlit run app.py`); the role chosen at registration decides which half of the app you see.

---

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Data files](#data-files)
- [Module reference](#module-reference)
- [Known limitations](#known-limitations)
- [Acknowledgements](#acknowledgements)

---

## Features

### Consumer
- **Label OCR** with Qwen2.5-VL-7B → structured nutrition facts + ingredient list
- **Allergen detection** against Japan's mandatory (8) and recommended (20) allergens, plus global-only allergens (e.g. fish, tree nut)
- **Additive detection** against a 351-entry additive database, tiered *benign / caution / avoid*
- **Diet classification** — Vegan, Vegetarian, Keto, Diabetic-friendly, each with reasons
- **Personalised health score** (0–100, grade A–F) driven by your USDA DRI profile
- **DRI calculator** — EER, macros, vitamins and minerals from sex, age, height, weight and activity level
- **Smart food pairing** — finds products that fill your nutritional gaps using L2-distance scoring and Pareto-front ranking
- Macronutrient pie chart, DRI comparison chart, and purchase-intent feedback capture

### Business
- **Multi-pass Qwen2.5-VL OCR** — main pass, focused second pass, and four structured-JSON prompts (company / recycling / claims / allergens)
- **OpenCV preprocessing** — upscale, Hough-line deskew, denoise, CLAHE, unsharp mask
- **Compliance checker** — JAS and CAA Food Labelling Act rules, with error / warning / info findings
- **Competitor benchmarking** — per-100 g comparison against 50 products in 10 categories, with live re-scoring when you change category
- **Market Readiness Score** (A–F, 0–100) from three weighted components
- **AI suggestions** via Groq (`llama-3.3-70b-versatile`) with an automatic rule-based fallback
- **Symbol detection** — recycling marks and certification logos
- **Hallucination guard** — rejects company fields that match known prompt examples

### Shared
- Role-based routing — consumer and business features are fully isolated
- Shared user store with file locking and live registration validation

---

## Quick start

```bash
git clone https://github.com/a-k-u-k-o/japan-food-label-analyzer.git
cd japan-food-label-analyzer

python -m venv venv
source venv/bin/activate          # Linux / macOS / WSL

pip install -r requirements.txt

cp .env.example .env              # optional: add your GROQ_API_KEY
streamlit run app.py
```

Open <http://localhost:8501>, register as a **Consumer** or **Business** user, and upload a label image.

> The first scan downloads Qwen2.5-VL-7B (~16 GB of weights from Hugging Face), so expect a long first run.

Useful variants:

```bash
streamlit run app.py --server.port 8080        # different port
streamlit run app.py --server.address 0.0.0.0  # expose on a cloud VM
```

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.12 |
| OS | Linux, macOS, or **WSL** on Windows | Linux |
| GPU | NVIDIA, ~7 GB VRAM (e.g. T4) | A100 / H100 |
| Disk | ~20 GB (model cache) | |

- **Both roles need the GPU.** Consumer and business OCR both load Qwen2.5-VL-7B with 4-bit NF4 quantisation via `bitsandbytes`, which requires CUDA.
- **Windows is not supported natively** — storage uses the Unix-only `fcntl` module. Use WSL (see [Known limitations](#known-limitations)).
- Flash Attention 2 is optional. It is used automatically if installed (`pip install flash-attn`, Ampere or newer GPUs).

---

## Configuration

### Environment variables

Copy `.env.example` to `.env` (git-ignored). The app loads it with `python-dotenv`.

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | No | Enables LLM-written improvement suggestions. Free key at [console.groq.com](https://console.groq.com). Without it, suggestions fall back to rule-based generation. |
| `PYTORCH_ALLOC_CONF` | Auto-set | Set to `expandable_segments:True` by `business/services/ocr.py` to reduce CUDA fragmentation OOMs. |

### Business model loader

`business/services/ocr.py` tries three loading strategies and uses the first that works:

```
Path 1 — 4-bit NF4 + Flash Attention 2   (A100 / H100)     ~6.5 GB VRAM
Path 2 — 4-bit NF4, standard attention   (T4 / V100)       ~6.5 GB VRAM
Path 3 — float16, no quantisation        (any GPU or CPU)  ~14 GB, slow
```

### Tuning the reference data

| To change… | Edit | Notes |
|---|---|---|
| DRI standards | `consumer/services/dri.py` | `VITAMIN_TABLE`, `MINERAL_TABLE`, `PA_MALE`, `PA_FEMALE` (USDA DRI 2002/2005) |
| Diet rules | `data/diet_blocklists.json` | `vegan_block`, `keto_block`, `diabetic_caution`, carb limits, … |
| Competitor products | `data/benchmark_products.json` | Per-100 g `calories, protein, fat, carbohydrates, sodium, sugar, fiber`; add `off_barcode` for traceability |

---

## How it works

### Consumer flow

```
Register (Consumer) ── body profile → DRI computed and stored
        │
Upload ── label photo
        ├─ extract_label()        OCR → nutrition + ingredients
        ├─ analyze_ingredients()  allergens + additives
        ├─ classify_diet()        vegan / vegetarian / keto / diabetic
        ├─ compute_health_score() 0–100, A–F
        └─ save_image() / save_record()
        │
Results ── allergens, additives, health card, diet cards, nutrition + DRI charts
        │
Pairing ── smart_pair(): L2 gap analysis + Pareto ranking of complementary products
        │
Feedback ── will_buy? paired product? → save_behavior()
```

**Health score (100 points)**

| Section | Points | Measures |
|---|---|---|
| Ingredient quality | 30 | Red-flag ingredients in top positions, positive ingredients |
| Macronutrient balance | 30 | Sugar, saturated fat, fibre, protein vs your DRI |
| Sodium | 20 | Share of your daily sodium AI |
| Additive risk | 20 | benign −0 · caution −3 · avoid −8 |

### Business flow

```
Register (Business) ── no body profile needed

Upload ── 7-step pipeline
  1   preprocess_label_image()   OpenCV: deskew + CLAHE + unsharp mask
  1   load_qwen_model()          4-bit Qwen2.5-VL-7B (cached for app lifetime)
  1   run_ocr()                  main pass → structured text
  1b  focused_pass()             second pass if recycling or company info missed
  1c  run_json_ocr()             4 JSON prompts: company / recycling / claims / allergens
  2   extract_nutrition()        regex parser → NutritionData
  3   extract_label_fields()     regex parser for all non-nutrition fields
      merge_json_into_fields()   JSON OCR wins where confidence is higher
      _validate_company_fields() strips hallucinated values
  4   detect_all_symbols()       recycling marks + certifications
  5   translate_ingredients()    Japanese → English dictionary
  6   classify_health()          rule-based threshold scoring
  7   run_compliance_check() · detect_category() · benchmark()
      generate_suggestions() · compute_readiness()

Results
  Market Readiness card  →  nutrition, allergens, ingredients, label info, symbols
  Tab 1  Compliance Validator       per-standard badges + error/warning/info cards
  Tab 2  Competitor Benchmarking    pick a category → benchmark and grade update live
  Tab 3  Improvement Suggestions    grouped High / Medium / Low
```

**Market Readiness Score (100 points)**

| Component | Points | Basis |
|---|---|---|
| Regulatory compliance | 50 | −8 per error, −3 per warning |
| Competitive position | 30 | Percentile rank vs category peers |
| Label quality | 20 | −5 per high-priority issue, −2 per medium |

Grades: **A** ≥ 90 · **B** ≥ 75 · **C** ≥ 60 · **D** ≥ 45 · **F** < 45

---

## Project structure

```
japan-food-label-analyzer/
├── app.py                          # Entry point — auth, nav bar, role router
├── requirements.txt
├── .env.example
├── .streamlit/config.toml          # Light theme
│
├── shared/
│   ├── auth.py                     # register / login / profile update
│   └── storage.py                  # save_image, save_record, save_behavior
│
├── consumer/
│   ├── pages.py                    # Upload, Results, Pairing, Feedback
│   └── services/
│       ├── ocr.py                  # Qwen2.5-VL label extraction
│       ├── detection.py            # Allergen + additive detection
│       ├── diet.py                 # Vegan / vegetarian / keto / diabetic
│       ├── dri.py                  # DRI calculator
│       ├── health_score.py         # 100-point health score
│       └── smart_pairing.py        # L2 + Pareto pairing
│
├── business/
│   ├── pages.py                    # Upload page + 3-tab results page
│   └── services/
│       ├── ocr.py                  # Qwen2.5-VL pipeline + preprocessing
│       ├── label_field_extractor.py# Non-nutrition field parser
│       ├── extractor.py            # Nutrition parser
│       ├── translator.py           # JP → EN ingredient dictionary
│       ├── classifier.py           # Rule-based health classification
│       ├── compliance.py           # JAS / CAA checks
│       ├── benchmarking.py         # Per-100 g competitor comparison
│       ├── suggestions.py          # Groq suggestions + fallback
│       ├── readiness.py            # Market Readiness Score
│       └── symbol_detector.py      # Recycling / certification symbols
│
├── data/                           # Reference data (committed) + runtime data (git-ignored)
└── shared_data/                    # users.csv — created on first run (git-ignored)
```

---

## Data files

Reference data (committed):

| File | Contents | Used by |
|---|---|---|
| `allergen_master.csv` | 35 allergens — 8 mandatory, 20 recommended, 7 global-only | `detection.py` |
| `allergen_tokens.csv` | Japanese / English token → allergen id | `detection.py` |
| `ingredient_sources.csv` | Ingredient → allergen id (supplementary) | `detection.py` |
| `japanese_food_additives.csv` | 351 additives with safety notes | `detection.py` |
| `food_products_cleaned.csv` | ~75 Japanese products for pairing | `smart_pairing.py` |
| `diet_blocklists.json` | Ingredient lists for diet classification | `diet.py` |
| `benchmark_products.json` | 50 competitor products, 10 categories, per-100 g | `benchmarking.py` |

Runtime data (auto-created, **git-ignored** because it contains user data): `shared_data/users.csv`, `data/scan_records.csv`, `data/purchase_behavior.csv`, `data/images/`.

<details>
<summary>Benchmark categories (5 products each)</summary>

| Key | Category | Example brands |
|---|---|---|
| `instant_noodles` | Instant Noodles & Ramen | Nissin, Maruchan, Myojo, Acecook |
| `potato_snacks` | Potato & Corn Snacks | Calbee, Tohato |
| `chocolate` | Chocolate & Confectionery | Meiji, Lotte, Glico |
| `beverages` | RTD Beverages | Ito En, Kirin, Suntory, Coca-Cola |
| `instant_beverages` | Instant Coffee & Powder Drinks | AGF Blendy, Nescafé, UCC |
| `rice_crackers` | Rice Crackers & Senbei | Kameda, Iwatsuka, Bourbon |
| `packaged_meals` | Packaged / Retort Meals | House, S&B, Otsuka, Ajinomoto |
| `supplements` | Protein & Supplement Bars | Otsuka, Morinaga, DHC, Asahi |
| `soft_candy` | Soft Candy & Gummies | HI-CHEW, Kasugai, UHA, Nobel, Meiji |
| `soup` | Soup & Miso Soup | Marukome, Hanamaruki, Knorr, Pokka |

Values come from Open Food Facts barcode entries and official Japanese brand nutrition pages.
</details>

---

## Module reference

<details>
<summary><b>shared/</b></summary>

| Function | Description |
|---|---|
| `register_user(...)` | Validates input, checks uniqueness, hashes password, appends to CSV |
| `login_user(username, password)` | Hash comparison; returns the user dict including role |
| `update_user_profile(user_id, updates)` | Rewrites the CSV with one row updated |
| `get_user_by_id(user_id)` | Returns the user dict or `None` |
| `save_image(source_path)` | Copies to `data/images/` under a UUID filename |
| `save_record(...)` | Appends a scan row to `scan_records.csv`, returns `scan_id` |
| `save_behavior(...)` | Appends a purchase-intent row, returns `behavior_id` |
</details>

<details>
<summary><b>consumer/services/</b></summary>

| File | Key function | Purpose |
|---|---|---|
| `detection.py` | `analyze_ingredients(text)` | Allergen + additive lists via token matching |
| `diet.py` | `classify_diet(ingredients, nutrition, allergens)` | Four diet verdicts, each with reasons |
| `dri.py` | `calculate_dri(sex, age, weight_kg, height_cm, activity, ...)` | EER → full DRI profile |
| `health_score.py` | `compute_health_score(ingredients, nutrition, additives, dri_raw)` | 100-point score with grade, flags, boosts |
| `smart_pairing.py` | `smart_pair(...)` | Gap scoring + Pareto ranking of complementary products |
</details>

<details>
<summary><b>business/services/</b></summary>

| File | Key function | Purpose |
|---|---|---|
| `ocr.py` | `load_qwen_model()` | Loads Qwen2.5-VL-7B (NF4), cached |
| `ocr.py` | `run_ocr()` / `run_json_ocr()` | Main text pass / four structured-JSON passes |
| `ocr.py` | `merge_json_into_fields()` | Merges JSON OCR into text OCR |
| `label_field_extractor.py` | `extract_label_fields(raw_text)` | Product name, manufacturer, expiry, claims, warnings, recycling |
| `extractor.py` | `extract_nutrition(raw_text)` | Parses 栄養成分表示 into `NutritionData` |
| `compliance.py` | `run_compliance_check(nutrition_dict)` | JAS + CAA checks → `ComplianceReport`s |
| `benchmarking.py` | `benchmark()` / `detect_category()` | Percentile vs peers; category detection |
| `suggestions.py` | `generate_suggestions(...)` | Groq LLM with rule-based fallback |
| `readiness.py` | `compute_readiness(...)` | A–F grade and 0–100 score |
</details>

---

## Known limitations

- **GPU required.** Both roles load a 7B vision-language model; CPU-only machines are not practical.
- **No native Windows support.** `shared/auth.py` and `shared/storage.py` import `fcntl` (Unix only). Run under WSL, or swap in `msvcrt` / a database.
- **Weak password hashing.** Passwords use unsalted SHA-256. Replace with bcrypt or Argon2 before any real deployment.
- **CSV storage.** Fine for a single machine and a handful of users; move to SQLite/PostgreSQL for anything more.
- **Business latency.** The 7-step pipeline takes 30–90 s per scan, and concurrent scans queue behind the single cached model. For multiple users, serve Qwen separately (e.g. vLLM + a task queue).
- **Groq rate limits.** On the free tier, `generate_suggestions()` silently falls back to rule-based suggestions.
- **OCR quality depends on the photo.** Use good lighting, hold the camera parallel to the label, and keep the nutrition table fully in frame.
- **Not legal or medical advice.** Compliance and health outputs are decision aids, not certifications.

---

## License

Released under the [MIT License](LICENSE). Reference data keeps its original terms (see [Acknowledgements](#acknowledgements)).

## Acknowledgements

- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) — Alibaba Cloud
- [Open Food Facts](https://world.openfoodfacts.org) — competitor nutrition data
- [Groq](https://groq.com) — fast LLM inference
- USDA / National Academies — Dietary Reference Intake tables
- Japan Consumer Affairs Agency — Food Labelling Standards (食品表示基準)
