"""CSV and image storage for scan records, purchase behavior and label images.
Files live under data/ at the project root."""

import csv
import uuid
import shutil
import fcntl
from datetime import datetime
from pathlib import Path

# data/ sits at the project root, one level above shared/
DATA_DIR   = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"

RECORDS_FILE  = DATA_DIR / "scan_records.csv"
BEHAVIOR_FILE = DATA_DIR / "purchase_behavior.csv"

SCAN_FIELDS = [
    "scan_id", "user_id", "username", "user_allergen",
    "dri_sex", "dri_age", "dri_height_cm", "dri_weight_kg",
    "dri_activity", "dri_bmi", "dri_eer_kcal",
    "image_path", "product_name", "ingredients",
    "detected_allergens", "detected_additives",
    "health_score", "health_grade", "health_verdict",
    "timestamp",
]

BEHAVIOR_FIELDS = [
    "behavior_id", "scan_id", "user_id", "product_name",
    "will_buy",                   # yes / no
    "paired_product_selected",    # yes / no / na
    "paired_product_name",
    "paired_product_id",
    "timestamp",
]


def _ensure(path: Path, fields: list[str]):
    """Create CSV with headers if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _append(path: Path, fields: list[str], row: dict):
    """Append a single row to a CSV file with file locking."""
    _ensure(path, fields)
    with open(path, "a", newline="", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read(path: Path, fields: list[str]) -> list[dict]:
    """Read all rows from a CSV file."""
    _ensure(path, fields)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_image(source_path: str) -> str:
    """Copy a label image into data/images/ under a unique name and return its path."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ext  = Path(source_path).suffix or ".jpg"
    dest = IMAGES_DIR / f"{uuid.uuid4().hex}{ext}"
    shutil.copy2(source_path, dest)
    return str(dest)


def save_record(
    user_id:            str,
    username:           str,
    user_allergen:      str,
    image_path:         str,
    product_name:       str,
    ingredients:        str,
    detected_allergens: list,
    detected_additives: list,
    health_score:       int  = 0,
    health_grade:       str  = "",
    health_verdict:     str  = "",
    dri:                dict | None = None,
) -> str:
    """Save a scan result to scan_records.csv and return the new scan_id."""
    scan_id = uuid.uuid4().hex[:12].upper()
    dri_inp = (dri or {}).get("inputs") or {}

    row = {
        "scan_id":            scan_id,
        "user_id":            user_id,
        "username":           username,
        "user_allergen":      user_allergen or "",
        "dri_sex":            dri_inp.get("sex", ""),
        "dri_age":            dri_inp.get("age", ""),
        "dri_height_cm":      dri_inp.get("height_cm", ""),
        "dri_weight_kg":      dri_inp.get("weight_kg", ""),
        "dri_activity":       dri_inp.get("activity", ""),
        "dri_bmi":            (dri or {}).get("bmi", ""),
        "dri_eer_kcal":       (dri or {}).get("eer", ""),
        "image_path":         image_path,
        "product_name":       product_name,
        "ingredients":        ingredients,
        "detected_allergens": "; ".join(detected_allergens),
        "detected_additives": "; ".join(
            f"{a}:{n}" for a, n in detected_additives
        ),
        "health_score":       health_score,
        "health_grade":       health_grade,
        "health_verdict":     health_verdict,
        "timestamp":          datetime.now().isoformat(timespec="seconds"),
    }
    _append(RECORDS_FILE, SCAN_FIELDS, row)
    return scan_id


def get_records_for_user(user_id: str) -> list[dict]:
    """Return all scan records for a given user_id."""
    return [
        r for r in _read(RECORDS_FILE, SCAN_FIELDS)
        if r["user_id"] == user_id
    ]


def save_behavior(
    scan_id:                  str,
    user_id:                  str,
    product_name:             str,
    will_buy:                 bool,
    paired_product_selected:  bool | None = None,
    paired_product_name:      str  = "",
    paired_product_id:        str  = "",
) -> str:
    """Save a purchase-intent event to purchase_behavior.csv and return the new behavior_id."""
    bid        = uuid.uuid4().hex[:10].upper()
    paired_sel = (
        ("yes" if paired_product_selected else "no") if will_buy else "na"
    )
    row = {
        "behavior_id":             bid,
        "scan_id":                 scan_id,
        "user_id":                 user_id,
        "product_name":            product_name,
        "will_buy":                "yes" if will_buy else "no",
        "paired_product_selected": paired_sel,
        "paired_product_name":     paired_product_name,
        "paired_product_id":       str(paired_product_id),
        "timestamp":               datetime.now().isoformat(timespec="seconds"),
    }
    _append(BEHAVIOR_FILE, BEHAVIOR_FIELDS, row)
    return bid


def get_behavior_for_user(user_id: str) -> list[dict]:
    """Return all purchase behavior rows for a given user_id."""
    return [
        b for b in _read(BEHAVIOR_FILE, BEHAVIOR_FIELDS)
        if b["user_id"] == user_id
    ]