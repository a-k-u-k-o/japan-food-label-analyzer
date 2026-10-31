"""User registration and login for consumer and business roles.
Users are stored in shared_data/users.csv."""

import csv
import uuid
import hashlib
import fcntl
from datetime import datetime
from pathlib import Path

# shared_data/ sits at the project root, one level above this file
SHARED_DATA_DIR = Path(__file__).parent.parent / "shared_data"
USERS_FILE      = SHARED_DATA_DIR / "users.csv"

USER_FIELDS = [
    "user_id", "username", "password_hash", "name", "allergen",
    "sex", "age", "height_cm", "weight_kg", "activity", "pregnancy",
    "user_role",        # "consumer" or "business"
    "registered_at",
]

VALID_ROLES = {"consumer", "business"}


def _hash(plain: str) -> str:
    """Return the SHA-256 hex digest of a password."""
    return hashlib.sha256(plain.encode()).hexdigest()


def _ensure_file():
    """Create users.csv with headers if it doesn't exist yet."""
    SHARED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        with open(USERS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=USER_FIELDS).writeheader()


def _read_users() -> list[dict]:
    """Return all user rows from users.csv."""
    _ensure_file()
    with open(USERS_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_user(row: dict):
    """Append one user row to users.csv under an exclusive lock."""
    _ensure_file()
    with open(USERS_FILE, "a", newline="", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            csv.DictWriter(f, fieldnames=USER_FIELDS).writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _rewrite_users(rows: list[dict]):
    """Overwrite users.csv with the given rows under an exclusive lock."""
    _ensure_file()
    with open(USERS_FILE, "w", newline="", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            w = csv.DictWriter(f, fieldnames=USER_FIELDS)
            w.writeheader()
            w.writerows(rows)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def register_user(
    username:   str,
    password:   str,
    name:       str,
    user_role:  str,            # "consumer" or "business"
    allergen:   str  = "",
    sex:        str  = "",
    age:        int  = 0,
    height_cm:  float = 0.0,
    weight_kg:  float = 0.0,
    activity:   str  = "",
    pregnancy:  str  = "None",
) -> dict:
    """Register a new user and save them to users.csv.
    Returns {"ok": True, "user": ...} on success or {"ok": False, "error": ...}."""
    if not username.strip():
        return {"ok": False, "error": "Username cannot be blank."}
    if not password:
        return {"ok": False, "error": "Password cannot be blank."}
    if not name.strip():
        return {"ok": False, "error": "Full name cannot be blank."}
    if user_role not in VALID_ROLES:
        return {"ok": False, "error": f"Invalid role '{user_role}'. Must be 'consumer' or 'business'."}

    users = _read_users()
    if any(u["username"].lower() == username.strip().lower() for u in users):
        return {"ok": False, "error": "Username already taken. Please choose another."}

    user_id = uuid.uuid4().hex[:10].upper()
    row = {
        "user_id":       user_id,
        "username":      username.strip(),
        "password_hash": _hash(password),
        "name":          name.strip(),
        "allergen":      allergen.strip().lower(),
        "sex":           sex,
        "age":           age,
        "height_cm":     height_cm,
        "weight_kg":     weight_kg,
        "activity":      activity,
        "pregnancy":     pregnancy,
        "user_role":     user_role,
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }
    _append_user(row)
    return {"ok": True, "user_id": user_id, "user": row}


def login_user(username: str, password: str) -> dict:
    """Check a username and password against users.csv.
    Returns {"ok": True, "user": ...} on success or {"ok": False, "error": ...}."""
    ph = _hash(password)
    for u in _read_users():
        if (u["username"].lower() == username.strip().lower()
                and u["password_hash"] == ph):
            return {"ok": True, "user": u}
    return {"ok": False, "error": "Invalid username or password."}


def update_user_profile(user_id: str, updates: dict) -> bool:
    """Update fields for an existing user. Returns True on success."""
    users = _read_users()
    found = False
    for u in users:
        if u["user_id"] == user_id:
            u.update(updates)
            found = True
            break
    if not found:
        return False
    _rewrite_users(users)
    return True


def get_user_by_id(user_id: str) -> dict | None:
    """Return user dict by user_id, or None if not found."""
    for u in _read_users():
        if u["user_id"] == user_id:
            return u
    return None