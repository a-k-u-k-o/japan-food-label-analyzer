"""Streamlit entry point for the Japan Food App: login/register, then routing by user role.
Consumers go to consumer.pages and businesses to business.pages."""

import re
import streamlit as st
from shared.auth import register_user, login_user

# Page config
st.set_page_config(
    page_title="🍱 Japan Food App",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1100px; }

/* ── Auth page ── */
.auth-header {
  text-align: center; padding: 2.5rem 0 1.5rem;
}
.auth-header .brand {
  font-size: 32px; font-weight: 700; color: #0f172a; letter-spacing: -0.03em;
}
.auth-header .brand span { color: #3b82f6; }
.auth-header .sub {
  font-size: 14px; color: #64748b; margin-top: 6px;
}

/* ── Role selector cards ── */
.role-cards {
  display: flex; gap: 14px; margin: 16px 0;
}
.role-card {
  flex: 1; border: 2px solid #e2e8f0; border-radius: 14px;
  padding: 18px 16px; cursor: pointer; transition: all .15s;
  background: #fff;
}
.role-card:hover { border-color: #3b82f6; background: #f0f7ff; }
.role-card.selected { border-color: #3b82f6; background: #eff6ff; }
.role-card .icon  { font-size: 26px; margin-bottom: 8px; }
.role-card .title { font-size: 14px; font-weight: 700; color: #0f172a; }
.role-card .desc  { font-size: 12px; color: #64748b; margin-top: 3px; line-height: 1.5; }

/* ── Nav bar (shared) ── */
.top-nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 0 18px 0; border-bottom: 1.5px solid #e8edf5;
  margin-bottom: 28px;
}
.nav-brand { font-size: 20px; font-weight: 700; color: #0f172a; letter-spacing: -0.02em; }
.nav-brand span { color: #3b82f6; }
.nav-role-badge {
  font-size: 11px; font-weight: 600; padding: 3px 12px; border-radius: 20px;
  letter-spacing: .04em; text-transform: uppercase;
}
.badge-consumer { background: #e0f2fe; color: #0369a1; }
.badge-business { background: #ede9fe; color: #6d28d9; }
.nav-user { font-size: 13px; color: #64748b; font-weight: 500; }
.nav-user b { color: #0f172a; }

/* ── Page titles ── */
.page-title {
  font-size: 28px; font-weight: 700; color: #0f172a;
  letter-spacing: -0.03em; margin-bottom: 4px;
}
.page-sub { font-size: 14px; color: #64748b; margin-bottom: 28px; }

/* ── Pills / chips ── */
.pill { display:inline-block; border-radius:20px; padding:3px 13px; margin:3px; font-size:13px; font-weight:500; }
.pill-red    { background:#fde8e8; color:#c0392b; }
.pill-yellow { background:#fef9e2; color:#9a6600; }
.pill-orange { background:#fef0e0; color:#c05000; }
.pill-green  { background:#e6f9ee; color:#1a6e3a; }
.chip { display:inline-block; background:#f1f5f9; color:#334155; border-radius:6px; padding:2px 9px; margin:2px; font-size:12.5px; }

/* ── Metrics ── */
[data-testid="stMetricValue"] { font-size:26px !important; font-weight:700 !important; }
[data-testid="stMetricLabel"] { font-size:12px !important; color:#64748b !important; }

/* ── Section header ── */
.sec-hd {
  font-size:15px; font-weight:700; color:#0f172a; letter-spacing:.01em;
  margin:20px 0 8px 0; border-left: 3px solid #3b82f6; padding-left: 10px;
}

/* ── Consumer health card ── */
.health-card { border-radius:16px; padding:20px 24px; margin-bottom:16px; display:flex; align-items:center; gap:20px; border:2px solid; }
.health-healthy   { background:#f0fdf4; border-color:#22c55e; }
.health-moderate  { background:#fffbeb; border-color:#f59e0b; }
.health-unhealthy { background:#fff1f2; border-color:#ef4444; }
.health-grade { font-size:48px; font-weight:900; line-height:1; min-width:56px; text-align:center; }
.health-healthy   .health-grade { color:#16a34a; }
.health-moderate  .health-grade { color:#d97706; }
.health-unhealthy .health-grade { color:#dc2626; }
.health-body { flex:1; }
.health-verdict { font-size:18px; font-weight:700; margin-bottom:2px; }
.health-score-line { font-size:13px; color:#64748b; margin-bottom:8px; }
.health-bar-wrap { background:#e2e8f0; border-radius:99px; height:8px; overflow:hidden; margin-bottom:10px; }
.health-bar { height:8px; border-radius:99px; }
.health-healthy   .health-bar { background:linear-gradient(90deg,#22c55e,#16a34a); }
.health-moderate  .health-bar { background:linear-gradient(90deg,#fbbf24,#d97706); }
.health-unhealthy .health-bar { background:linear-gradient(90deg,#f87171,#dc2626); }
.health-tags { display:flex; flex-wrap:wrap; gap:6px; }
.htag { font-size:11px; padding:2px 9px; border-radius:10px; font-weight:600; }
.htag-neg { background:#fee2e2; color:#991b1b; }
.htag-pos { background:#dcfce7; color:#166534; }

/* ── Consumer diet cards ── */
.diet-card { border-radius:14px; padding:14px 18px; margin-bottom:10px; border-left:5px solid; }
.diet-yes       { background:#f0fdf4; border-color:#22c55e; }
.diet-no        { background:#fff1f2; border-color:#ef4444; }
.diet-caution   { background:#fffbeb; border-color:#f59e0b; }
.diet-uncertain { background:#f0f4ff; border-color:#6366f1; }
.diet-icon  { font-size:20px; margin-right:8px; vertical-align:middle; }
.diet-title { font-size:14px; font-weight:700; vertical-align:middle; }
.diet-label { font-size:11px; font-weight:600; margin-left:6px; padding:1px 8px; border-radius:10px; vertical-align:middle; }
.diet-yes     .diet-label { background:#dcfce7; color:#166534; }
.diet-no      .diet-label { background:#fee2e2; color:#991b1b; }
.diet-caution .diet-label { background:#fef3c7; color:#92400e; }
.diet-uncertain .diet-label { background:#e0e7ff; color:#3730a3; }
.diet-reason { font-size:12px; color:#64748b; margin-top:4px; padding-left:30px; }

/* ── Consumer pairing cards ── */
.pair-card { border-radius:14px; padding:16px 20px; margin-bottom:12px; background:#f8faff; border:1.5px solid #c7d5f5; }
.pair-rank { font-size:20px; font-weight:900; color:#3b82f6; margin-right:8px; }
.pair-name { font-size:15px; font-weight:700; color:#0f172a; }
.pair-cat  { font-size:11px; color:#64748b; background:#e8f0fe; border-radius:8px; padding:2px 8px; margin-left:8px; }
.pair-bar-wrap { background:#e2e8f0; border-radius:99px; height:6px; overflow:hidden; margin:8px 0; }
.pair-bar  { height:6px; border-radius:99px; background:linear-gradient(90deg,#6366f1,#3b82f6); }

/* ── Consumer feedback ── */
.buy-card { border-radius:16px; padding:28px 32px; background:#f8faff; border:2px solid #c7d5f5; margin-top:16px; }
.buy-q { font-size:20px; font-weight:700; color:#0f172a; margin-bottom:18px; }

/* ── Business: market readiness card ── */
.readiness-card {
  border-radius: 16px; padding: 24px 28px; margin-bottom: 24px;
  display: flex; align-items: center; gap: 24px;
  border: 2px solid;
}
.readiness-A { background:#f0fdf4; border-color:#22c55e; }
.readiness-B { background:#f0fdf4; border-color:#86efac; }
.readiness-C { background:#fffbeb; border-color:#f59e0b; }
.readiness-D { background:#fff7ed; border-color:#fb923c; }
.readiness-F { background:#fff1f2; border-color:#ef4444; }
.readiness-grade {
  font-size: 56px; font-weight: 900; line-height: 1;
  min-width: 64px; text-align: center;
}
.readiness-A .readiness-grade,
.readiness-B .readiness-grade { color: #16a34a; }
.readiness-C .readiness-grade { color: #d97706; }
.readiness-D .readiness-grade { color: #ea580c; }
.readiness-F .readiness-grade { color: #dc2626; }
.readiness-body { flex: 1; }
.readiness-headline { font-size: 20px; font-weight: 700; color: #0f172a; margin-bottom: 2px; }
.readiness-sub { font-size: 13px; color: #64748b; margin-bottom: 10px; }
.readiness-bar-wrap { background:#e2e8f0; border-radius:99px; height:8px; overflow:hidden; margin-bottom:10px; }
.readiness-bar { height:8px; border-radius:99px; background:linear-gradient(90deg,#3b82f6,#6366f1); }
.readiness-actions { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.readiness-action {
  font-size: 11px; padding: 3px 10px; border-radius: 10px; font-weight: 600;
  background: #fee2e2; color: #991b1b;
}

/* ── Business: compliance issues ── */
.issue-error   { background:#fff1f2; border-left:4px solid #ef4444; padding:10px 14px; border-radius:0 8px 8px 0; margin:6px 0; font-size:13px; }
.issue-warning { background:#fffbeb; border-left:4px solid #f59e0b; padding:10px 14px; border-radius:0 8px 8px 0; margin:6px 0; font-size:13px; }
.issue-info    { background:#eff6ff; border-left:4px solid #3b82f6; padding:10px 14px; border-radius:0 8px 8px 0; margin:6px 0; font-size:13px; }
.score-badge { display:inline-block; min-width:42px; padding:4px 10px; border-radius:20px; font-size:13px; font-weight:700; text-align:center; }

/* ── Business: suggestions ── */
.sug-card { border-radius:12px; padding:16px 18px; margin-bottom:10px; border:1.5px solid; }
.sug-card-high   { background:#fff1f2; border-color:#fca5a5; }
.sug-card-medium { background:#fffbeb; border-color:#fcd34d; }
.sug-card-low    { background:#f0f9ff; border-color:#bae6fd; }
.sug-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }
.sug-title  { font-size:14px; font-weight:700; color:#0f172a; }
.sug-cat-tag { font-size:11px; background:#f1f5f9; color:#475569; border-radius:6px; padding:2px 8px; }
.sug-detail { font-size:13px; color:#334155; margin-bottom:8px; }
.sug-meta   { display:flex; gap:12px; flex-wrap:wrap; }
.sug-impact { font-size:12px; color:#16a34a; font-weight:600; }
.sug-how    { font-size:12px; color:#64748b; }

/* ── Business: benchmarking ── */
.bench-table { width:100%; border-collapse:collapse; font-size:.84rem; }
.bench-table th { font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; color:#64748b; border-bottom:2px solid #0f172a; padding:.55rem .7rem; text-align:left; }
.bench-table td { padding:.5rem .7rem; border-bottom:1px solid #e2e8f0; }
.bench-table tr:last-child td { border-bottom:none; }
.bench-table tr:hover td { background:#f8fafc; }
.pending-note { background:#fffbeb; border:1px solid #fcd34d; border-radius:10px; padding:12px 16px; font-size:13px; margin-bottom:14px; }

/* ── Buttons ── */
.stButton > button {
  border-radius: 10px !important; font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ── Validation hint text ── */
.field-hint {
  font-size: 11px; color: #94a3b8; margin-top: -12px; margin-bottom: 8px;
}
.field-hint-error {
  font-size: 11px; color: #ef4444; margin-top: -12px; margin-bottom: 8px; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


def validate_username(username: str) -> list[str]:
    """Check a username (3-20 chars, lowercase letters/digits/_/., at least one letter).
    Returns a list of error strings, empty if valid."""
    errors = []
    if not username:
        errors.append("Username is required.")
        return errors
    if len(username) < 3:
        errors.append("Username must be at least 3 characters.")
    if len(username) > 20:
        errors.append("Username cannot exceed 20 characters.")
    if " " in username:
        errors.append("Username cannot contain spaces.")
    if re.search(r"[A-Z]", username):
        errors.append("Username must be lowercase only — no uppercase letters.")
    if not re.match(r"^[a-z0-9_.]+$", username):
        errors.append("Username can only contain lowercase letters, numbers, _ and .")
    if not re.search(r"[a-z]", username):
        errors.append("Username must contain at least one letter (a–z) — not numbers only.")
    return errors


def validate_display_name(name: str) -> list[str]:
    """Check a display name (2-50 chars after trimming, letters and spaces only).
    Returns a list of error strings, empty if valid."""
    errors = []
    name = name.strip()
    if not name:
        errors.append("Full name is required.")
        return errors
    if len(name) < 2:
        errors.append("Full name must be at least 2 characters.")
    if len(name) > 50:
        errors.append("Full name cannot exceed 50 characters.")
    # Allow only Unicode letters and spaces -- reject digits and any symbol
    if not re.match(r"^[A-Za-zÀ-ɏ\s]+$", name):
        errors.append("Full name can only contain letters and spaces -- no numbers or symbols.")
    return errors

def validate_password(password: str) -> list[str]:
    """Check a password (8+ chars with upper, lower, digit and special character).
    Returns a list of error strings, empty if valid."""
    errors = []
    if not password:
        errors.append("Password is required.")
        return errors
    if password.strip() == "":
        errors.append("Password cannot be whitespace only.")
        return errors
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter (A–Z).")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter (a–z).")
    if not re.search(r"\d", password):
        errors.append("Password must contain at least one number (0–9).")
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?`~]", password):
        errors.append("Password must contain at least one special character (!@#$%^&* …).")
    return errors


def password_strength_label(password: str) -> tuple[str, str]:
    """Return a (label, colour) pair rating password strength, for the live hint."""
    if not password:
        return "", "#94a3b8"
    score = 0
    if len(password) >= 8:  score += 1
    if len(password) >= 12: score += 1
    if re.search(r"[A-Z]", password): score += 1
    if re.search(r"[a-z]", password): score += 1
    if re.search(r"\d",    password): score += 1
    if re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?`~]", password): score += 1

    if score <= 2:   return "Weak",   "#ef4444"
    if score <= 4:   return "Fair",   "#f59e0b"
    if score == 5:   return "Good",   "#3b82f6"
    return               "Strong", "#22c55e"


# Session state defaults
_SS_DEFAULTS = {
    "user":           None,    # full user dict from users.csv (includes user_role)
    "page":           "auth",  # current page within consumer or business flow
    "dri":            None,    # consumer: calculated DRI
    "scan_result":    None,    # consumer: last scan analysis result
    "scan_id":        None,    # consumer: last saved scan_id
    "pairing_result": None,    # consumer: smart pairing result
    "_last_sig":      "",
    "_current_sig":   "",
    # Register form persistence
    "reg_sex":        "Female",
    "reg_age":        25,
    "reg_height":     165.0,
    "reg_weight":     60.0,
    "reg_activity":   "Active",
    "reg_pregnancy":  "None",
    "reg_role":       "consumer",
}
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _nav():
    """Top navigation bar shown on every page after login."""
    user = st.session_state.user
    role = (user or {}).get("user_role", "consumer")
    name = (user or {}).get("name") or (user or {}).get("username", "")

    role_badge = (
        '<span class="nav-role-badge badge-consumer">Consumer</span>'
        if role == "consumer"
        else '<span class="nav-role-badge badge-business">Business</span>'
    )
    brand_icon  = "🍱" if role == "consumer" else "🇯🇵"
    brand_label = "Food Label Scanner" if role == "consumer" else "Market Readiness"

    nav_left, nav_mid, nav_right = st.columns([3, 6, 2])

    with nav_left:
        st.markdown(
            f'<div class="nav-brand" style="padding-top:6px;">'
            f'{brand_icon} <span>{brand_label}</span></div>',
            unsafe_allow_html=True,
        )

    with nav_mid:
        if user:
            st.markdown(
                f'<div style="padding-top:6px;text-align:center;">'
                f'<span class="nav-user">👤 <b>{name}</b> {role_badge}</span></div>',
                unsafe_allow_html=True,
            )

    with nav_right:
        if user:
            if st.button("Logout", key="nav_logout", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

    st.markdown(
        '<hr style="margin:0 0 24px 0;border:none;border-top:1.5px solid #e8edf5;">',
        unsafe_allow_html=True,
    )


def page_auth():
    """Render the login and register tabs and handle both submissions."""
    st.markdown("""
    <div class="auth-header">
      <div class="brand">🍱 <span>Japan</span> Food App</div>
      <div class="sub">Scan Japanese food labels — powered by AI</div>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_reg = st.tabs(["🔑 Login", "✨ Register"])

    with tab_login:
        with st.form("login_form"):
            uname = st.text_input("Username")
            pwd   = st.text_input("Password", type="password")
            sub   = st.form_submit_button(
                "Sign In", type="primary", use_container_width=True
            )

        if sub:
            login_errors = []
            if not uname.strip():
                login_errors.append("Username is required.")
            if not pwd:
                login_errors.append("Password is required.")

            if login_errors:
                for e in login_errors:
                    st.error(e)
            else:
                res = login_user(uname.strip().lower(), pwd)
                if res["ok"]:
                    st.session_state.user = res["user"]
                    role = res["user"].get("user_role", "consumer")

                    # rebuild the DRI from the saved profile
                    if role == "consumer":
                        u = res["user"]
                        try:
                            from consumer.services.dri import calculate_dri
                            if u.get("sex") and int(u.get("age", 0)) > 0:
                                preg      = u.get("pregnancy", "None") or "None"
                                pregnancy = preg if "trimester" in preg else "None"
                                lactation = (
                                    preg.replace("Breastfeeding ", "")
                                    if "Breastfeeding" in preg else "None"
                                )
                                st.session_state.dri = calculate_dri(
                                    sex=u["sex"].lower(),
                                    age=int(u["age"]),
                                    weight_kg=float(u["weight_kg"]),
                                    height_cm=float(u["height_cm"]),
                                    activity=u["activity"],
                                    pregnancy=pregnancy,
                                    lactation=lactation,
                                )
                        except (ValueError, TypeError, KeyError):
                            pass
                        st.session_state.page = "upload"
                    else:
                        st.session_state.page = "biz_upload"

                    st.rerun()
                else:
                    st.error(res["error"])

    with tab_reg:

        st.markdown("##### Choose your account type")
        role_col1, role_col2 = st.columns(2)
        with role_col1:
            consumer_selected = st.session_state.reg_role == "consumer"
            if st.button(
                "👤  Consumer\n\nPersonal nutrition analysis,\nallergen checks & health scores",
                use_container_width=True,
                type="primary" if consumer_selected else "secondary",
                key="role_btn_consumer",
            ):
                st.session_state.reg_role = "consumer"
                st.rerun()
        with role_col2:
            business_selected = st.session_state.reg_role == "business"
            if st.button(
                "🏢  Business\n\nCompliance validation,\ncompetitor benchmarking & insights",
                use_container_width=True,
                type="primary" if business_selected else "secondary",
                key="role_btn_business",
            ):
                st.session_state.reg_role = "business"
                st.rerun()

        selected_role = st.session_state.reg_role
        st.markdown("---")

        st.markdown("##### Personal Details")

        rc1, rc2 = st.columns(2)
        with rc1:
            r_uname = st.text_input(
                "Username *",
                key="r_uname",
                placeholder="e.g. john_doe · lowercase only",
                max_chars=20,
            )
            if r_uname:
                uname_errs = validate_username(r_uname)
                if uname_errs:
                    for e in uname_errs:
                        st.markdown(f'<p class="field-hint-error">⚠ {e}</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="field-hint">✓ Username looks good</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="field-hint">Lowercase letters, numbers, _ and . only · Must include at least one letter</p>', unsafe_allow_html=True)

        with rc2:
            if selected_role == 'consumer':
                r_name = st.text_input(
                    "Full Name *",
                    key="r_name",
                    placeholder="e.g. John Doe · letters only",
                    max_chars=50,
            )
            else:
                r_name = st.text_input(
                    "Business Name *",
                    key="r_name",
                    placeholder="e.g. John Doe · letters only",
                    max_chars=50,
            )
            if r_name:
                name_errs = validate_display_name(r_name)
                if name_errs:
                    for e in name_errs:
                        st.markdown(f'<p class="field-hint-error">⚠ {e}</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="field-hint">✓ Full name looks good</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="field-hint">Letters and spaces only · No numbers or symbols</p>', unsafe_allow_html=True)

        rp1, rp2 = st.columns(2)
        with rp1:
            r_pwd = st.text_input(
                "Password *",
                type="password",
                key="r_pwd",
                placeholder="Min 8 chars with upper, lower, number, symbol",
            )
            if r_pwd:
                pwd_errs = validate_password(r_pwd)
                strength_label, strength_color = password_strength_label(r_pwd)
                if pwd_errs:
                    for e in pwd_errs:
                        st.markdown(f'<p class="field-hint-error">⚠ {e}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<p class="field-hint" style="color:{strength_color};font-weight:600;">'
                        f'● Strength: {strength_label}</p>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<p class="field-hint">'
                    'Must have uppercase · lowercase · number · special char'
                    '</p>',
                    unsafe_allow_html=True,
                )

        with rp2:
            r_pwd2 = st.text_input(
                "Confirm Password *",
                type="password",
                key="r_pwd2",
                placeholder="Re-enter your password",
            )
            if r_pwd2:
                if r_pwd2 == r_pwd:
                    st.markdown('<p class="field-hint" style="color:#22c55e;font-weight:600;">✓ Passwords match</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="field-hint-error">⚠ Passwords do not match</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="field-hint">Must match the password above</p>', unsafe_allow_html=True)

        if selected_role == "consumer":
            r_allergen = st.text_input(
                "Known allergen (optional)",
                placeholder="e.g. peanut, milk",
                key="r_allergen",
            )

            st.markdown("##### Body & Activity  *(used for personal nutrition targets)*")
            bc1, bc2 = st.columns(2)
            with bc1:
                st.session_state.reg_sex = st.selectbox(
                    "Sex", ["Female", "Male"],
                    index=["Female", "Male"].index(st.session_state.reg_sex),
                    key="rs_sex",
                )
            with bc2:
                st.session_state.reg_age = st.number_input(
                    "Age (yr)", 5, 100,
                    value=st.session_state.reg_age, step=1, key="rs_age",
                )
            bc3, bc4 = st.columns(2)
            with bc3:
                st.session_state.reg_height = st.number_input(
                    "Height (cm)", 100.0, 250.0,
                    value=st.session_state.reg_height, step=0.5, key="rs_height",
                )
            with bc4:
                st.session_state.reg_weight = st.number_input(
                    "Weight (kg)", 20.0, 300.0,
                    value=st.session_state.reg_weight, step=0.5, key="rs_weight",
                )

            act_opts = ["Sedentary", "Low Active", "Active", "Very Active"]
            st.session_state.reg_activity = st.selectbox(
                "Activity Level", act_opts,
                index=act_opts.index(st.session_state.reg_activity), key="rs_act",
            )
            if st.session_state.reg_sex == "Female":
                preg_opts = [
                    "None", "1st trimester", "2nd trimester", "3rd trimester",
                    "Breastfeeding 0-6 months", "Breastfeeding 7-12 months",
                ]
                st.session_state.reg_pregnancy = st.selectbox(
                    "Pregnancy / Breastfeeding", preg_opts,
                    index=preg_opts.index(st.session_state.reg_pregnancy), key="rs_preg",
                )

        else:
            # business accounts have no allergen field
            r_allergen = ""

        if st.button(
            "Create Account", type="primary",
            use_container_width=True, key="reg_btn"
        ):
            all_errors = []

            uname_errs = validate_username(r_uname)
            all_errors.extend(uname_errs)

            name_errs = validate_display_name(r_name)
            all_errors.extend(name_errs)

            pwd_errs = validate_password(r_pwd)
            all_errors.extend(pwd_errs)

            if not r_pwd2:
                all_errors.append("Please confirm your password.")
            elif r_pwd != r_pwd2:
                all_errors.append("Passwords do not match.")

            if all_errors:
                for e in all_errors:
                    st.error(e)
            else:
                res = register_user(
                    username=r_uname.strip().lower(),
                    password=r_pwd,
                    name=r_name.strip(),
                    user_role=selected_role,
                    allergen=r_allergen,
                    sex=st.session_state.reg_sex,
                    age=int(st.session_state.reg_age),
                    height_cm=float(st.session_state.reg_height),
                    weight_kg=float(st.session_state.reg_weight),
                    activity=st.session_state.reg_activity,
                    pregnancy=st.session_state.reg_pregnancy,
                )
                if res["ok"]:
                    st.session_state.user = res["user"]

                    if selected_role == "consumer":
                        preg      = st.session_state.reg_pregnancy
                        pregnancy = preg if "trimester" in preg else "None"
                        lactation = (
                            preg.replace("Breastfeeding ", "")
                            if "Breastfeeding" in preg else "None"
                        )
                        try:
                            from consumer.services.dri import calculate_dri
                            st.session_state.dri = calculate_dri(
                                sex=st.session_state.reg_sex.lower(),
                                age=int(st.session_state.reg_age),
                                weight_kg=float(st.session_state.reg_weight),
                                height_cm=float(st.session_state.reg_height),
                                activity=st.session_state.reg_activity,
                                pregnancy=pregnancy,
                                lactation=lactation,
                            )
                        except (ValueError, TypeError):
                            pass
                        st.session_state.page = "upload"
                        st.success(f"✅ Account created! Welcome, {r_name.strip()}.")
                        st.rerun()

                    else:
                        st.session_state.page = "biz_upload"
                        st.success(f"✅ Business account created! Welcome, {r_name.strip()}.")
                        st.rerun()
                else:
                    st.error(res["error"])


# Router
user = st.session_state.user
page = st.session_state.page

if not user:
    if page != "auth":
        st.session_state.page = "auth"
    page_auth()

else:
    role = user.get("user_role", "consumer")

    if role == "business":
        from business.pages import route as business_route
        business_route(nav_fn=_nav)

    else:
        from consumer.pages import route as consumer_route
        consumer_route(nav_fn=_nav)