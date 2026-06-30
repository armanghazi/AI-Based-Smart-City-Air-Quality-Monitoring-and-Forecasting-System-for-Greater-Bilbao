# streamlit_app.py (replaces app.py)
# Thin router only — no page body here.
# All content lives in pages/.

import streamlit as st
from i18n_auto import language_selector, apply_lang_styles
from pathlib import Path
import base64

# --------------------------------------------------
# Global page config (applies to every page)
# --------------------------------------------------
st.set_page_config(
    page_title="Bilbao Air Intelligence",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared i18n state — called here so language persists across pages
language_selector()
apply_lang_styles()

# --------------------------------------------------
# Page registry — explicit paths, no auto-discovery
# --------------------------------------------------
home       = st.Page("pages/Home.py",
                     title="Home",                        icon="🌬️",  default=True)
monitoring = st.Page("pages/1_Air_Quality_Monitoring.py",
                     title="Air Quality Monitoring",      icon="📡")
temporal   = st.Page("pages/2_Temporal_Trends.py",
                     title="Temporal Trends",             icon="📈")
forecast   = st.Page("pages/5_Forecasting.py",
                     title="Forecast Explorer",           icon="🔮")
spatial    = st.Page("pages/3_GeoAI_Spatial_Analysis.py",
                     title="Spatial Deep-Dive",           icon="🗺️")
weather    = st.Page("pages/4_Weather_Drivers.py",
                     title="Weather Drivers",             icon="💨")
decision   = st.Page("pages/6_Smart_City_Decision_Support.py",
                     title="Smart City Decision Support", icon="🏙️")
assistant  = st.Page("pages/8_Project_Assistant.py",
                     title="Project Assistant",           icon="🤖")
methods    = st.Page("pages/7_Scope_and_Limitations.py",
                     title="Methodology",                 icon="📖")
operations = st.Page("pages/9_Smart_City_Operations.py",
                     title="Smart City Operations",       icon="⚙️")

# --------------------------------------------------
# Navigation structure
# "" = header-less group at the top (Home only)
# Single-page sections (Forecasting) get a header but page title differs
# --------------------------------------------------
pages: dict = {
    "":                       [home],
    "Monitoring":             [monitoring, temporal],
    "Forecasting":            [forecast],
    "GeoAI Spatial Analysis": [spatial, weather, decision],
    "Project":                [assistant, methods],  # assistant برمی‌گردد
}



_logo = Path(__file__).parent / "static" / "geoai_logo.svg"
if _logo.exists():
    _svg = _logo.read_text(encoding="utf-8")
    _b64 = base64.b64encode(_svg.encode()).decode()
    st.sidebar.markdown(
        f'<div style="padding:0.8rem 0.5rem 1.4rem;">'
        f'<img src="data:image/svg+xml;base64,{_b64}" style="width:100%;max-width:3500px;display:block;">'
        f'</div>',
        unsafe_allow_html=True,
    )

# Admin cluster — pages absent from st.navigation cannot render at all,
# so this IS real access control (not just UI hiding).
# is_admin() is a safe wrapper that returns False on any auth error.

pages["Admin"] = [operations]

pg = st.navigation(pages)

# --------------------------------------------------
# Post-login redirect-back — return the user to the page that
# triggered st.login() instead of always landing on Home.
# Streamlit's docs confirm st.login() always redirects to the app's
# homepage in a brand-new session — session_state and query_params are
# both wiped in that round-trip. A real browser cookie (set in
# auth._login_and_remember(), same EncryptedCookieManager pattern used
# for the favourite-station cookie) is the only state that survives it.
# --------------------------------------------------
if st.user.is_logged_in and not st.session_state.get("_login_redirect_done"):
    st.session_state["_login_redirect_done"] = True
    try:
        from streamlit_cookies_manager import EncryptedCookieManager
        _cookies = EncryptedCookieManager(
            prefix="smart_city_air",
            password=st.secrets.get("cookie_password", "local-dev-only-change-me"),
        )
        if _cookies.ready():
            _target = _cookies.get("login_redirect_target")
            if _target:
                del _cookies["login_redirect_target"]
                _cookies.save()
                _page_map = {
                    "Smart_City_Operations":        operations,
                    "Air_Quality_Monitoring":        monitoring,
                    "Temporal_Trends":               temporal,
                    "Forecasting":                    forecast,
                    "GeoAI_Spatial_Analysis":         spatial,
                    "Weather_Drivers":                weather,
                    "Smart_City_Decision_Support":    decision,
                    "Project_Assistant":              assistant,
                    "Scope_and_Limitations":          methods,
                }
                _matched = _page_map.get(_target)
                if _matched:
                    st.switch_page(_matched)
    except ImportError:
        pass

st.markdown("""
<style>
/* hide Project Assistant from nav — accessible via Home button only */
[data-testid="stSidebarNav"] a[href*="Project_Assistant"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


pg.run()