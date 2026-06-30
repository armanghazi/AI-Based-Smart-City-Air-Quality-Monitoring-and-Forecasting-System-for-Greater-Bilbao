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

# --------------------------------------------------
# Auto-redirect admin to Operations Dashboard once per session after login.
# Runs BEFORE st.navigation/pg.run() so it fires before Home ever renders.
# "_admin_welcomed" guards against redirect-looping on every rerun.
# --------------------------------------------------
try:
    if st.user.is_logged_in:
        from auth import current_role
        role = current_role()
        if role == "admin" and not st.session_state.get("_admin_welcomed"):
            st.session_state["_admin_welcomed"] = True
            st.switch_page("pages/9_Smart_City_Operations.py")
except Exception:
    pass

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

st.markdown("""
<style>
/* hide Project Assistant from nav — accessible via Home button only */
[data-testid="stSidebarNav"] a[href*="Project_Assistant"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


pg.run()