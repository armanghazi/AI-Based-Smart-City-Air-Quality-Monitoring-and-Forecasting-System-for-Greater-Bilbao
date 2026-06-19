"""
i18n_auto.py — on-the-fly machine translation + RTL/font support.

Three languages: English (en), Spanish (es), Persian/Farsi (fa).
UI strings are translated via Google Translate (deep-translator), cached per session.
Persian triggers RTL text direction and Vazirmatn web font automatically.

IMPORTANT / honest caveat:
  Machine translation can mangle multi-word technical terms ("data leakage",
  "time-based split", "exceedance"). The PROTECTED list restores single
  proper nouns/acronyms only. For titles and key terminology, a hand-written
  dictionary is safer. Use this module for longer, low-risk descriptive text.

Requires: deep-translator  (add `deep-translator` to requirements.txt)
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------
# LANGUAGE METADATA
# --------------------------------------------------
LANGUAGES: dict[str, dict] = {
    "en": {"label": "English",  "dir": "ltr", "rtl": False},
    "es": {"label": "Español",  "dir": "ltr", "rtl": False},
    # "fa": {"label": "فارسی",    "dir": "rtl", "rtl": True},
}

# Terms kept verbatim after translation (acronyms, units, station codes, model names)
PROTECTED = [
    "WHO", "EU", "AQI", "EAQI", "ICA", "EPA",
    "XGBoost", "SHAP", "SARIMA", "MLP", "R2", "R²",
    "PM2.5", "PM10", "NO2", "SO2",
    "GeoAI", "IDW", "Petronor", "Bilbao", "Bizkaia",
    "MAZARREDO", "BASAURI", "BARAKALDO", "ERANDIO",
    "SANTURCE", "MUSKIZ", "ALGORTA_BBIZI2",
]


# --------------------------------------------------
# TRANSLATION
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def translate(text: str, target: str = "es") -> str:
    """Translate a UI string to `target` language code.
    Cached — each unique (text, target) pair is fetched once per session.
    Falls back to the original English string if translation fails."""
    if not text or target == "en":
        return text
    try:
        from deep_translator import GoogleTranslator
        out = GoogleTranslator(source="en", target=target).translate(text)
        # Restore protected terms the translator may have altered or lowercased
        for term in PROTECTED:
            out = out.replace(term.lower(), term).replace(term.capitalize(), term)
        return out
    except Exception:
        return text  # graceful degradation to English


def tr(text: str) -> str:
    """Shorthand: translate `text` into the currently selected language."""
    return translate(text, st.session_state.get("lang", "en"))


# --------------------------------------------------
# RTL + FONT INJECTION
# --------------------------------------------------
def _inject_fa_styles() -> None:
    """Inject Vazirmatn web font and RTL direction for Persian UI text.

    Targets only markdown/text containers so charts, tables, and inputs
    (which should stay LTR for numbers and axis labels) are unaffected.
    Relies on Streamlit's internal data-testid attributes — verify after
    major Streamlit upgrades (another reason to pin streamlit>=1.42,<2).
    """
    st.markdown(
        """
        <style>
        /* ── Vazirmatn — best Persian web font ── */
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600&display=swap');

        /* ── RTL text blocks only (not charts / inputs / tables) ── */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        [data-testid="stMarkdownContainer"] blockquote {
            direction: rtl;
            text-align: right;
            font-family: 'Vazirmatn', Tahoma, Arial, sans-serif;
        }

        /* ── Sidebar text for Persian ── */
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label {
            font-family: 'Vazirmatn', Tahoma, Arial, sans-serif;
        }

        /* ── Chat messages for Persian ── */
        [data-testid="stChatMessage"] p {
            direction: rtl;
            text-align: right;
            font-family: 'Vazirmatn', Tahoma, Arial, sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_lang_styles() -> None:
    """Call once per page (after set_page_config) to apply RTL/font if needed."""
    lang = st.session_state.get("lang", "en")
    if LANGUAGES.get(lang, {}).get("rtl"):
        _inject_fa_styles()


# --------------------------------------------------
# LANGUAGE SELECTOR
# --------------------------------------------------
def language_selector(location=st.sidebar) -> str:
    """Render the language radio in `location` and return the current code.
    Always call this before any tr(...) or apply_lang_styles() on the page."""
    options = list(LANGUAGES.keys())
    labels  = [LANGUAGES[k]["label"] for k in options]
    idx = location.radio(
        "🌐 Language / Idioma ",
        options=range(len(options)),
        format_func=lambda i: labels[i],
        key="lang_idx",
        horizontal=True,
    )
    lang = options[idx]
    st.session_state["lang"] = lang
    return lang


# --------------------------------------------------
# USAGE TEMPLATE (copy into any page)
# --------------------------------------------------
# import sys; from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))
# from i18n_auto import language_selector, apply_lang_styles, tr
#
# st.set_page_config(...)
# language_selector()        # sidebar — sets st.session_state["lang"]
# apply_lang_styles()        # injects RTL + Vazirmatn if lang == "fa"
#
# st.markdown(f"## {tr('City-Wide Status')}")
# st.caption(tr("Each station sits in a zone defined by its dominant emission source."))