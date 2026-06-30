"""
auth.py — native Streamlit OIDC authentication (st.login) + role-based authorization.

Single source of truth (mirrors forecast_utils / spatial_utils / aqi modules):
import this in any page that needs protection.

Streamlit's st.login handles AUTHENTICATION (who the user is). OIDC does NOT provide
authorization, so this module adds the AUTHORIZATION layer (what they may do) via a
secrets-driven whitelist, with two roles: "admin" and "viewer".

Requirements:
  - streamlit >= 1.42
  - Authlib >= 1.3.2
  - an [auth] block in secrets.toml (see the template at the bottom of this file)
  - an [access] block in secrets.toml listing admins / viewers / allowed_domain
"""

from __future__ import annotations

import streamlit as st


# --------------------------------------------------
# Access lists (read from secrets, never hard-coded in source)
# --------------------------------------------------
def _allowed() -> tuple[set[str], set[str], str]:
    """Return (admins, viewers, allowed_domain). Missing config -> deny by default."""
    access = st.secrets.get("access", {})
    admins = {e.lower() for e in access.get("admins", [])}
    viewers = {e.lower() for e in access.get("viewers", [])}
    domain = (access.get("allowed_domain") or "").lower().lstrip("@")
    return admins, viewers, domain


def current_role() -> str | None:
    """'admin' | 'viewer' | None for the logged-in user (None = not authorized)."""
    if not st.user.is_logged_in:
        return None
    email = (st.user.email or "").lower()
    admins, viewers, domain = _allowed()
    if email in admins:
        return "admin"
    if email in viewers:
        return "viewer"
    if domain and email.endswith("@" + domain):
        return "viewer"
    return None


# --------------------------------------------------
# UI helpers
# --------------------------------------------------
def _login_screen() -> None:
    st.title("🔒 This area is private")
    st.write("Please sign in with your Google account to continue.")

    def _login_and_remember():
        # Remember which page triggered the login so app.py can route back
        # to it after st.login's OIDC redirect — otherwise Streamlit always
        # lands on the default page (Home).
        st.session_state["_login_redirect_target"] = st.context.url
        st.login()

    st.button("Log in with Google", type="primary", on_click=_login_and_remember)


def logout_button(location=st.sidebar) -> None:
    """Render a logout button (defaults to the sidebar)."""
    location.button("Log out", on_click=st.logout)


# --------------------------------------------------
# The gate — call at the TOP of a protected page
# --------------------------------------------------
def require_auth(role: str = "viewer") -> dict:
    """
    Gate a page. Call this BEFORE any protected content renders.
      role="viewer" -> any whitelisted user (admin or viewer)
      role="admin"  -> admins only
    Stops the script (st.stop) if the user is not logged in or not authorized.
    Returns {"email", "name", "role"} on success.
    """
    # Friendly message if secrets.toml has no [auth] block yet (setup phase)
    try:
        logged_in = st.user.is_logged_in
    except Exception:
        st.error("Authentication is not configured. Add an [auth] block to secrets.toml.")
        st.stop()

    if not logged_in:
        _login_screen()
        st.stop()

    user_role = current_role()
    if user_role is None or (role == "admin" and user_role != "admin"):
        st.error("You don't have access to this page.")
        st.caption(f"Signed in as {st.user.email}")
        logout_button(st)  # inline logout so a wrong account can switch
        st.stop()

    return {"email": st.user.email, "name": st.user.name, "role": user_role}


def is_admin() -> bool:
    """Safe wrapper for entry-script: never raises, returns False on any error."""
    try:
        return current_role() == "admin"
    except Exception:
        return False
# ======================================================================
# secrets.toml TEMPLATE  (do NOT commit real secrets; keep .streamlit/ in .gitignore)
# ----------------------------------------------------------------------
# [auth]
# redirect_uri        = "http://localhost:8501/oauth2callback"   # LOCAL
# # redirect_uri      = "https://geoai-dashboard.streamlit.app/oauth2callback"  # CLOUD
# cookie_secret       = "REPLACE_WITH_A_LONG_RANDOM_STRING"
# client_id           = "xxxx.apps.googleusercontent.com"
# client_secret       = "GOCSPX-xxxx"
# server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
#
# [access]
# admins         = ["arman@example.com"]
# viewers        = ["teacher@example.com"]
# allowed_domain = ""   # e.g. "alumni.uni.edu" to admit a whole domain as viewers
# ======================================================================