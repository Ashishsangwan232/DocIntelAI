"""
ui/theme.py
===========
Injects DocIntel AI's custom visual theme and provides
`render_match_dial()`, the signature "ink-dial" component used
wherever a similarity/match score is shown (chat citations, search
results).

The base dark palette comes from `.streamlit/config.toml` (Streamlit's
officially supported theming mechanism); this module layers
glassmorphism, typography, animations, and the match-dial on top via
a single injected stylesheet, per the Phase 1 wireframes.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "styles.css"

# Injected once per session — re-injecting identical CSS on every
# rerun is harmless but wasteful; this mirrors the guard pattern
# already used in src/utils/logger.py for handler registration.
_STATE_INJECTED = "_theme_css_injected"


def inject_custom_css() -> None:
    """Inject the custom stylesheet into the page. Call once near the top of app.py."""
    if st.session_state.get(_STATE_INJECTED):
        return

    try:
        css = _CSS_PATH.read_text()
    except OSError:
        # Missing stylesheet should never crash the app — it just
        # means the page renders with Streamlit's base dark theme
        # (from config.toml) and no glassmorphism/animation layer.
        return

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.session_state[_STATE_INJECTED] = True


def render_match_dial(similarity_score: float, label: str | None = None) -> str:
    """
    Build the HTML for the signature "ink-dial" match indicator: a
    small conic-gradient ring filled to the similarity percentage,
    paired with a text label (never color/visual-only, for
    accessibility and so existing text-based assertions/screen readers
    both work).

    Args:
        similarity_score: A value in [0, 1].
        label: Optional override for the text label; defaults to
            "NN% match".

    Returns:
        An HTML snippet — pass to `st.markdown(..., unsafe_allow_html=True)`.
    """
    percent = round(max(0.0, min(1.0, similarity_score)) * 100)
    text = label if label is not None else f"{percent}% match"
    return (
        f'<span class="match-dial-wrap">'
        f'<span class="match-dial" style="--pct:{percent};" '
        f'role="img" aria-label="{percent} percent match"></span>'
        f'<span>{text}</span>'
        f'</span>'
    )
