"""Unit tests for src/ui/theme.py"""

from __future__ import annotations

from src.ui.theme import render_match_dial


class TestRenderMatchDial:
    def test_renders_percentage_and_dial(self) -> None:
        html = render_match_dial(0.87)
        assert "87% match" in html
        assert '--pct:87' in html
        assert 'class="match-dial"' in html

    def test_rounds_to_nearest_percent(self) -> None:
        html = render_match_dial(0.876)
        assert "88% match" in html

    def test_zero_score(self) -> None:
        html = render_match_dial(0.0)
        assert "0% match" in html
        assert '--pct:0' in html

    def test_full_score(self) -> None:
        html = render_match_dial(1.0)
        assert "100% match" in html

    def test_clamps_above_one(self) -> None:
        html = render_match_dial(1.5)
        assert "100% match" in html

    def test_clamps_below_zero(self) -> None:
        html = render_match_dial(-0.3)
        assert "0% match" in html

    def test_custom_label_overrides_default_text(self) -> None:
        html = render_match_dial(0.9, label="Strong match")
        assert "Strong match" in html
        assert "90% match" not in html

    def test_includes_accessible_aria_label(self) -> None:
        html = render_match_dial(0.42)
        assert 'aria-label="42 percent match"' in html

    def test_includes_role_img(self) -> None:
        html = render_match_dial(0.5)
        assert 'role="img"' in html
