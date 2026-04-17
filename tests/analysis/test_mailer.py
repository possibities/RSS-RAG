from __future__ import annotations

from analysis.mailer import build_email_html


def test_build_email_html_escapes_all_rss_sourced_content() -> None:
    html = build_email_html(
        [
            {
                "category": "<LLM>",
                "digest": "<b>digest</b>",
                "sources": [
                    {"title": "<script>alert(1)</script>", "url": 'https://example.com/?q=<bad>'},
                ],
            }
        ],
        "2026-04-17",
    )

    assert "&lt;LLM&gt;" in html
    assert "&lt;b&gt;digest&lt;/b&gt;" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "https://example.com/?q=&lt;bad&gt;" in html
    assert "<script>alert(1)</script>" not in html
