from __future__ import annotations

from bs4 import BeautifulSoup

from config import MAX_CONTENT_CHARS


def clean_content(raw_html: str) -> str:
    """
    Strip noisy HTML, preserve readable text, and cap size for downstream models.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside", "iframe"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)[:MAX_CONTENT_CHARS]
