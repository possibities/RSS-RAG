from __future__ import annotations

import asyncio
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from zoneinfo import ZoneInfo

from config import EMAIL_CONFIG, SCHEDULER_TIMEZONE, require_setting

from analysis.digest import generate_daily_digest


def build_email_html(digests: list[dict[str, object]], date_str: str) -> str:
    sections = ""
    for digest in digests:
        sources_html = "".join(
            f'<li><a href="{escape(str(source["url"]))}">{escape(str(source["title"]))}</a></li>'
            for source in digest["sources"]
        )
        sections += f"""
        <div style="margin-bottom:32px; border-left:3px solid #4a90e2; padding-left:16px;">
            <h2 style="color:#4a90e2;">{escape(str(digest["category"]))} 日报</h2>
            <div style="white-space:pre-wrap; line-height:1.8;">{escape(str(digest["digest"]))}</div>
            <details style="margin-top:12px;">
                <summary style="cursor:pointer; color:#888;">参考来源 ({len(digest["sources"])} 篇)</summary>
                <ul>{sources_html}</ul>
            </details>
        </div>"""

    return f"""
    <html><body style="font-family:sans-serif; max-width:680px; margin:auto; padding:24px;">
        <h1 style="border-bottom:1px solid #eee; padding-bottom:12px;">
            AI 技术日报 | {escape(date_str)}
        </h1>
        {sections}
        <p style="color:#aaa; font-size:12px; margin-top:32px;">
            本邮件由私有知识库系统自动生成，内容来源于 RSS 订阅。
        </p>
    </body></html>"""


async def send_daily_email(db) -> bool:
    to_addrs = list(EMAIL_CONFIG["to_addrs"])
    if not to_addrs:
        return False

    today = datetime.now(ZoneInfo(SCHEDULER_TIMEZONE)).date().isoformat()
    categories = await db.fetch("SELECT id, name FROM categories WHERE parent_id IS NULL")

    digests: list[dict[str, object]] = []
    for category in categories:
        result = await generate_daily_digest(category["name"], str(category["id"]), today, db)
        if result["sources"]:
            digests.append(result)

    if not digests:
        return False

    html = build_email_html(digests, today)
    message = MIMEMultipart("alternative")
    message["Subject"] = f"AI 技术日报 {today}"
    message["From"] = str(EMAIL_CONFIG["from_addr"])
    message["To"] = ", ".join(to_addrs)
    message.attach(MIMEText(html, "html", "utf-8"))

    def _send_smtp() -> None:
        smtp_host = require_setting("SMTP_HOST", str(EMAIL_CONFIG["smtp_host"]))
        from_addr = require_setting("SMTP_FROM_ADDR", str(EMAIL_CONFIG["from_addr"]))
        with smtplib.SMTP(smtp_host, int(EMAIL_CONFIG["smtp_port"])) as server:
            if EMAIL_CONFIG["use_starttls"]:
                server.starttls()
            if EMAIL_CONFIG["username"]:
                server.login(str(EMAIL_CONFIG["username"]), str(EMAIL_CONFIG["password"]))
            server.sendmail(from_addr, to_addrs, message.as_string())

    await asyncio.to_thread(_send_smtp)
    return True
