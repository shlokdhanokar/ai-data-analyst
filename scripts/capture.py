"""Capture README screenshots (and optionally a demo video) of the running app with Playwright.

Usage:  python scripts/capture.py [--video] [--url http://localhost:8501]
Requires: pip install playwright && playwright install chromium
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs"
QUESTIONS = [
    "Which region generated the highest revenue?",
    "Show monthly sales trends.",
    "Detect anomalies in the dataset.",
]


def wait_ready(page: Page, url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            page.goto(url, wait_until="networkidle")
            page.wait_for_selector("text=AI Data Analyst", timeout=15000)
            return
        except Exception:
            if time.time() > deadline:
                raise
            page.wait_for_timeout(2000)


def idle(page: Page, extra_ms: int = 800) -> None:
    """Wait until Streamlit has finished running the script."""
    page.wait_for_timeout(500)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120000)
    page.wait_for_timeout(extra_ms)


def ask(page: Page, q: str) -> None:
    before = page.get_by_test_id("stChatMessage").count()
    box = page.get_by_test_id("stChatInputTextArea")
    box.click()
    box.type(q, delay=25)
    box.press("Enter")
    # Wait for the answer to be committed (user + assistant messages added, no spinner).
    page.wait_for_function(
        "(n) => document.querySelectorAll('[data-testid=\"stChatMessage\"]').length >= n + 2"
        " && !document.body.innerText.includes('Analysing')",
        arg=before, timeout=180000,
    )
    idle(page, 2000)
    # Scroll so the newest question sits at the top of the viewport.
    msgs = page.get_by_test_id("stChatMessage")
    msgs.nth(msgs.count() - 2).evaluate("el => el.scrollIntoView({block: 'start'})")
    page.wait_for_timeout(1200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--no-chat", action="store_true", help="skip chat screenshots (no LLM key)")
    args = ap.parse_args()
    shots = OUT / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:  # fall back to a locally installed Chrome
            browser = p.chromium.launch(channel="chrome")
        ctx_kwargs = dict(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        if args.video:
            ctx_kwargs.update(record_video_dir=str(OUT / "video_raw"), record_video_size={"width": 1440, "height": 900})
        ctx = browser.new_context(**ctx_kwargs)
        page = ctx.new_page()
        wait_ready(page, args.url)
        page.get_by_role("button", name="Load sample datasets").click()
        idle(page)

        if not args.no_chat:
            for i, q in enumerate(QUESTIONS, 1):
                ask(page, q)
                page.screenshot(path=str(shots / f"chat_{i}.png"), full_page=False)
                if args.video:
                    page.mouse.wheel(0, 500)
                    page.wait_for_timeout(1500)

        page.get_by_role("tab", name="📈 Dashboard").click()
        idle(page, 2500)
        page.screenshot(path=str(shots / "dashboard.png"))
        page.get_by_role("tab", name="🩺 Data quality").click()
        idle(page, 1500)
        page.screenshot(path=str(shots / "data_quality.png"))
        if args.video:
            page.wait_for_timeout(1500)
        ctx.close()
        browser.close()
    print("saved to", shots)


if __name__ == "__main__":
    main()
