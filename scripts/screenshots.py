from __future__ import annotations

import os

from playwright.sync_api import sync_playwright

OUT = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)
URL = "http://localhost:8501"


def main() -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=2)
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(2500)

        # Hero: header + two pattern cards + query bar (pre-run state)
        page.screenshot(path=os.path.join(OUT, "01-hero.png"))

        # Run the comparison and wait for the post-run sections to render.
        page.get_by_role("button", name="Run comparison").click()
        page.wait_for_selector("text=What this query actually did", timeout=90000)
        page.wait_for_timeout(4000)

        # Full page overview after running
        page.screenshot(path=os.path.join(OUT, "02-overview.png"), full_page=True)

        # Side-by-side result columns (the horizontal block that holds the judge sliders)
        cols = page.locator('div[data-testid="stHorizontalBlock"]').filter(
            has_text="Faithfulness / groundedness"
        ).last
        cols.scroll_into_view_if_needed()
        page.wait_for_timeout(800)
        cols.screenshot(path=os.path.join(OUT, "03-columns.png"))

        # D3 pipeline flow (first components iframe)
        frames = page.locator("iframe")
        frames.nth(0).scroll_into_view_if_needed()
        page.wait_for_timeout(800)
        frames.nth(0).screenshot(path=os.path.join(OUT, "04-flow.png"))

        # D3 retrieval explain (second components iframe)
        frames.nth(1).scroll_into_view_if_needed()
        page.wait_for_timeout(800)
        frames.nth(1).screenshot(path=os.path.join(OUT, "05-explain.png"))

        browser.close()
        print("done")


if __name__ == "__main__":
    main()
