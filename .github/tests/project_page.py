"""Browser regression checks for the static research page."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import csv
import json
import threading
import unittest

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS = ROOT / "page-screenshots"


class ProjectPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCREENSHOTS.mkdir(exist_ok=True)
        handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / "docs"))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/"
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.context = self.browser.new_context(
            viewport={"width": 1440, "height": 1050},
            permissions=["clipboard-read", "clipboard-write"],
            reduced_motion="reduce",
        )
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(self.url, wait_until="networkidle")

    def tearDown(self):
        self.assertEqual(self.errors, [])
        self.context.close()

    def assert_no_overflow(self):
        self.assertTrue(self.page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        ))

    def select_svg_text(self, selector):
        self.page.locator(selector).scroll_into_view_if_needed()
        self.page.wait_for_function(
            "selector => document.querySelector(selector).contentDocument?.querySelector('text')",
            arg=selector,
        )
        position = self.page.locator(selector).evaluate("""object => {
            const text = [...object.contentDocument.querySelectorAll('text')].find(node =>
                node.textContent.trim() === 'Pass 1: Initial Decision' &&
                node.getBoundingClientRect().width > 0
            );
            const frame = object.getBoundingClientRect();
            const box = text.getBoundingClientRect();
            return {x: frame.x + box.x + box.width / 2, y: frame.y + box.y + box.height / 2};
        }""")
        # A real pointer event focuses the embedded document, as a reader would.
        self.page.mouse.click(position["x"], position["y"])
        return self.page.locator(selector).evaluate("""object => {
            const doc = object.contentDocument;
            const text = [...doc.querySelectorAll('text')].find(node =>
                node.textContent.trim() === 'Pass 1: Initial Decision' &&
                node.getBoundingClientRect().width > 0
            );
            const range = doc.createRange();
            range.selectNodeContents(text);
            const selection = doc.defaultView.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            return {selected: selection.toString(), expected: text.textContent};
        }""")

    def test_desktop_and_mobile_layouts(self):
        for width in (1440, 768, 390, 320):
            with self.subTest(width=width):
                self.page.set_viewport_size({"width": width, "height": 1050})
                self.assert_no_overflow()
                self.assertTrue(self.page.locator("h1").is_visible())
                self.assertEqual(self.page.locator(".publication-links a").count(), 4)
                for image in self.page.locator("figure img:visible").all():
                    image.scroll_into_view_if_needed()
                    self.page.wait_for_function(
                        "img => img.complete && img.naturalWidth > 0", arg=image.element_handle()
                    )
                self.page.evaluate("window.scrollTo(0, 0)")
                self.page.screenshot(path=str(SCREENSHOTS / f"hero-{width}.png"))
                self.page.screenshot(path=str(SCREENSHOTS / f"page-{width}.png"), full_page=True)

    def test_selectable_vector_and_modal(self):
        selected = self.select_svg_text(".teaser object")
        self.assertTrue(selected["selected"].strip())
        self.assertEqual(selected["selected"], selected["expected"])
        self.page.keyboard.press("Control+c")
        self.page.wait_for_function(
            "async expected => await navigator.clipboard.readText() === expected",
            arg=selected["selected"],
        )
        self.assertEqual(self.page.evaluate("navigator.clipboard.readText()"), selected["selected"])
        self.page.locator("[data-enlarge-methodology]").click()
        self.assertTrue(self.page.locator("dialog").evaluate("dialog => dialog.open"))
        selected = self.select_svg_text("[data-figure-vector]")
        self.assertEqual(selected["selected"], selected["expected"])
        self.page.keyboard.press("Control+c")
        self.page.wait_for_function(
            "async expected => await navigator.clipboard.readText() === expected",
            arg=selected["selected"],
        )
        self.assertEqual(self.page.evaluate("navigator.clipboard.readText()"), selected["selected"])
        self.page.screenshot(path=str(SCREENSHOTS / "methodology-modal.png"))
        self.page.keyboard.press("Escape")
        self.assertFalse(self.page.locator("dialog").evaluate("dialog => dialog.open"))
        self.assertTrue(self.page.locator("[data-enlarge-methodology]").evaluate(
            "button => button === document.activeElement"
        ))

    def test_figure_zoom_and_copy_controls(self):
        for button in self.page.locator(".figure-zoom:visible").all():
            button.click()
            self.assertTrue(self.page.locator("[data-figure-image]").is_visible())
            self.page.locator("[data-close-figure]").click()
        for target in ("bibtex", "reproduction-code"):
            self.page.locator(f'[data-copy-target="{target}"]').click()
            self.assertEqual(self.page.evaluate("navigator.clipboard.readText()"),
                             self.page.locator(f"#{target}").inner_text().strip())

    def test_local_links_and_no_javascript(self):
        links = self.page.locator("a[href]").evaluate_all("nodes => nodes.map(node => node.href)")
        for link in set(links):
            if link.startswith(self.url):
                self.assertLess(self.context.request.get(link.split('#')[0]).status, 400, link)
        with self.browser.new_context(java_script_enabled=False) as context:
            page = context.new_page()
            page.goto(self.url)
            self.assertTrue(page.locator("h1").is_visible())
            self.assertEqual(page.locator("figure img").count(), 3)
            self.assertTrue(page.locator(".teaser object").is_visible())
            self.assertFalse(page.locator("[data-enlarge-methodology]").is_visible())

    def test_paired_effect_data_and_interaction(self):
        payload = json.loads((ROOT / 'docs/assets/results.json').read_text())
        with (ROOT / 'reproducibility/rebuttal/analysis/core_nine_models.csv').open() as source:
            original = {row['model']: row for row in csv.DictReader(source)}
        self.assertEqual(len(payload['models']), 9)
        self.assertEqual(self.page.locator('.effect-row').count(), 9)
        self.assertEqual(self.page.locator('.effect-group').count(), 2)
        self.assertEqual(self.page.locator('.effect-group').nth(0).locator('button').count(), 5)
        self.assertEqual(self.page.locator('.effect-group').nth(1).locator('button').count(), 4)
        for row in payload['models']:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    self.assertAlmostEqual(value, float(original[row['model']][key]), places=10)
                else:
                    self.assertEqual(value, original[row['model']][key])
            self.assertAlmostEqual(row['wrong_correct'] - row['correct_wrong'], row['delta_acc'])
            button = self.page.locator(f'.effect-row[data-model="{row["model"]}"]')
            button.click()
            detail = self.page.locator('[data-model-detail]')
            self.assertEqual(detail.locator('h4').inner_text(), row['model'])
            self.assertIn(f'{row["n_paired"]:,}', detail.inner_text())
            self.assertEqual(self.page.locator('.effect-row[aria-pressed="true"]').count(), 1)
        self.page.locator('.effect-row').first.focus()
        self.page.keyboard.press('Enter')
        self.assertEqual(self.page.locator('[data-model-detail] h4').inner_text(), 'GPT-4o-mini')
        for width in (1440, 390, 320):
            self.page.set_viewport_size({'width': width, 'height': 1050})
            self.assert_no_overflow()
            self.page.locator('#effect-explorer').screenshot(path=str(SCREENSHOTS / f'effects-{width}.png'))

    def test_data_network_failure_keeps_static_figure(self):
        self.page.route('**/assets/results.json', lambda route: route.abort())
        self.page.reload(wait_until='networkidle')
        self.assertFalse(self.page.locator('#effect-explorer').is_visible())
        self.assertTrue(self.page.locator('.reversal-static').is_visible())


if __name__ == "__main__":
    unittest.main(verbosity=2)
