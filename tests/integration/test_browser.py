"""Interactive frontend checks. Network fixtures are test-only."""

import json
import os
import threading
import time
from pathlib import Path

import pytest
from werkzeug.serving import make_server

from musicgen.frontend.app import create_app
from musicgen.generation.runtime.midi import Note, write_midi

pytestmark = pytest.mark.browser


def test_browser_genre_duration_poll_download_and_mobile(settings, tmp_path):
    if os.environ.get("MUSICGEN_BROWSER_TESTS") != "1":
        pytest.skip("Set MUSICGEN_BROWSER_TESTS=1 after installing Playwright Chromium.")
    playwright = pytest.importorskip("playwright.sync_api")
    settings.run_mode = "distributed"
    server = make_server("127.0.0.1", 0, create_app(settings))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    midi_path = tmp_path / "browser_fixture.mid"
    write_midi(midi_path, [Note(0, 120, 60, 0, 80)], bpm=120, end_beat=120)
    payloads, errors = [], []
    job_id = "b" * 32
    polls = 0

    def intercept(route):
        nonlocal polls
        path = route.request.url.split("/api/", 1)[-1]
        data = {}
        headers = {"Content-Type": "application/json"}
        if path == "health/ready":
            data = {"ready": True}
        elif path == "config":
            data = {"max_duration_seconds": 240, "default_duration_seconds": 45}
        elif path == "generations" and route.request.method == "POST":
            payloads.append(route.request.post_data_json)
            data = {
                "id": job_id,
                "status": "queued",
                "created_at": time.time(),
                **payloads[-1],
            }
        elif path == "generations/" + job_id:
            polls += 1
            data = {
                "id": job_id,
                "status": "generating" if polls == 1 else "completed",
                "created_at": time.time(),
                "actual_duration_seconds": 60,
                **payloads[-1],
            }
        elif path.endswith("/download"):
            route.fulfill(
                status=200,
                content_type="audio/midi",
                body=midi_path.read_bytes(),
                headers={"Content-Disposition": 'attachment; filename="composition.mid"'},
            )
            return
        route.fulfill(
            status=202 if path == "generations" else 200, body=json.dumps(data), headers=headers
        )

    try:
        with playwright.sync_playwright() as play:
            browser = play.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/api/**", intercept)
            page.goto("http://127.0.0.1:" + str(server.server_port))
            playwright.expect(page.locator("#generate-button")).to_be_enabled()
            if os.environ.get("MUSICGEN_UI_ARTIFACT_DIR"):
                target = Path(os.environ["MUSICGEN_UI_ARTIFACT_DIR"])
                target.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(target / "desktop.png"), full_page=True)
            page.locator('label:has(input[value="rock_pop"])').click()
            page.get_by_role("button", name="1 min", exact=True).click()
            assert page.locator("#duration").input_value() == "60"
            page.get_by_role("button", name="Generate MIDI", exact=True).click()
            playwright.expect(page.locator("#result-heading")).to_have_text(
                "Your MIDI is ready.", timeout=10000
            )
            assert payloads == [{"genre": "rock_pop", "duration_seconds": 60}]
            with page.expect_download() as download_info:
                page.get_by_role("link", name="Download MIDI", exact=True).click()
            download = download_info.value
            assert Path(download.path()).read_bytes().startswith(b"MThd")
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            if os.environ.get("MUSICGEN_UI_ARTIFACT_DIR"):
                page.screenshot(path=str(target / "mobile.png"), full_page=True)
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=3)
