"""Benchmark the real Flask -> FastAPI -> PyTorch flow and the direct HTTP API.

Runs serial requests with the application's unchanged sampling controls. Each
request receives the normal randomly assigned seed, recorded in its metadata.
The dedicated loopback services are stopped on exit. No production settings or
checkpoint files are replaced. Requires the existing Playwright installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright


def request(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.status, response.read(), dict(response.headers)


def get_json(url, payload=None):
    status, data, _ = request(url, payload)
    return status, json.loads(data)


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--frontend-port", type=int, default=5050)
    parser.add_argument("--api-port", type=int, default=8050)
    parser.add_argument("--expected-epoch", type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.trials <= 10:
        parser.error("trials must be between 1 and 10")
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "benchmark.json").exists():
        parser.error("Choose a new output directory; benchmark.json already exists")
    checkpoint = args.checkpoint.resolve()
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    api = f"http://127.0.0.1:{args.api_port}"
    frontend = f"http://127.0.0.1:{args.frontend_port}"
    env = dict(os.environ)
    env.update(
        MUSICGEN_RUN_MODE="local",
        MUSICGEN_CHECKPOINT_PATH=str(checkpoint),
        MUSICGEN_OUTPUT_DIR=str(output / "jobs"),
        MUSICGEN_DEVICE="cpu",
        MUSICGEN_TORCH_THREADS="2",
        MUSICGEN_PALETTE="original",
        MUSICGEN_ENDING_STYLE="release",
        MUSICGEN_REPETITION_PENALTY="0.3",
        MUSICGEN_MAX_TOKENS="30000",
        MUSICGEN_GENERATION_TIMEOUT_SECONDS="1800",
        MUSICGEN_API_BASE_URL=api,
        MUSICGEN_FRONTEND_ORIGIN=frontend,
    )
    report = {
        "started_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha,
        "expected_epoch": args.expected_epoch,
        "configuration": {
            "duration_seconds": 60,
            "trials_per_genre_per_route": args.trials,
            "device": "cpu",
            "torch_threads": 2,
            "palette": "original",
            "ending_style": "release",
            "repetition_penalty": 0.3,
            "max_tokens": 30000,
            "run_mode": "local",
            "serial_requests": True,
            "frontend_url": frontend,
            "api_url": api,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "method": (
            "Actual Flask page, unmocked FastAPI HTTP endpoints and loaded PyTorch model. "
            "Serial jobs alternate frontend and backend routes within each genre/trial. "
            "Frontend time is click to visible download link; backend time is POST to "
            "observed completion (0.2s polling). Worker time is completed_at-started_at. "
            "Checkpoint loading is excluded from generation, recorded separately. "
            "All requests use normal random server seeds, recorded in sidecars."
        ),
        "runs": [],
        "browser_errors": [],
    }
    cpu_info = Path("/proc/cpuinfo").read_text()
    report["configuration"]["cpu_model"] = next(
        (
            line.split(":", 1)[1].strip()
            for line in cpu_info.splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    logs, processes = [], []
    startup = time.perf_counter()
    try:
        for name, command in [
            (
                "api",
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "musicgen.api.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.api_port),
                    "--no-access-log",
                ],
            ),
            (
                "frontend",
                [
                    sys.executable,
                    "-m",
                    "flask",
                    "--app",
                    "musicgen.frontend.app:create_app()",
                    "run",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.frontend_port),
                ],
            ),
        ]:
            log = (output / f"{name}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            )
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A service exited; inspect api.log/frontend.log")
            if time.perf_counter() - startup > 90:
                raise TimeoutError("Services did not become ready")
            try:
                _, ready = get_json(api + "/api/health/ready")
                status, _, _ = request(frontend)
                if ready["ready"] and status == 200:
                    break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.2)
        report["startup_to_ready_seconds"] = time.perf_counter() - startup
        _, metrics, _ = request(api + "/metrics")
        report["startup_metrics"] = [
            line
            for line in metrics.decode().splitlines()
            if line.startswith("musicgen_model_load_seconds ")
        ]
        print(f"Services ready in {report['startup_to_ready_seconds']:.2f}s", flush=True)

        with sync_playwright() as play:
            browser = play.chromium.launch(headless=True, args=["--no-sandbox"])
            report["configuration"]["browser_version"] = browser.version
            page = browser.new_page(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
            page.on("pageerror", lambda error: report["browser_errors"].append(str(error)))
            page.goto(frontend)
            page.locator("#generate-button").wait_for(state="visible")
            page.wait_for_function("!document.getElementById('generate-button').disabled")
            page.screenshot(path=str(output / "frontend_ready.png"), full_page=True)
            for trial in range(1, args.trials + 1):
                for genre in ("classical", "romantic", "rock_pop"):
                    for route in ("frontend", "backend"):
                        target = output / route
                        target.mkdir(exist_ok=True)
                        stem = f"{genre}_60s_trial{trial}"
                        midi_path = target / (stem + ".mid")
                        payload = {"genre": genre, "duration_seconds": 60}
                        print(f"START {route} {genre} trial {trial}", flush=True)
                        if route == "frontend":
                            page.locator(f'label:has(input[value="{genre}"])').click()
                            page.get_by_role("button", name="1 min", exact=True).click()
                            assert page.locator("#duration").input_value() == "60"
                            started = time.perf_counter()
                            with page.expect_response(
                                lambda r: (
                                    r.url == api + "/api/generations" and r.request.method == "POST"
                                ),
                                timeout=10000,
                            ) as response_info:
                                page.get_by_role("button", name="Generate MIDI", exact=True).click()
                            response = response_info.value
                            submission_seconds = time.perf_counter() - started
                            assert response.status == 202
                            assert response.request.post_data_json == payload
                            job = response.json()
                            while True:
                                try:
                                    page.locator(
                                        f'#download-button[href="{api}{job["status_url"]}/download"]'
                                    ).wait_for(state="visible", timeout=30000)
                                    break
                                except PlaywrightTimeout:
                                    _, status_job = get_json(api + job["status_url"])
                                    if status_job["status"] == "failed":
                                        raise RuntimeError(status_job["error"])
                                    if time.perf_counter() - started > 1850:
                                        raise TimeoutError("Frontend completion wait exceeded")
                                    print(
                                        f"WAIT {route} {genre}: "
                                        f"{time.perf_counter() - started:.0f}s",
                                        flush=True,
                                    )
                            assert (
                                page.locator("#result-heading").inner_text()
                                == "Your MIDI is ready."
                            )
                            observed = time.perf_counter() - started
                            with page.expect_download() as download_info:
                                page.get_by_role("link", name="Download MIDI", exact=True).click()
                            download_info.value.save_as(str(midi_path))
                            download_elapsed = time.perf_counter() - started
                            if trial == 1:
                                page.screenshot(path=str(target / (stem + ".png")), full_page=True)
                        else:
                            started = time.perf_counter()
                            status, job = get_json(api + "/api/generations", payload)
                            submission_seconds = time.perf_counter() - started
                            assert status == 202
                            next_progress = started + 30
                            while True:
                                _, job = get_json(api + job["status_url"])
                                if job["status"] in ("completed", "failed"):
                                    break
                                if time.perf_counter() > next_progress:
                                    print(
                                        f"WAIT {route} {genre}: "
                                        f"{time.perf_counter() - started:.0f}s",
                                        flush=True,
                                    )
                                    next_progress += 30
                                if time.perf_counter() - started > 1850:
                                    raise TimeoutError("Backend completion wait exceeded")
                                time.sleep(0.2)
                            if job["status"] != "completed":
                                raise RuntimeError(job["error"])
                            observed = time.perf_counter() - started
                            status, data, _ = request(api + job["download_url"])
                            assert status == 200
                            midi_path.write_bytes(data)
                            download_elapsed = time.perf_counter() - started
                        _, job = get_json(api + job["status_url"])
                        assert job["status"] == "completed"
                        _, metadata = get_json(api + job["metadata_url"])
                        assert midi_path.read_bytes().startswith(b"MThd")
                        _, authoritative_midi, _ = request(api + job["download_url"])
                        assert midi_path.read_bytes() == authoritative_midi
                        assert metadata["checkpoint_sha256"] == sha
                        assert metadata["checkpoint_epoch"] == args.expected_epoch
                        assert abs(metadata["actual_duration_seconds"] - 60) < 0.01
                        assert abs(metadata["midi_integrity"]["duration_seconds"] - 60) < 0.01
                        assert metadata["midi_integrity"]["dangling_on"] == 0
                        assert metadata["midi_integrity"]["unmatched_off"] == 0
                        save_json(target / (stem + ".metadata.json"), metadata)
                        row = {
                            "route": route,
                            "genre": genre,
                            "trial": trial,
                            "first_inference_on_loaded_model": len(report["runs"]) == 0,
                            "midi_path": str(midi_path.relative_to(output)),
                            "metadata_path": str(
                                (target / (stem + ".metadata.json")).relative_to(output)
                            ),
                            "job": job,
                            "submit_seconds": round(submission_seconds, 4),
                            "queue_seconds": round(job["started_at"] - job["created_at"], 4),
                            "worker_seconds": round(job["completed_at"] - job["started_at"], 4),
                            "generation_seconds": metadata["generation_seconds"],
                            "observed_ready_seconds": round(observed, 4),
                            "download_saved_seconds": round(download_elapsed, 4),
                            "seed": metadata["seed"],
                            "notes": metadata["notes"],
                            "tokens": metadata["tokens"],
                        }
                        report["runs"].append(row)
                        save_json(output / "benchmark.json", report)
                        print(
                            f"DONE {route} {genre} trial {trial}: "
                            f"generation={metadata['generation_seconds']:.3f}s "
                            f"ready={observed:.3f}s notes={metadata['notes']} "
                            f"tokens={metadata['tokens']}",
                            flush=True,
                        )
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            page.screenshot(path=str(output / "frontend_mobile.png"), full_page=True)
            browser.close()
        report["summary"] = []
        for genre in ("classical", "romantic", "rock_pop"):
            for route in ("frontend", "backend"):
                rows = [r for r in report["runs"] if r["genre"] == genre and r["route"] == route]
                report["summary"].append(
                    {
                        "genre": genre,
                        "route": route,
                        "n": len(rows),
                        **{
                            name + "_" + stat: round(fn(r[name] for r in rows), 4)
                            for name in (
                                "generation_seconds",
                                "worker_seconds",
                                "observed_ready_seconds",
                            )
                            for stat, fn in (
                                ("median", statistics.median),
                                ("min", min),
                                ("max", max),
                            )
                        },
                    }
                )
        report["completed_utc"] = datetime.now(UTC).isoformat()
        assert not report["browser_errors"], report["browser_errors"]
        print("EVALUATION COMPLETE", flush=True)
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        save_json(output / "benchmark.json", report)
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
