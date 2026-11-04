from uuid import uuid4

import pytest

from musicgen.frontend.app import create_app
from musicgen.shared.storage import OutputStore


@pytest.mark.parametrize("job_id", ["../escape", "/tmp/outside", "abc", "a" * 33])
def test_output_paths_reject_invalid_ids(tmp_path, job_id):
    with pytest.raises(ValueError):
        OutputStore(tmp_path).path(job_id)


def test_output_paths_reject_symlinks(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    job_id = uuid4().hex
    (root / job_id).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        OutputStore(root).path(job_id)


def test_template_and_static_assets_work_without_checkpoint(settings):
    client = create_app(settings).test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert b'id="generation-form"' in page.data
    assert b'value="rock_pop"' in page.data
    assert client.get("/static/css/styles.css").status_code == 200
    assert client.get("/static/js/app.js").status_code == 200
    assert client.get("/health/live").json == {"status": "ok"}
    assert b"musicgen_http_requests_total" in client.get("/metrics").data
    assert "script-src 'self'" in page.headers["Content-Security-Policy"]
