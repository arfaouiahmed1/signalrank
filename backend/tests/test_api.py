import json
from pathlib import Path

def test_health():
    from backend.app.main import app
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_jobs_endpoint(tmp_path, monkeypatch):
    from backend.app.main import app
    from fastapi.testclient import TestClient
    # Use tmp file to avoid polluting real data/raw
    sample = [{"id":1,"title":"AI Engineer","company":"Test","location":"Remote","description":"Python FastAPI","skills":["Python"],"source":"test"}]
    tmp_jobs = tmp_path / "jobs.jsonl"
    tmp_jobs.write_text("\n".join(json.dumps(j) for j in sample), encoding="utf-8")
    # Monkeypatch loader to use tmp file via env or direct patch: we check fallback by ensuring real file exists then test
    # Instead, just ensure API handles missing/empty gracefully and test with real file restored after
    orig = Path("data/raw/jobs.jsonl")
    backup = None
    if orig.exists():
        backup = orig.read_text(encoding="utf-8")
    try:
        orig.parent.mkdir(parents=True, exist_ok=True)
        orig.write_text("\n".join(json.dumps(j) for j in sample), encoding="utf-8")
        c = TestClient(app)
        r = c.get("/jobs?limit=10")
        assert r.status_code == 200
        assert r.json()["total"] >= 1
    finally:
        if backup is not None:
            orig.write_text(backup, encoding="utf-8")
        elif orig.exists():
            orig.unlink()
