# Security Policy

## Reporting a Vulnerability

Please report vulnerabilities via GitHub Security Advisories: `https://github.com/arfaouiahmed1/signalrank/security/advisories/new` or email `ahmedarfaoui2000@gmail.com`. Do not open public issues for sensitive reports. We aim to respond within 48h.

## Supported Versions

| Version | Supported |
|---|---|
| `main` (latest) | ✅ |
| `v0.1.x` | ✅ |

## Security Measures in SignalRank

**API (`backend/app/main.py:12`):**
- **CORS hardened:** `ALLOW_CREDENTIALS=False`, `allow_origins` from `ALLOWED_ORIGINS` env (default `https://arfaouiahmed1.github.io,https://ahmedarfaoui99-signalrank.hf.space,http://localhost:3000,http://localhost:8000`). No `*` with credentials. See `app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False, allow_methods=["GET","POST"], allow_headers=["Content-Type","Authorization"])`.
- **Security headers:** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy: geolocation=(), microphone=(), camera=()`, `Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; connect-src 'self' https://ahmedarfaoui99-signalrank.hf.space https://arfaouiahmed1.github.io; img-src 'self' data: https:; frame-ancestors 'none'`, `Strict-Transport-Security` when HTTPS.
- **Rate limiting:** `slowapi` `Limiter(key_func=get_remote_address, default_limits=["60/minute"])` + `@limiter.limit("30/minute")` on `/rank`, `/rank/json` and `"10/minute"` on `/ingest`. Returns `429`.
- **Input validation:** `RankRequest` validates `cv_text 20..20000 chars`, `k 1..50`, `method in {embedding,vector,bm25,hybrid,hybrid+ce,hybrid+lgbm}`. `/rank` checks `file ≤5 MB`, `content-type in {text,pdf,octet-stream}`, `cv_text ≤20000`. `/ingest` checks `jobs_path` inside `data/` (path traversal guard via `is_relative_to`).
- **PDF parsing:** `pypdf` with `try/except` → `400`.
- **No secrets in code:** all `HF_TOKEN`, `DOCKERHUB_*`, `KAGGLE_*`, `DATABASE_URL` from env/secrets. `.env` is gitignored, `.env.example` has placeholders. Gitleaks runs in CI.
- **Dependencies pinned:** `backend/requirements.txt` pinned, Dependabot weekly for pip/npm/docker/actions, `pip-audit` + `npm audit` in CI.

**Frontend (`frontend/src/App.jsx:16`):**
- No secrets baked; `VITE_API_URL` is public `https://ahmedarfaoui99-signalrank.hf.space` for Pages, `http://localhost:8000` locally. Handles `429` with user message.
- Built with `Vite` base `/signalrank/` for Pages, served via `actions/deploy-pages` with `X-Frame-Options: DENY` etc via `_headers`.

**Infra:**
- **Docker:** `infra/Dockerfile.api:7` `apt-get` minimal (`libgomp1 curl`), `pip --no-cache-dir`, non-root? (TODO: add `USER` — currently root, tracked). `sbom:true` + `provenance:true` + `cache` in `docker.yml:40`.
- **CI/CD (`security.yml:1`):**
  - **CodeQL** for `python` + `javascript` on push/PR/schedule.
  - **Dependency Review** on PR (fail on high).
  - **pip-audit** for `requirements.txt` + `requirements-full.txt`.
  - **npm audit** (`--audit-level=high`).
  - **gitleaks** (full history, `GITHUB_TOKEN`).
  - **Trivy** (`aquasecurity/trivy-action@0.24.0` on `signalrank:scan` → `sarif` → CodeQL upload, `CRITICAL,HIGH`).

**GitHub Pages:**
- `pages.yml:15` builds with `VITE_API_URL` → HF Space, `base: /signalrank/`, uploads `frontend/dist` (404 fallback + `_headers`), deploys via `actions/deploy-pages@v4` with `contents: read, pages: write, id-token: write` + `concurrency: pages`. Repo homepage `https://arfaouiahmed1.github.io/signalrank/` set via `gh repo edit`.

## Hardening TODO (tracked, not blocking demo)

- [ ] Run backend as non-root `USER appuser` in `infra/Dockerfile.api` + `hf/space/Dockerfile`
- [ ] Add ` Dependabot` auto-merge for patch
- [ ] Add `Content-Security-Policy-Report-Only` + reporting endpoint
- [ ] Add per-IP `/rank` concurrency limit (e.g., 5 concurrent) + `slowapi` storage `redis` for multi-replica
- [ ] Sign Docker images with `cosign`

See `SECURITY.md` history for fixes: CORS `*` + `allow_credentials=True` → `False` + allowlist, 5 MB file cap, `is_relative_to` guard, rate limits.
