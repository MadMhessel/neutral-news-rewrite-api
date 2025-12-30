# Cloud Run: Gemini rewrite service (FastAPI)

## Endpoints
- GET `/healthz` -> `{"ok": true}`
- POST `/rewrite` -> rewrite JSON (HMAC required)
- POST `/` -> alias of `/rewrite` (HMAC required)
- GET `/` -> routes info

## Environment variables (Cloud Run)
- `GEMINI_API_KEY` (required)
- `HMAC_SECRET` (required, must match the site)
- `GEMINI_MODEL` (optional, default: `gemini-1.5-pro`)

## Deploy example
Build & deploy (one of many ways):
```bash
gcloud run deploy neutral-news-ai-editor \
  --source . \
  --region us-west1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=YOUR_KEY,HMAC_SECRET=YOUR_SECRET,GEMINI_MODEL=gemini-1.5-pro
```

## Test
Health:
```bash
curl -i https://YOUR.run.app/healthz
```

`/rewrite` requires HMAC headers (`X-Timestamp`, `X-Signature`) so testing is usually done from your site (rewrite.php).
