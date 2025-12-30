import os
import time
import hmac
import hashlib
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx

app = FastAPI()

HMAC_SECRET = os.environ.get("HMAC_SECRET", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")


def _need_env(name: str, val: str):
    if not val:
        raise HTTPException(500, f"{name} is not set")


def _no_store_headers():
    # чтобы проверки здоровья не кэшировались где-нибудь по пути
    return {"Cache-Control": "no-store, max-age=0"}


def verify_hmac(raw_body: bytes, ts: str, sig: str):
    _need_env("HMAC_SECRET", HMAC_SECRET)

    if not ts or not sig:
        raise HTTPException(401, "Missing signature headers")

    try:
        t = int(ts)
    except Exception:
        raise HTTPException(401, "Bad timestamp")

    # 5-minute window
    if abs(int(time.time()) - t) > 300:
        raise HTTPException(401, "Stale request")

    msg = ts.encode("utf-8") + b"." + raw_body
    expected = hmac.new(HMAC_SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected, sig):
        raise HTTPException(401, "Bad signature")


@app.get("/")
def root():
    return JSONResponse(
        {
            "ok": True,
            "routes": {
                "health": "GET /healthz (and /health, /_health, /ping)",
                "rewrite": "POST /rewrite (HMAC required)",
                "rewrite_alias": "POST / (HMAC required)",
            },
        },
        headers=_no_store_headers(),
    )


# Несколько health-эндпоинтов: выбирайте любой в админке
@app.get("/healthz")
@app.get("/health")
@app.get("/_health")
@app.get("/ping")
def health():
    return JSONResponse({"ok": True}, headers=_no_store_headers())


def build_prompt(payload: dict) -> str:
    return f"""\
Ты — редактор новостей. Переформулируй материал нейтрально.
НЕЛЬЗЯ добавлять факты, которых нет в исходном тексте.
НЕЛЬЗЯ копировать исходник длинными кусками. Нужен полный перефраз.
Выводи ТОЛЬКО валидный JSON без Markdown и без комментариев.

Вход:
source_title: {payload.get("source_title","")}
source_text: {payload.get("source_text","")}
source_url: {payload.get("source_url","")}
source_site: {payload.get("source_site","")}
source_published_at: {payload.get("source_published_at","")}
source_image: {payload.get("source_image","")}
region_hint: {payload.get("region_hint","")}

Верни строго JSON такого вида:
{{
  "ok": true,
  "confidence": 0.0,
  "flags": [],
  "newsItem": {{
    "id": "ai-xxxxxxxxxxxx",
    "slug": "kebab-case-xxxx",
    "title": "",
    "excerpt": "",
    "content": [
      {{ "type":"paragraph", "value":"" }}
    ],
    "category": {{ "slug":"city", "title":"Город" }},
    "tags": [],
    "author": {{ "name":"Редакция", "role":"Новости" }},
    "publishedAt": "",
    "heroImage": "",
    "readingTime": 1
  }}
}}

Правила:
- Не выдумывай факты. Если чего-то нет — "не уточняется".
- category.slug только из: city, transport, incidents, sports, events, real-estate.
- heroImage: если source_image пустой — ставь "".
- В content сделай 3–6 абзацев (paragraph). Список (list) — только если уместно.
""".strip()


async def call_gemini(prompt: str) -> dict:
    _need_env("GEMINI_API_KEY", GEMINI_API_KEY)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1800},
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=body)
        if r.status_code >= 400:
            raise HTTPException(502, f"Gemini error: {r.status_code} {r.text[:200]}")

        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]

        try:
            return json.loads(text)
        except Exception:
            raise HTTPException(502, "Model returned non-JSON")


@app.post("/rewrite")
async def rewrite(req: Request):
    raw = await req.body()
    verify_hmac(raw, req.headers.get("X-Timestamp", ""), req.headers.get("X-Signature", ""))

    payload = await req.json()
    prompt = build_prompt(payload)
    out = await call_gemini(prompt)

    if not isinstance(out, dict) or out.get("ok") is not True:
        raise HTTPException(502, "Bad output format")

    return JSONResponse(out)


# Алиас: если кто-то бьёт POST /
@app.post("/")
async def rewrite_alias(req: Request):
    return await rewrite(req)
