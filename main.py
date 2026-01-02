import os
import json
import time
import hmac
import hashlib
import re
import logging
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("neutral-news-rewrite-api")

app = FastAPI(title="Neutral News Rewrite API")

# ---- ENV ----
# Поддерживаем оба имени переменной, чтобы не ловить 500 из-за одной буквы.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY") or ""
HMAC_SECRET = os.environ.get("HMAC_SECRET", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta")

LOG_GEMINI_ERROR_BODY_LIMIT = 1000
LOG_MODEL_RAW_LIMIT = 2000
RESPONSE_MODEL_RAW_LIMIT = 2000

class ServiceError(Exception):
    def __init__(self, status_code: int, detail: str, raw: Optional[str] = None) -> None:
        self.status_code = status_code
        self.detail = detail
        self.raw = raw

@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    payload = {"detail": exc.detail}
    if exc.raw is not None:
        payload["raw"] = exc.raw
    return JSONResponse(status_code=exc.status_code, content=payload)

def _need_env(name: str, val: str) -> str:
    if not val:
        raise HTTPException(status_code=500, detail=f"Missing required env: {name}")
    return val

# ---- HMAC ----
def verify_hmac(req: Request) -> None:
    secret = _need_env("HMAC_SECRET", HMAC_SECRET).encode()

    ts = req.headers.get("X-Timestamp", "")
    sig = req.headers.get("X-Signature", "")
    if not ts or not sig:
        raise HTTPException(status_code=401, detail="Missing signature headers")

    # защита от повторов/задержек
    try:
        ts_int = int(ts)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid timestamp")

    now = int(time.time())
    if abs(now - ts_int) > 300:
        raise HTTPException(status_code=401, detail="Timestamp out of window")

    raw_body = getattr(req.state, "raw_body", b"")
    msg = ts.encode() + b"." + raw_body
    expected = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(status_code=401, detail="Bad signature")

# ---- Input/Output ----
class RewriteRequest(BaseModel):
    source_title: str = Field(default="")
    source_text: str = Field(...)
    source_url: Optional[str] = Field(default="")
    source_site: Optional[str] = Field(default="")
    source_published_at: Optional[str] = Field(default="")
    source_image: Optional[str] = Field(default="")
    region_hint: Optional[str] = Field(default="")

class RewriteResponse(BaseModel):
    title: str
    text: str
    summary: str
    tags: list[str]
    category: str
    image_prompt: str

# ---- Prompt ----
def build_prompt(payload: RewriteRequest) -> str:
    # Важно: требуем "только JSON", без обёрток и пояснений.
    return f"""Ты редактор новостного сайта. Задача: переписать новость нейтрально, без выдумок, сохраняя факты.
Язык: русский.

Жёсткие правила:
1) НИЧЕГО не добавляй от себя. Если в тексте нет факта — не придумывай.
2) Не упоминай исходный источник, не пиши "по данным" и т.п., если этого нет в тексте.
3) Не используй оценочные суждения и эмоции. Никакой агитации.
4) Не меняй смысл. Исправляй стиль, структуру, ясность.
5) Не вставляй ссылки, если их нет в исходных данных.
6) Верни результат СТРОГО в виде JSON-объекта без лишнего текста, без Markdown, без ```.
7) Ответ должен начинаться с символа {{ и заканчиваться }}.

Входные данные:
Заголовок: {payload.source_title}
Регион (подсказка): {payload.region_hint}
Дата/время (если есть): {payload.source_published_at}
Текст: {payload.source_text}

Требуемый JSON (строго эти ключи):
{{
  "title": "короткий заголовок 60–90 знаков",
  "text": "полный текст новости 900–1800 знаков, абзацы разделяй \n\n",
  "summary": "1–2 предложения 200–350 знаков",
  "tags": ["3–8 коротких тегов, без решёток"],
  "category": "одна категория: Город|Происшествия|Политика|Экономика|Культура|Спорт|Общество|Транспорт|Недвижимость|Погода|Другое",
  "image_prompt": "краткое описание изображения без текста на картинке, фотореализм"
}}
"""

# ---- JSON extraction fallback ----
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def _parse_json_strict_or_fallback(text: str) -> Dict[str, Any]:
    """Пытаемся распарсить JSON максимально строго, но если модель всё же добавила мусор — вытащим JSON из текста."""
    s = text.strip()

    # Частый случай: модель обернула в ```json ... ```
    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    # 1) строго
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) вытащить максимально крупный JSON-объект {...}
    candidate = s
    for _ in range(5):
        obj_start = candidate.find("{")
        obj_end = candidate.rfind("}")
        if obj_start == -1 or obj_end == -1 or obj_end <= obj_start:
            break
        sliced = candidate[obj_start:obj_end + 1].strip()
        try:
            return json.loads(sliced)
        except Exception:
            if sliced == candidate:
                break
            candidate = sliced

    raw = s[:RESPONSE_MODEL_RAW_LIMIT]
    logger.error("Model returned non-JSON: %s", s[:LOG_MODEL_RAW_LIMIT])
    raise ServiceError(status_code=502, detail="Model returned non-JSON", raw=raw)

# ---- Gemini call ----
async def call_gemini(prompt: str) -> Dict[str, Any]:
    api_key = _need_env("GEMINI_API_KEY (or API_KEY)", GEMINI_API_KEY)
    model = GEMINI_MODEL.strip()
    # На всякий случай убираем лишний префикс, если кто-то вписал "models/..."
    model = model.replace("models/", "")

    url = f"{GEMINI_ENDPOINT}/models/{model}:generateContent?key={api_key}"
    logger.info("Gemini model: %s", model)
    logger.info("Gemini endpoint: %s", url)

    # JSON Mode: просим JSON и задаём схему.
    # Даже если конкретная версия API проигнорирует эти поля, у нас есть строгий промт + запасной парсер.
    generation_config = {
        "temperature": 0.2,
        "maxOutputTokens": 2200,
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "text": {"type": "string"},
                "summary": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "category": {"type": "string"},
                "image_prompt": {"type": "string"},
            },
            "required": ["title", "text", "summary", "tags", "category", "image_prompt"],
        },
    }

    body = {
        "systemInstruction": {
            "parts": [{"text": "Верни строго JSON, без Markdown, без пояснений. Начинай с { и заканчивай }."}]
        },
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(url, json=body)
    except httpx.HTTPError as exc:
        logger.exception("HTTPX error calling Gemini: %s", exc)
        raise ServiceError(status_code=502, detail=f"HTTPX error: {exc}")

    if r.status_code >= 400:
        truncated = r.text[:LOG_GEMINI_ERROR_BODY_LIMIT]
        logger.error("Gemini error %s: %s", r.status_code, truncated)
        raise ServiceError(
            status_code=502,
            detail=f"Gemini error: {r.status_code}: {truncated}",
        )

    try:
        data = r.json()
    except json.JSONDecodeError:
        truncated = r.text[:LOG_GEMINI_ERROR_BODY_LIMIT]
        logger.error("Gemini response was not JSON: %s", truncated)
        raise ServiceError(status_code=502, detail="Unexpected Gemini response structure")

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        data_preview = json.dumps(data)[:LOG_GEMINI_ERROR_BODY_LIMIT]
        logger.error("Unexpected Gemini response structure: %s", data_preview)
        raise ServiceError(status_code=502, detail="Unexpected Gemini response structure")

    parsed = _parse_json_strict_or_fallback(text)

    # Нормализация типов
    if not isinstance(parsed.get("tags", []), list):
        parsed["tags"] = []

    return parsed

# ---- Middleware: keep raw body for signature ----
@app.middleware("http")
async def capture_raw_body(request: Request, call_next):
    request.state.raw_body = await request.body()
    return await call_next(request)

@app.get("/")
def root():
    return {"ok": True, "routes": {"health": "GET /healthz", "rewrite": "POST /rewrite (HMAC required)", "rewrite_alias": "POST / (HMAC required)"}}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/rewrite", response_model=RewriteResponse)
async def rewrite(payload: RewriteRequest, req: Request, _=Depends(verify_hmac)):
    prompt = build_prompt(payload)
    out = await call_gemini(prompt)

    # Подстраховка: если модель вернула лишние ключи — игнорируем.
    return RewriteResponse(
        title=str(out.get("title", "")).strip(),
        text=str(out.get("text", "")).strip(),
        summary=str(out.get("summary", "")).strip(),
        tags=[str(t).strip() for t in (out.get("tags") or []) if str(t).strip()],
        category=str(out.get("category", "Другое")).strip() or "Другое",
        image_prompt=str(out.get("image_prompt", "")).strip(),
    )

# Алиас: POST /
@app.post("/", response_model=RewriteResponse)
async def rewrite_alias(payload: RewriteRequest, req: Request, _=Depends(verify_hmac)):
    return await rewrite(payload, req)
