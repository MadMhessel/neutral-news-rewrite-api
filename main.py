import os
import json
import time
import hmac
import hashlib
import re
import logging
from typing import Optional, Dict, Any, List

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
LOG_MODEL_RAW_LIMIT = 1000
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

class ContentBlock(BaseModel):
    type: str
    value: Optional[str] = None
    items: Optional[List[str]] = None
    author: Optional[str] = None
    kind: Optional[str] = None
    title: Optional[str] = None

class FocalPoint(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None

class AuthorInfo(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None

class SourceInfo(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None

class LocationInfo(BaseModel):
    city: Optional[str] = None
    district: Optional[str] = None
    address: Optional[str] = None

class RewriteResponse(BaseModel):
    title: Optional[str] = None
    excerpt: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list[str]] = None
    content: Optional[list[ContentBlock]] = None
    heroImage: Optional[str] = None
    heroImageSquare: Optional[str] = None
    heroImageAuthor: Optional[str] = None
    heroFocalX: Optional[float] = None
    heroFocalY: Optional[float] = None
    heroFocal: Optional[FocalPoint] = None
    status: Optional[str] = None
    scheduledAt: Optional[str] = None
    slug: Optional[str] = None
    authorName: Optional[str] = None
    authorRole: Optional[str] = None
    sourceName: Optional[str] = None
    sourceUrl: Optional[str] = None
    author: Optional[AuthorInfo] = None
    source: Optional[SourceInfo] = None
    locationCity: Optional[str] = None
    locationDistrict: Optional[str] = None
    locationAddress: Optional[str] = None
    location: Optional[LocationInfo] = None
    isVerified: Optional[bool] = None
    isFeatured: Optional[bool] = None
    isBreaking: Optional[bool] = None
    pinnedNowReading: Optional[bool] = None
    pinnedNowReadingRank: Optional[int] = None
    flags: Optional[list[str]] = None
    confidence: Optional[float] = None

# ---- Prompt ----
def build_prompt(payload: RewriteRequest) -> str:
    # Важно: требуем "только JSON", без обёрток и пояснений.
    return f"""Ты — новостной редактор. Сделай нейтральный реврайт на русском языке строго по исходному тексту.

ЖЁСТКИЕ ПРАВИЛА:
- Не добавляй новых фактов, версий и деталей.
- Все числа, даты, имена собственные, адреса и названия организаций сохраняй БЕЗ ИЗМЕНЕНИЙ.
- Если фактов мало — пиши коротко и нейтрально, без «воды».
- Стиль: информационный, без оценок, без эмоций, без обращений к читателю.

ФОРМАТ ВЫВОДА:
Верни ТОЛЬКО один валидный JSON-объект. Без Markdown, без пояснений.
Ответ должен начинаться с {{ и заканчиваться }}.
В строковых полях НЕ используй символы перевода строки. Абзацы делай только через массив content (блоки).

ПРАВИЛА ФОРМАТА:
- Все поля опциональны. Если поле не задано — просто не включай его в JSON.
- title: до 110 символов
- excerpt: до 240 символов (1–2 предложения, отвечает на «что произошло?»)
- content: желательно 3+ блока
  - 1-й блок paragraph — лид (1–2 предложения)
  - затем 1–4 блока paragraph с деталями
  - при необходимости добавь 1 блок list (2–6 пунктов)
  - heading для подзаголовков
  - quote только если цитата реально есть в исходнике
  - divider допустим как разделитель
  - callout допустим при необходимости (kind: info|warning|important)
- tags: 3–7, короткие, без #, нижний регистр
- category: используй slug категории сайта (пример: city, transport, incidents, russia-world). Если сомневаешься — city
- heroImage: если в исходнике есть ссылка на изображение — поставь её
- flags: если есть редакционные метки — перечисли
- confidence: число от 0.0 до 1.0 — уверенность в корректности фактов

Поддерживаемые поля:
{{
  "title": "...",
  "excerpt": "...",
  "category": "slug",
  "tags": ["...", "..."],
  "heroImage": "...",
  "heroImageSquare": "...",
  "heroImageAuthor": "...",
  "heroFocalX": 0.5,
  "heroFocalY": 0.3,
  "heroFocal": {{"x": 0.5, "y": 0.3}},
  "status": "draft",
  "scheduledAt": "2024-01-01T12:00:00Z",
  "slug": "...",
  "authorName": "...",
  "authorRole": "...",
  "sourceName": "...",
  "sourceUrl": "...",
  "author": {{"name":"...","role":"..."}},
  "source": {{"name":"...","url":"..."}},
  "locationCity": "...",
  "locationDistrict": "...",
  "locationAddress": "...",
  "location": {{"city":"...","district":"...","address":"..."}},
  "isVerified": true,
  "isFeatured": false,
  "isBreaking": false,
  "pinnedNowReading": true,
  "pinnedNowReadingRank": 1,
  "content": [
    {{"type":"paragraph","value":"..."}}
  ],
  "flags": ["..."],
  "confidence": 0.0
}}

ИСХОДНЫЕ ДАННЫЕ:
Заголовок: {payload.source_title}
Текст: {payload.source_text}
Ссылка: {payload.source_url}
Сайт: {payload.source_site}
Дата/время публикации: {payload.source_published_at}
Изображение: {payload.source_image}
Регион: {payload.region_hint}
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

    # 2) извлекаем JSON между первой "{" и последней "}"
    obj_start = s.find("{")
    obj_end = s.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidate = s[obj_start:obj_end + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raw = s[:RESPONSE_MODEL_RAW_LIMIT]
    logger.error("Model returned non-JSON: %s", s[:LOG_MODEL_RAW_LIMIT])
    raise ServiceError(status_code=502, detail="Model returned non-JSON", raw=raw)

# ---- Gemini call ----
async def call_gemini(prompt: str) -> Dict[str, Any]:
    api_key = _need_env("GEMINI_API_KEY (or API_KEY)", GEMINI_API_KEY)
    model = GEMINI_MODEL.strip()
    # На всякий случай убираем лишний префикс, если кто-то вписал "models/..."
    model = model.replace("models/", "")

    endpoint_path = f"{GEMINI_ENDPOINT}/models/{model}:generateContent"
    url = f"{endpoint_path}?key={api_key}"
    logger.info("Gemini model: %s", model)
    logger.info("Gemini endpoint: %s", endpoint_path)

    # JSON Mode: просим JSON и задаём схему.
    # Даже если конкретная версия API проигнорирует эти поля, у нас есть строгий промт + запасной парсер.
    generation_config = {
        "temperature": 0.2,
        "maxOutputTokens": 4096,
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "excerpt": {"type": "string"},
                "category": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "content": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "value": {"type": "string"},
                            "items": {"type": "array", "items": {"type": "string"}},
                            "author": {"type": "string"},
                            "kind": {"type": "string"},
                            "title": {"type": "string"},
                        },
                        "required": ["type"],
                    },
                },
                "heroImage": {"type": "string"},
                "heroImageSquare": {"type": "string"},
                "heroImageAuthor": {"type": "string"},
                "heroFocalX": {"type": "number"},
                "heroFocalY": {"type": "number"},
                "heroFocal": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                },
                "status": {"type": "string"},
                "scheduledAt": {"type": "string"},
                "slug": {"type": "string"},
                "authorName": {"type": "string"},
                "authorRole": {"type": "string"},
                "sourceName": {"type": "string"},
                "sourceUrl": {"type": "string"},
                "author": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "role": {"type": "string"}},
                },
                "source": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "url": {"type": "string"}},
                },
                "locationCity": {"type": "string"},
                "locationDistrict": {"type": "string"},
                "locationAddress": {"type": "string"},
                "location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "district": {"type": "string"},
                        "address": {"type": "string"},
                    },
                },
                "isVerified": {"type": "boolean"},
                "isFeatured": {"type": "boolean"},
                "isBreaking": {"type": "boolean"},
                "pinnedNowReading": {"type": "boolean"},
                "pinnedNowReadingRank": {"type": "number"},
                "flags": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
        },
    }

    body = {
        "systemInstruction": {
            "parts": [{"text": "Верни строго один JSON-объект. Никакого Markdown. Начни с { и закончи }."}]
        },
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    timeout = httpx.Timeout(connect=10.0, read=90.0, write=90.0, pool=90.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=body)
    except httpx.TimeoutException as exc:
        logger.exception("Gemini timeout: %s", exc)
        raise ServiceError(status_code=502, detail=f"Gemini timeout: {exc}")
    except httpx.RequestError as exc:
        logger.exception("Gemini request error: %s", exc)
        raise ServiceError(status_code=502, detail=f"Gemini request error: {exc}")

    logger.info("Gemini status: %s", r.status_code)

    if r.status_code >= 400:
        truncated = r.text[:LOG_GEMINI_ERROR_BODY_LIMIT]
        logger.error("Gemini error %s: %s", r.status_code, truncated)
        raise ServiceError(
            status_code=502,
            detail=f"Gemini error: {r.status_code}: {truncated}",
            raw=truncated,
        )

    try:
        data = r.json()
    except json.JSONDecodeError:
        truncated = r.text[:RESPONSE_MODEL_RAW_LIMIT]
        logger.error("Gemini response was not JSON: %s", truncated[:LOG_GEMINI_ERROR_BODY_LIMIT])
        raise ServiceError(
            status_code=502,
            detail="Gemini returned non-JSON response",
            raw=truncated,
        )

    try:
        parts = data["candidates"][0]["content"]["parts"]
    except Exception:
        data_preview = json.dumps(data)[:RESPONSE_MODEL_RAW_LIMIT]
        logger.error("Unexpected Gemini response structure: %s", data_preview[:LOG_GEMINI_ERROR_BODY_LIMIT])
        raise ServiceError(
            status_code=502,
            detail="Unexpected Gemini response structure",
            raw=data_preview,
        )

    if not isinstance(parts, list):
        data_preview = json.dumps(data)[:RESPONSE_MODEL_RAW_LIMIT]
        logger.error("Unexpected Gemini response structure: %s", data_preview[:LOG_GEMINI_ERROR_BODY_LIMIT])
        raise ServiceError(
            status_code=502,
            detail="Unexpected Gemini response structure",
            raw=data_preview,
        )

    text_chunks = []
    for part in parts:
        if isinstance(part, dict):
            text_value = part.get("text")
            if isinstance(text_value, str) and text_value.strip():
                text_chunks.append(text_value)

    if not text_chunks:
        data_preview = json.dumps(data)[:RESPONSE_MODEL_RAW_LIMIT]
        logger.error("Unexpected Gemini response structure: %s", data_preview[:LOG_GEMINI_ERROR_BODY_LIMIT])
        raise ServiceError(
            status_code=502,
            detail="Unexpected Gemini response structure",
            raw=data_preview,
        )

    text = "\n".join(text_chunks)
    logger.info("Gemini response text length: %s", len(text))

    parsed = _parse_json_strict_or_fallback(text)

    # Нормализация типов
    if not isinstance(parsed.get("tags", []), list):
        parsed["tags"] = []
    if not isinstance(parsed.get("content", []), list):
        parsed["content"] = []

    return parsed

def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()

def _clean_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    cleaned = _clean_str(value)
    return cleaned or None

def _normalize_category(value: Any) -> Optional[str]:
    normalized = _clean_str(value)
    if not normalized:
        return None
    if normalized and re.fullmatch(r"[a-z0-9-]+", normalized):
        return normalized
    return "city"

def _normalize_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(confidence, 0.0), 1.0)

def _normalize_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _normalize_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _normalize_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    return None

def _normalize_str_list(value: Any) -> Optional[list[str]]:
    if not isinstance(value, list):
        return None
    return [_clean_str(item) for item in value if _clean_str(item)]

def _normalize_focal_point(value: Any) -> Optional[FocalPoint]:
    if not isinstance(value, dict):
        return None
    x = _normalize_optional_float(value.get("x"))
    y = _normalize_optional_float(value.get("y"))
    if x is None and y is None:
        return None
    return FocalPoint(x=x, y=y)

def _normalize_author_info(value: Any) -> Optional[AuthorInfo]:
    if not isinstance(value, dict):
        return None
    name = _clean_optional_str(value.get("name"))
    role = _clean_optional_str(value.get("role"))
    if not name and not role:
        return None
    return AuthorInfo(name=name, role=role)

def _normalize_source_info(value: Any) -> Optional[SourceInfo]:
    if not isinstance(value, dict):
        return None
    name = _clean_optional_str(value.get("name"))
    url = _clean_optional_str(value.get("url"))
    if not name and not url:
        return None
    return SourceInfo(name=name, url=url)

def _normalize_location_info(value: Any) -> Optional[LocationInfo]:
    if not isinstance(value, dict):
        return None
    city = _clean_optional_str(value.get("city"))
    district = _clean_optional_str(value.get("district"))
    address = _clean_optional_str(value.get("address"))
    if not city and not district and not address:
        return None
    return LocationInfo(city=city, district=district, address=address)

def _convert_legacy_format(data: Dict[str, Any], payload: RewriteRequest) -> Dict[str, Any]:
    if not data.get("content") and (data.get("summary") or data.get("text")):
        excerpt = _clean_str(data.get("summary", ""))
        raw_text = data.get("text", "")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", str(raw_text)) if p.strip()]
        content = [{"type": "paragraph", "value": _clean_str(p)} for p in paragraphs]
        data["excerpt"] = excerpt
        data["content"] = content
        if not data.get("heroImage"):
            data["heroImage"] = payload.source_image or ""
    return data

def _normalize_content_block(item: Dict[str, Any]) -> Optional[ContentBlock]:
    block_type = _clean_str(item.get("type"))
    if not block_type:
        return None

    if block_type == "list":
        raw_items = item.get("items") or []
        items = [_clean_str(v) for v in raw_items if _clean_str(v)]
        return ContentBlock(type=block_type, items=items)

    if block_type == "quote":
        return ContentBlock(
            type=block_type,
            value=_clean_str(item.get("value")),
            author=_clean_str(item.get("author")),
        )

    if block_type == "callout":
        return ContentBlock(
            type=block_type,
            kind=_clean_str(item.get("kind")),
            title=_clean_str(item.get("title")),
            value=_clean_str(item.get("value")),
        )

    if block_type == "divider":
        return ContentBlock(type=block_type)

    return ContentBlock(type=block_type, value=_clean_str(item.get("value")))

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

@app.get("/healthz/healthz")
def healthz_alias():
    return {"ok": True}

@app.post("/rewrite", response_model=RewriteResponse, response_model_exclude_none=True)
async def rewrite(payload: RewriteRequest, req: Request, _=Depends(verify_hmac)):
    prompt = build_prompt(payload)
    out = await call_gemini(prompt)
    out = _convert_legacy_format(out, payload)

    response: Dict[str, Any] = {}

    response["title"] = _clean_optional_str(out.get("title"))
    response["excerpt"] = _clean_optional_str(out.get("excerpt"))
    response["category"] = _normalize_category(out.get("category"))

    if "tags" in out:
        response["tags"] = _normalize_str_list(out.get("tags"))

    if "content" in out and isinstance(out.get("content"), list):
        response["content"] = [
            block
            for block in (
                _normalize_content_block(item)
                for item in (out.get("content") or [])
                if isinstance(item, dict)
            )
            if block is not None
        ]

    response["heroImage"] = _clean_optional_str(out.get("heroImage"))
    response["heroImageSquare"] = _clean_optional_str(out.get("heroImageSquare"))
    response["heroImageAuthor"] = _clean_optional_str(out.get("heroImageAuthor"))
    response["heroFocalX"] = _normalize_optional_float(out.get("heroFocalX"))
    response["heroFocalY"] = _normalize_optional_float(out.get("heroFocalY"))
    response["heroFocal"] = _normalize_focal_point(out.get("heroFocal"))
    response["status"] = _clean_optional_str(out.get("status"))
    response["scheduledAt"] = _clean_optional_str(out.get("scheduledAt"))
    response["slug"] = _clean_optional_str(out.get("slug"))
    response["authorName"] = _clean_optional_str(out.get("authorName"))
    response["authorRole"] = _clean_optional_str(out.get("authorRole"))
    response["sourceName"] = _clean_optional_str(out.get("sourceName"))
    response["sourceUrl"] = _clean_optional_str(out.get("sourceUrl"))
    response["author"] = _normalize_author_info(out.get("author"))
    response["source"] = _normalize_source_info(out.get("source"))
    response["locationCity"] = _clean_optional_str(out.get("locationCity"))
    response["locationDistrict"] = _clean_optional_str(out.get("locationDistrict"))
    response["locationAddress"] = _clean_optional_str(out.get("locationAddress"))
    response["location"] = _normalize_location_info(out.get("location"))
    response["isVerified"] = _normalize_optional_bool(out.get("isVerified"))
    response["isFeatured"] = _normalize_optional_bool(out.get("isFeatured"))
    response["isBreaking"] = _normalize_optional_bool(out.get("isBreaking"))
    response["pinnedNowReading"] = _normalize_optional_bool(out.get("pinnedNowReading"))
    response["pinnedNowReadingRank"] = _normalize_optional_int(out.get("pinnedNowReadingRank"))

    if "flags" in out:
        response["flags"] = _normalize_str_list(out.get("flags"))

    response["confidence"] = _normalize_confidence(out.get("confidence"))

    cleaned = {key: value for key, value in response.items() if value is not None}
    return RewriteResponse(**cleaned)

# Алиас: POST /
@app.post("/", response_model=RewriteResponse, response_model_exclude_none=True)
async def rewrite_alias(payload: RewriteRequest, req: Request, _=Depends(verify_hmac)):
    return await rewrite(payload, req)
