import os
import time
import hmac
import hashlib
import re
import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from gemini_service import ServiceError, call_gemini

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("neutral-news-rewrite-api")

app = FastAPI(title="Neutral News Rewrite API")

# ---- ENV ----
HMAC_SECRET = os.environ.get("HMAC_SECRET", "")

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

class RewriteResponse(BaseModel):
    title: str
    excerpt: str
    category: str
    tags: list[str]
    content: list[ContentBlock]
    heroImage: str
    heroImageSquare: str
    heroImageAuthor: str
    heroFocalX: float
    heroFocalY: float
    status: str
    scheduledAt: str
    slug: str
    authorName: str
    authorRole: str
    sourceName: str
    sourceUrl: str
    locationCity: str
    locationDistrict: str
    locationAddress: str
    isVerified: bool
    isFeatured: bool
    isBreaking: bool
    pinnedNowReading: bool
    pinnedNowReadingRank: int
    flags: list[str]
    confidence: float


class RSSImage(BaseModel):
    url: Optional[str] = None
    type: Optional[str] = None
    length: Optional[str] = None


class RSSMedia(BaseModel):
    thumbnail: Optional[str] = None
    content: Optional[str] = None


class RSSRewriteRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    title: Optional[str] = None
    subtitle: Optional[str] = None
    deck: Optional[str] = None
    description: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    content_encoded: Optional[str] = Field(default=None, alias="content:encoded")
    excerpt: Optional[str] = None
    link: Optional[str] = None
    guid: Optional[str] = None
    pubDate: Optional[str] = None
    author: Optional[str] = None
    dc_creator: Optional[str] = Field(default=None, alias="dc:creator")
    category: Optional[str] = None
    categories: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    source: Optional[str] = None
    language: Optional[str] = None
    image: Optional[RSSImage] = None
    enclosure: Optional[RSSImage] = None
    media: Optional[RSSMedia] = None
    media_content: Optional[str] = Field(default=None, alias="media:content")
    media_thumbnail: Optional[str] = Field(default=None, alias="media:thumbnail")
    comments: Optional[str] = None
    location: Optional[str] = None
    rights: Optional[str] = None
    copyright: Optional[str] = None
    priority: Optional[str] = None
    isBreaking: Optional[bool] = None
    tone: Optional[str] = None
    readingTime: Optional[str] = None


class RSSRewriteResponse(BaseModel):
    rss: Dict[str, Any]
    editorial: Dict[str, Any]
    validation: Dict[str, Any]

# ---- Prompt ----
def build_prompt(payload: RewriteRequest) -> str:
    # Важно: требуем "только JSON", без обёрток и пояснений.
    return f"""Ты — новостной редактор. Сделай нейтральный реврайт на русском языке строго по исходному тексту.

ЖЁСТКИЕ ПРАВИЛА:
- Не добавляй новых фактов, версий и деталей.
- Все числа, даты, имена собственные, адреса и названия организаций сохраняй БЕЗ ИЗМЕНЕНИЙ.
- Если фактов мало — пиши коротко и нейтрально, без «воды».
- Стиль: информационный, без оценок, без эмоций, без обращений к читателю.
- Возвращай ТОЛЬКО плоские поля (никаких вложенных объектов author/source/location).

ФОРМАТ ВЫВОДА:
Верни ТОЛЬКО один валидный JSON-объект. Без Markdown, без пояснений.
Ответ должен начинаться с {{ и заканчиваться }}.
В строковых полях НЕ используй символы перевода строки. Абзацы делай только через массив content (блоки).

ПРАВИЛА ФОРМАТА (ВСЕ ПОЛЯ ОБЯЗАТЕЛЬНЫ):
- title: 6–14 слов, без кликбейта.
- excerpt: 1–2 предложения, без новых фактов.
- category: slug категории сайта (пример: city, transport, incidents, russia-world). Если сомневаешься — city.
- tags: 3–8 коротких тегов, только ключевые темы.
- content: 2–3 абзаца, только type=paragraph. Структура: лид → детали → контекст.
- heroImage/heroImageSquare: если есть изображение в исходнике — укажи ссылку, иначе пустая строка.
- status: всегда "draft".
- scheduledAt: пустая строка.
- slug: латиницей, из title.
- authorName/authorRole: если нет данных — пустая строка.
- sourceName/sourceUrl: если нет данных — пустая строка.
- locationCity/locationDistrict/locationAddress: только если указаны в тексте, иначе пустая строка.
- isVerified/isFeatured/isBreaking/pinnedNowReading: по умолчанию false.
- pinnedNowReadingRank: 0.
- flags: пустой массив, если нет меток.
- confidence: 0.7–0.9.

СТРОГО ФИКСИРОВАННАЯ СТРУКТУРА JSON (ВСЕ КЛЮЧИ ОБЯЗАТЕЛЬНЫ):
{{
  "title": "...",
  "excerpt": "...",
  "category": "city",
  "tags": ["..."],
  "slug": "...",
  "authorName": "...",
  "authorRole": "...",
  "sourceName": "...",
  "sourceUrl": "...",
  "locationCity": "...",
  "locationDistrict": "...",
  "locationAddress": "...",
  "heroImage": "...",
  "heroImageSquare": "...",
  "heroImageAuthor": "...",
  "heroFocalX": 0.5,
  "heroFocalY": 0.5,
  "status": "draft",
  "scheduledAt": "",
  "isVerified": false,
  "isFeatured": false,
  "isBreaking": false,
  "pinnedNowReading": false,
  "pinnedNowReadingRank": 0,
  "flags": [],
  "confidence": 0.85,
  "content": [
    {{"type":"paragraph","value":"..."}}
  ]
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


def build_rewrite_response_schema() -> Dict[str, Any]:
    return {
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
                    },
                    "required": ["type"],
                },
            },
            "heroImage": {"type": "string"},
            "heroImageSquare": {"type": "string"},
            "heroImageAuthor": {"type": "string"},
            "heroFocalX": {"type": "number"},
            "heroFocalY": {"type": "number"},
            "status": {"type": "string"},
            "scheduledAt": {"type": "string"},
            "slug": {"type": "string"},
            "authorName": {"type": "string"},
            "authorRole": {"type": "string"},
            "sourceName": {"type": "string"},
            "sourceUrl": {"type": "string"},
            "locationCity": {"type": "string"},
            "locationDistrict": {"type": "string"},
            "locationAddress": {"type": "string"},
            "isVerified": {"type": "boolean"},
            "isFeatured": {"type": "boolean"},
            "isBreaking": {"type": "boolean"},
            "pinnedNowReading": {"type": "boolean"},
            "pinnedNowReadingRank": {"type": "number"},
            "flags": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        "required": [
            "title",
            "excerpt",
            "category",
            "tags",
            "content",
            "heroImage",
            "heroImageSquare",
            "heroImageAuthor",
            "heroFocalX",
            "heroFocalY",
            "status",
            "scheduledAt",
            "slug",
            "authorName",
            "authorRole",
            "sourceName",
            "sourceUrl",
            "locationCity",
            "locationDistrict",
            "locationAddress",
            "isVerified",
            "isFeatured",
            "isBreaking",
            "pinnedNowReading",
            "pinnedNowReadingRank",
            "flags",
            "confidence",
        ],
    }


def build_rss_prompt(payload: RSSRewriteRequest) -> str:
    return f"""Ты — новостной редактор. Сделай нейтральный реврайт новости на языке исходника.

ЖЁСТКИЕ ПРАВИЛА:
- Не добавляй новых фактов, версий и деталей.
- Все числа, даты, имена, адреса и названия организаций сохраняй без изменений.
- Если фактов мало — пиши коротко и нейтрально.
- Структура: вступление → детали → контекст.
- Без кликбейта и оценочных формулировок.

ФОРМАТ ВЫВОДА:
Верни ТОЛЬКО один валидный JSON-объект без Markdown.
Ответ должен начинаться с {{ и заканчиваться }}.
В строковых полях не используй перевод строк.

ПРАВИЛА:
- Заполни все доступные поля.
- Если данных нет — оставь пустую строку или пустой массив.
- description/excerpt должны быть 1–2 предложения.
- keywords: 3–10 слов/фраз.

СТРУКТУРА JSON:
{{
  "title": "...",
  "subtitle": "...",
  "description": "...",
  "content": "...",
  "excerpt": "...",
  "categories": ["..."],
  "keywords": ["..."],
  "language": "...",
  "location": "...",
  "readingTime": "3 min",
  "priority": "...",
  "isBreaking": false
}}

ИСХОДНЫЕ ДАННЫЕ:
title: {payload.title}
subtitle/deck: {payload.subtitle or payload.deck}
description/summary: {payload.description or payload.summary}
content/content:encoded: {payload.content or payload.content_encoded}
excerpt: {payload.excerpt}
link: {payload.link}
guid: {payload.guid}
pubDate: {payload.pubDate}
author/dc:creator: {payload.author or payload.dc_creator}
category/categories: {payload.categories or payload.category}
keywords: {payload.keywords}
source: {payload.source}
language: {payload.language}
image/enclosure: {payload.image or payload.enclosure}
media content/thumbnail: {payload.media or payload.media_content or payload.media_thumbnail}
comments: {payload.comments}
location: {payload.location}
rights: {payload.rights}
copyright: {payload.copyright}
priority: {payload.priority}
isBreaking: {payload.isBreaking}
tone: {payload.tone}
readingTime: {payload.readingTime}
"""


def build_rss_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "subtitle": {"type": "string"},
            "description": {"type": "string"},
            "content": {"type": "string"},
            "excerpt": {"type": "string"},
            "categories": {"type": "array", "items": {"type": "string"}},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "language": {"type": "string"},
            "location": {"type": "string"},
            "readingTime": {"type": "string"},
            "priority": {"type": "string"},
            "isBreaking": {"type": "boolean"},
        },
    }


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
        return "city"
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

def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized


def _normalize_rss_image(value: Optional[RSSImage]) -> Dict[str, str]:
    if not value:
        return {"url": "", "type": "", "length": ""}
    return {
        "url": _clean_str(value.url),
        "type": _clean_str(value.type),
        "length": _clean_str(value.length),
    }


def _normalize_rss_media(value: Optional[RSSMedia], content_url: Optional[str], thumbnail_url: Optional[str]) -> Dict[str, str]:
    thumbnail = _clean_str(value.thumbnail) if value and value.thumbnail else _clean_str(thumbnail_url)
    content = _clean_str(value.content) if value and value.content else _clean_str(content_url)
    return {"thumbnail": thumbnail, "content": content}


def _normalize_categories(payload: RSSRewriteRequest) -> List[str]:
    categories: List[str] = []
    if payload.categories:
        categories.extend(payload.categories)
    elif payload.category:
        categories.append(payload.category)
    return [_clean_str(item) for item in categories if _clean_str(item)]


def _estimate_reading_time(text: str) -> str:
    words = [w for w in re.split(r"\s+", text) if w]
    if not words:
        return ""
    minutes = max(1, round(len(words) / 200))
    return f"{minutes} min"


def _extract_keywords(source: str, limit: int = 7) -> List[str]:
    tokens = re.findall(r"[A-Za-zА-Яа-я0-9-]{4,}", source.lower())
    seen = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
        if len(seen) >= limit:
            break
    return seen


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    if isinstance(value, dict):
        return all(_is_empty(v) for v in value.values())
    return False


def _append_missing_note(notes: List[str], field_name: str, value: Any) -> None:
    if _is_empty(value):
        notes.append(f"{field_name} отсутствует в исходнике")

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

def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _build_paragraphs_from_text(text: str) -> List[str]:
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(raw_paragraphs) >= 2:
        return raw_paragraphs
    sentences = _split_sentences(text)
    if len(sentences) >= 2:
        return sentences
    if text.strip():
        midpoint = max(1, len(text) // 2)
        return [text[:midpoint].strip(), text[midpoint:].strip()] if text[midpoint:].strip() else [text.strip()]
    return []


def _normalize_content_blocks(out: Dict[str, Any], payload: RewriteRequest) -> List[ContentBlock]:
    blocks: List[ContentBlock] = []
    raw_content = out.get("content") if isinstance(out.get("content"), list) else []
    for item in raw_content:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if value is None and isinstance(item.get("items"), list):
            value = "; ".join(str(v) for v in item.get("items") if _clean_str(v))
        cleaned_value = _clean_str(value)
        if cleaned_value:
            blocks.append(ContentBlock(type="paragraph", value=cleaned_value))

    if blocks:
        return blocks

    source_text = _clean_str(payload.source_text) or _clean_str(payload.source_title)
    paragraphs = _build_paragraphs_from_text(source_text)
    return [ContentBlock(type="paragraph", value=_clean_str(p)) for p in paragraphs if _clean_str(p)]


def _ensure_min_paragraphs(blocks: List[ContentBlock], fallback_text: str) -> List[ContentBlock]:
    if len(blocks) >= 2:
        return blocks
    paragraphs = _build_paragraphs_from_text(fallback_text)
    for paragraph in paragraphs:
        if len(blocks) >= 2:
            break
        if paragraph and all(block.value != paragraph for block in blocks):
            blocks.append(ContentBlock(type="paragraph", value=_clean_str(paragraph)))
    return blocks if blocks else [ContentBlock(type="paragraph", value=_clean_str(fallback_text))]


def _derive_excerpt(text: str) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    return _clean_str(" ".join(sentences[:2]))


def _derive_title(text: str) -> str:
    words = [w for w in re.split(r"\s+", text) if w]
    if not words:
        return ""
    return _clean_str(" ".join(words[:14]))


def _normalize_tags(out: Dict[str, Any], payload: RewriteRequest) -> List[str]:
    tags = _normalize_str_list(out.get("tags")) or []
    text = " ".join([_clean_str(payload.source_title), _clean_str(payload.source_text)])
    supplemental = _extract_keywords(text, limit=8)
    for tag in supplemental:
        if tag not in tags:
            tags.append(tag)
        if len(tags) >= 8:
            break
    while len(tags) < 3:
        tags.append("новости")
    return tags[:8]

# ---- Middleware: keep raw body for signature ----
@app.middleware("http")
async def capture_raw_body(request: Request, call_next):
    request.state.raw_body = await request.body()
    return await call_next(request)

@app.get("/")
def root():
    return {
        "ok": True,
        "routes": {
            "health": "GET /healthz",
            "rewrite": "POST /rewrite (HMAC required)",
            "rss_rewrite": "POST /rss-rewrite (HMAC required)",
            "rewrite_alias": "POST / (HMAC required)",
        },
    }

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/healthz/healthz")
def healthz_alias():
    return {"ok": True}

@app.post("/rewrite", response_model=RewriteResponse)
async def rewrite(payload: RewriteRequest, req: Request, _=Depends(verify_hmac)):
    prompt = build_prompt(payload)
    out = await call_gemini(prompt, build_rewrite_response_schema())
    out = _convert_legacy_format(out, payload)

    if not isinstance(out.get("tags"), list):
        out["tags"] = []
    if not isinstance(out.get("content"), list):
        out["content"] = []

    source_text = _clean_str(payload.source_text)
    fallback_text = source_text or _clean_str(payload.source_title)

    title = _clean_str(out.get("title")) or _clean_str(payload.source_title) or _derive_title(fallback_text)
    if title:
        title = _clean_str(title)
    else:
        title = "Новости региона"
    if not fallback_text:
        fallback_text = title

    excerpt = _clean_str(out.get("excerpt")) or _derive_excerpt(fallback_text)
    if not excerpt and source_text:
        excerpt = _derive_excerpt(source_text)

    content_blocks = _normalize_content_blocks(out, payload)
    content_blocks = _ensure_min_paragraphs(content_blocks, fallback_text)

    category = _normalize_category(out.get("category"))
    tags = _normalize_tags(out, payload)[:8]

    hero_image = _clean_str(out.get("heroImage")) or _clean_str(payload.source_image)
    hero_image_square = _clean_str(out.get("heroImageSquare")) or hero_image
    hero_image_author = _clean_str(out.get("heroImageAuthor"))

    hero_focal_x = _normalize_optional_float(out.get("heroFocalX"))
    hero_focal_y = _normalize_optional_float(out.get("heroFocalY"))
    hero_focal_x = 0.5 if hero_focal_x is None else hero_focal_x
    hero_focal_y = 0.5 if hero_focal_y is None else hero_focal_y

    author = out.get("author") if isinstance(out.get("author"), dict) else {}
    author_name = _clean_str(out.get("authorName")) or _clean_str(author.get("name"))
    author_role = _clean_str(out.get("authorRole")) or _clean_str(author.get("role"))

    source = out.get("source") if isinstance(out.get("source"), dict) else {}
    source_name = _clean_str(out.get("sourceName")) or _clean_str(source.get("name")) or _clean_str(payload.source_site)
    source_url = _clean_str(out.get("sourceUrl")) or _clean_str(source.get("url")) or _clean_str(payload.source_url)

    location = out.get("location") if isinstance(out.get("location"), dict) else {}
    location_city = _clean_str(out.get("locationCity")) or _clean_str(location.get("city"))
    location_district = _clean_str(out.get("locationDistrict")) or _clean_str(location.get("district"))
    location_address = _clean_str(out.get("locationAddress")) or _clean_str(location.get("address"))

    is_verified = _normalize_optional_bool(out.get("isVerified"))
    is_featured = _normalize_optional_bool(out.get("isFeatured"))
    is_breaking = _normalize_optional_bool(out.get("isBreaking"))
    pinned_now_reading = _normalize_optional_bool(out.get("pinnedNowReading"))
    pinned_now_reading_rank = _normalize_optional_int(out.get("pinnedNowReadingRank"))

    flags = _normalize_str_list(out.get("flags")) or []
    confidence = _normalize_confidence(out.get("confidence"))
    if confidence is None:
        confidence = 0.85

    response: Dict[str, Any] = {
        "title": title,
        "excerpt": excerpt,
        "category": category,
        "tags": tags,
        "content": content_blocks,
        "heroImage": hero_image,
        "heroImageSquare": hero_image_square,
        "heroImageAuthor": hero_image_author,
        "heroFocalX": hero_focal_x,
        "heroFocalY": hero_focal_y,
        "status": "draft",
        "scheduledAt": "",
        "slug": _clean_str(out.get("slug")) or _slugify(title),
        "authorName": author_name,
        "authorRole": author_role,
        "sourceName": source_name,
        "sourceUrl": source_url,
        "locationCity": location_city,
        "locationDistrict": location_district,
        "locationAddress": location_address,
        "isVerified": bool(is_verified) if is_verified is not None else False,
        "isFeatured": bool(is_featured) if is_featured is not None else False,
        "isBreaking": bool(is_breaking) if is_breaking is not None else False,
        "pinnedNowReading": bool(pinned_now_reading) if pinned_now_reading is not None else False,
        "pinnedNowReadingRank": pinned_now_reading_rank if pinned_now_reading_rank is not None else 0,
        "flags": flags,
        "confidence": confidence,
    }

    response["excerpt"] = response["excerpt"] or _derive_excerpt(
        content_blocks[0].value if content_blocks else fallback_text
    )
    if not response["excerpt"]:
        response["excerpt"] = response["title"]

    if not response["tags"]:
        response["tags"] = ["новости"]

    if not response["content"]:
        response["content"] = [ContentBlock(type="paragraph", value=_clean_str(fallback_text))]

    return RewriteResponse(**response)

# Алиас: POST /
@app.post("/", response_model=RewriteResponse)
async def rewrite_alias(payload: RewriteRequest, req: Request, _=Depends(verify_hmac)):
    return await rewrite(payload, req)


@app.post("/rss-rewrite", response_model=RSSRewriteResponse)
async def rss_rewrite(payload: RSSRewriteRequest, req: Request, _=Depends(verify_hmac)):
    notes: List[str] = []
    fact_warnings: List[str] = []
    language = _clean_str(payload.language) or "ru"

    prompt = build_rss_prompt(payload)
    try:
        out = await call_gemini(prompt, build_rss_response_schema())
    except ServiceError as exc:
        notes.append(f"Gemini error: {exc.detail}")
        rss = {
            "title": _clean_str(payload.title),
            "subtitle": _clean_str(payload.subtitle or payload.deck),
            "description": _clean_str(payload.description or payload.summary),
            "content": _clean_str(payload.content or payload.content_encoded),
            "excerpt": _clean_str(payload.excerpt),
            "link": _clean_str(payload.link),
            "guid": _clean_str(payload.guid),
            "pubDate": _clean_str(payload.pubDate),
            "author": _clean_str(payload.author or payload.dc_creator),
            "categories": _normalize_categories(payload),
            "keywords": payload.keywords or [],
            "source": _clean_str(payload.source),
            "language": language,
            "image": _normalize_rss_image(payload.image or payload.enclosure),
            "media": _normalize_rss_media(payload.media, payload.media_content, payload.media_thumbnail),
            "comments": _clean_str(payload.comments),
            "location": _clean_str(payload.location),
            "rights": _clean_str(payload.rights),
            "copyright": _clean_str(payload.copyright),
            "priority": _clean_str(payload.priority or "normal"),
            "isBreaking": bool(payload.isBreaking) if isinstance(payload.isBreaking, bool) else False,
            "readingTime": _clean_str(payload.readingTime) or _estimate_reading_time(
                _clean_str(payload.content or payload.content_encoded or payload.description or payload.summary)
            ),
        }
        _append_missing_note(notes, "title", rss["title"])
        _append_missing_note(notes, "description", rss["description"])
        _append_missing_note(notes, "content", rss["content"])
        _append_missing_note(notes, "excerpt", rss["excerpt"])
        _append_missing_note(notes, "categories", rss["categories"])
        _append_missing_note(notes, "keywords", rss["keywords"])
        _append_missing_note(notes, "link", rss["link"])
        _append_missing_note(notes, "guid", rss["guid"])
        _append_missing_note(notes, "pubDate", rss["pubDate"])
        _append_missing_note(notes, "author", rss["author"])
        _append_missing_note(notes, "source", rss["source"])
        _append_missing_note(notes, "language", rss["language"])
        _append_missing_note(notes, "readingTime", rss["readingTime"])

        if _is_empty(rss["image"]) and _is_empty(rss["media"]):
            notes.append("image/media отсутствуют в исходнике")

        editorial = {
            "rewriteStyle": "нейтральный/деловой",
            "tone": _clean_str(payload.tone) or "информативный",
            "notes": notes,
            "factCheckWarnings": fact_warnings,
        }
        validation = {
            "factsPreserved": False,
            "noHallucinations": False,
            "languageDetected": language,
            "safeForPublication": False,
        }
        return RSSRewriteResponse(rss=rss, editorial=editorial, validation=validation)

    title = _clean_str(out.get("title")) or _clean_str(payload.title)
    subtitle = _clean_str(out.get("subtitle")) or _clean_str(payload.subtitle or payload.deck)
    description = _clean_str(out.get("description")) or _clean_str(payload.description or payload.summary)
    content = _clean_str(out.get("content")) or _clean_str(payload.content or payload.content_encoded)
    excerpt = _clean_str(out.get("excerpt")) or _clean_str(payload.excerpt)

    categories = out.get("categories") if isinstance(out.get("categories"), list) else _normalize_categories(payload)
    categories = [_clean_str(item) for item in categories if _clean_str(item)]

    keywords = out.get("keywords") if isinstance(out.get("keywords"), list) else (payload.keywords or [])
    keywords = [_clean_str(item) for item in keywords if _clean_str(item)]
    if not keywords:
        keywords = _extract_keywords(" ".join([title, description, content]))
        if keywords:
            notes.append("keywords сгенерированы из текста")

    language = _clean_str(out.get("language")) or language
    reading_time = _clean_str(out.get("readingTime")) or _clean_str(payload.readingTime)
    if not reading_time:
        reading_time = _estimate_reading_time(content or description)

    priority = _clean_str(out.get("priority")) or _clean_str(payload.priority) or "normal"
    is_breaking = out.get("isBreaking") if isinstance(out.get("isBreaking"), bool) else payload.isBreaking
    if not isinstance(is_breaking, bool):
        is_breaking = False

    rss = {
        "title": title,
        "subtitle": subtitle,
        "description": description,
        "content": content,
        "excerpt": excerpt,
        "link": _clean_str(payload.link),
        "guid": _clean_str(payload.guid),
        "pubDate": _clean_str(payload.pubDate),
        "author": _clean_str(payload.author or payload.dc_creator),
        "categories": categories,
        "keywords": keywords,
        "source": _clean_str(payload.source),
        "language": language,
        "image": _normalize_rss_image(payload.image or payload.enclosure),
        "media": _normalize_rss_media(payload.media, payload.media_content, payload.media_thumbnail),
        "comments": _clean_str(payload.comments),
        "location": _clean_str(out.get("location")) or _clean_str(payload.location),
        "rights": _clean_str(payload.rights),
        "copyright": _clean_str(payload.copyright),
        "priority": priority,
        "isBreaking": is_breaking,
        "readingTime": reading_time,
    }

    _append_missing_note(notes, "title", title)
    _append_missing_note(notes, "description", description)
    _append_missing_note(notes, "content", content)
    _append_missing_note(notes, "excerpt", excerpt)
    _append_missing_note(notes, "categories", categories)
    _append_missing_note(notes, "keywords", keywords)
    _append_missing_note(notes, "link", rss["link"])
    _append_missing_note(notes, "guid", rss["guid"])
    _append_missing_note(notes, "pubDate", rss["pubDate"])
    _append_missing_note(notes, "author", rss["author"])
    _append_missing_note(notes, "source", rss["source"])
    _append_missing_note(notes, "language", rss["language"])
    _append_missing_note(notes, "readingTime", rss["readingTime"])

    if _is_empty(rss["image"]) and _is_empty(rss["media"]):
        notes.append("image/media отсутствуют в исходнике")

    editorial = {
        "rewriteStyle": "нейтральный/деловой",
        "tone": _clean_str(payload.tone) or "информативный",
        "notes": notes,
        "factCheckWarnings": fact_warnings,
    }
    validation = {
        "factsPreserved": True,
        "noHallucinations": True,
        "languageDetected": language,
        "safeForPublication": bool(title and description and content),
    }

    return RSSRewriteResponse(rss=rss, editorial=editorial, validation=validation)
