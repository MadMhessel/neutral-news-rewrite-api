import json
import logging
import os
import re
from typing import Any, Dict, Optional

import httpx

LOG_GEMINI_ERROR_BODY_LIMIT = 1000
LOG_MODEL_RAW_LIMIT = 1000
RESPONSE_MODEL_RAW_LIMIT = 2000

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY") or ""
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta")

logger = logging.getLogger("neutral-news-rewrite-api.gemini")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class ServiceError(Exception):
    def __init__(self, status_code: int, detail: str, raw: Optional[str] = None) -> None:
        self.status_code = status_code
        self.detail = detail
        self.raw = raw


def _parse_json_strict_or_fallback(text: str) -> Dict[str, Any]:
    """Пытаемся распарсить JSON максимально строго, но если модель всё же добавила мусор — вытащим JSON из текста."""
    s = text.strip()

    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    obj_start = s.find("{")
    obj_end = s.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidate = s[obj_start : obj_end + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raw = s[:RESPONSE_MODEL_RAW_LIMIT]
    logger.error("Model returned non-JSON: %s", s[:LOG_MODEL_RAW_LIMIT])
    raise ServiceError(status_code=502, detail="Model returned non-JSON", raw=raw)


def _need_env(name: str, val: str) -> str:
    if not val:
        raise ServiceError(status_code=500, detail=f"Missing required env: {name}")
    return val


async def call_gemini(prompt: str, response_schema: Dict[str, Any]) -> Dict[str, Any]:
    api_key = _need_env("GEMINI_API_KEY (or API_KEY)", GEMINI_API_KEY)
    model = GEMINI_MODEL.strip().replace("models/", "")

    endpoint_path = f"{GEMINI_ENDPOINT}/models/{model}:generateContent"
    url = f"{endpoint_path}?key={api_key}"
    logger.info("Gemini model: %s", model)
    logger.info("Gemini endpoint: %s", endpoint_path)

    generation_config = {
        "temperature": 0.2,
        "maxOutputTokens": 4096,
        "responseMimeType": "application/json",
        "responseSchema": response_schema,
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

    return _parse_json_strict_or_fallback(text)
