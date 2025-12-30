# neutral-news-rewrite-api (Cloud Run)

Мини‑API для безопасного реврайта новостей через Gemini API.

## Переменные окружения (Cloud Run → Variables & secrets → Environment variables)
- GEMINI_API_KEY — ключ Gemini API (можно также задать как API_KEY)
- HMAC_SECRET — секрет для подписи запросов (обязателен)
- GEMINI_MODEL — модель (по умолчанию gemini-2.5-flash)
- GEMINI_ENDPOINT — (необязательно) https://generativelanguage.googleapis.com/v1beta

## Эндпойнты
- GET / — проверка, список маршрутов
- GET /healthz — health-check
- POST /rewrite — реврайт (нужны заголовки подписи)
- POST / — алиас на /rewrite (нужны заголовки подписи)

## Формат подписи (HMAC-SHA256)
Подписывается строка:  "<timestamp>.<raw_json_body>"
Заголовки:
- X-Timestamp: <unix_seconds>
- X-Signature: <hex_hmac_sha256>

## Быстрый тест (bash)
```bash
SERVICE_URL="https://YOUR-SERVICE-URL"
HMAC_SECRET="YOUR_SECRET"
TS=$(date +%s)

BODY='{"source_title":"Тест","source_text":"Тестовый текст без фактов.","source_url":"","source_site":"","source_published_at":"","source_image":"","region_hint":"Нижний Новгород"}'

SIG=$(TS="$TS" BODY="$BODY" HMAC_SECRET="$HMAC_SECRET" python3 - <<'PY'
import hmac, hashlib, os
ts=os.environ["TS"].encode()
body=os.environ["BODY"].encode()
secret=os.environ["HMAC_SECRET"].encode()
msg=ts+b"."+body
print(hmac.new(secret,msg,hashlib.sha256).hexdigest())
PY
)

curl -i "$SERVICE_URL/rewrite"   -H "Content-Type: application/json"   -H "X-Timestamp: $TS"   -H "X-Signature: $SIG"   -d "$BODY"
```

Если получаете `{"detail":"Model returned non-JSON"}` — значит модель вернула текст не в JSON.
В этой версии включён JSON Mode + есть запасной парсер, чтобы не падать на лишнем тексте.
