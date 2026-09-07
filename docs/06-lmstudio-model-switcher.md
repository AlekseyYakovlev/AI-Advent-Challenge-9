# Переключатель моделей LM Studio

Глобальный state-manager + Circuit Breaker для load/unload моделей через
LM Studio control API (`/api/v0`).

## Настройка `.env`

Скопируйте `.env.example` → `.env` и при необходимости измените:

```env
LMSTUDIO_BASE_URL=http://localhost:1234
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_LOAD_TIMEOUT=120
LMSTUDIO_EMERGENCY_UNLOAD_TIMEOUT=5
LMSTUDIO_UNLOAD_ON_SHUTDOWN=false
LMSTUDIO_CIRCUIT_BREAKER_THRESHOLD=3
LMSTUDIO_CIRCUIT_BREAKER_COOLDOWN_SECONDS=120
```

## `LMSTUDIO_UNLOAD_ON_SHUTDOWN`

- `false` (по умолчанию) — при остановке FastAPI модель **не** выгружается из VRAM.
- `true` — в `lifespan` shutdown вызывается `unload_model`.

При запуске через `chainlit run` FastAPI lifespan **не** выполняется — флаг
действует только при старте через uvicorn/`app.main:app`.

## Поведение

- Состояние загруженной модели **глобальное** (process-wide), общее для всех сессий UI.
- При сообщении «Сервис временно недоступен» сработал Circuit Breaker —
  подождите около **2 минут** (`LMSTUDIO_CIRCUIT_BREAKER_COOLDOWN_SECONDS`).
- В `model_id` нельзя использовать path traversal (`..`, `/../`) и пробелы —
  такие значения отклоняются валидацией.
