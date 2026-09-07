# AI Advent Challenge #9

Рабочий репозиторий заданий [AI Advent Challenge #9](https://mobiledeveloper.tech/ai_advent_9).

## Remote

```
origin  git@github-ai-challenge:AlekseyYakovlev/AI-Advent-Challenge-9.git
```

SSH-хост `github-ai-challenge` задан в `~/.ssh/config`.

## Синхронизация

```powershell
git add .
git commit -m "day N: краткое описание"
git push -u origin main
```

## Переключатель моделей LM Studio

Кратко:

1. Скопируйте `.env.example` → `.env`, укажите `LMSTUDIO_BASE_URL` (обычно `http://localhost:1234`).
2. `LMSTUDIO_UNLOAD_ON_SHUTDOWN=false` по умолчанию — при остановке приложения модель остаётся в VRAM; `true` — выгрузка в FastAPI lifespan.
3. Состояние модели глобальное (на весь процесс), не per-session.
4. При «Сервис временно недоступен» подождите ~2 минуты (cooldown Circuit Breaker).
5. В `model_id` нельзя передавать путь с `..` или пробелами.

Подробнее: [docs/06-lmstudio-model-switcher.md](docs/06-lmstudio-model-switcher.md).
