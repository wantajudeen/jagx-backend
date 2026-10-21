# JagX AI Backend 7.0.0

Created by **JagX & JRILICENSE**.

## Why you were stuck on 1.1.3
Render Docker was running `main.py` (old 1.1.3). Upgrades in `app.py` never started.

This repo uses **`main.py` = full v7 backend** so `CMD ["python", "main.py"]` is correct.

## Deploy on Render
1. Web Service → connect **wantajudeen/jagx-backend**
2. Runtime: **Docker**
3. Env vars: `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN` (optional), `JAGX_PERMANENT_KEYS`, `JAGX_ADMIN_SECRET`
4. After deploy open `/health` — expect `"version":"7.0.0"`

## Point the site
API base: `https://YOUR-SERVICE.onrender.com`.

- Trusted clients may continue using `POST /chat` with the `x-api-key` header.
- Public browser clients should use `POST /public-chat`; it has a guest IP rate limit (`JAGX_PUBLIC_CHAT_RPM`, default 12 requests/minute) and does not require a permanent key in JavaScript.
- Set `JAGX_CORS_ORIGINS` to a comma-separated list of exact allowed web origins when deploying.
- Chat transcript training logs are off by default; opt in only when the privacy notice and retention plan are ready with `JAGX_TRAINING_DATA_ENABLED=true`.
- Do not expose permanent API keys or provider keys in the frontend.
