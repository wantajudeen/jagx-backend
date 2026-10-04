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
API base: `https://YOUR-SERVICE.onrender.com` (keep path `/chat` + header `x-api-key`)
