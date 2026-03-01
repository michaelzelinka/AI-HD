# ================================
# Dockerfile for AI Helpdesk API
# ================================

# 1) Base image
FROM python:3.10-slim

# 2) System deps (volitelné – ponechat minimální stopu)
#    Pokud tvůj klasifikátor vyžaduje system knihovny (např. locales, libmagic atd.),
#    přidej je sem přes apt-get.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 3) Workdir
WORKDIR /app

# 4) Copy a instalace závislostí (využij cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 5) Copy zdrojů aplikace (FastAPI + klasifikátor skript)
#    Ujisti se, že v kontextu buildu jsou i soubory jako hd_classify6.py atd.
COPY . /app

# 6) (Volitelné) – bezpečnější runtime user
# RUN useradd -m apiuser
# USER apiuser

# 7) Expose – Render to nevyžaduje, ale neuškodí
EXPOSE 8000

# 8) Entrypoint/CMD – SHELL FORM, aby se expandoval $PORT od Renderu
#    Render nastavuje $PORT automaticky; uvicorn poběží na 0.0.0.0:$PORT
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
