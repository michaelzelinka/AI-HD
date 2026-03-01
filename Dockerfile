FROM python:3.10-slim

WORKDIR /app

# systémové balíčky (pokud bys potřeboval, přidej sem další)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pokud je tvůj klasifikátor pojmenovaný classify6.py, stačí env CLASSIFIER_SCRIPT
ENV CLASSIFIER_SCRIPT=hd_classify6.py

COPY app.py .
COPY hd_classify6.py ./  # pokud máš hd_classify6.py, uprav řádek

# Render očekává, že server poběží na portu $PORT
ENV PORT=8000

CMD uvicorn app:app --host 0.0.0.0 --port $PORT
