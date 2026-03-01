FROM python:3.10-slim

WORKDIR /app

# 1) Závislosti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Aplikace
COPY app.py .
COPY hd_classify6.py .

# 3) Konfigurace a start
ENV PORT=8000
ENV CLASSIFIER_SCRIPT=hd_classify6.py
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
