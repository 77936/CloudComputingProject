FROM python:3.12-slim

WORKDIR /app

COPY deploy/vertex_predictor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy/vertex_predictor/app.py /app/app.py
COPY deploy/vertex_predictor/scorer.py /app/scorer.py
COPY rtad /app/rtad

ENV PORT=8080
CMD ["python", "app.py"]
