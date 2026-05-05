FROM python:3.12-slim

WORKDIR /app

COPY deploy/batch_job/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY rtad /app/rtad
COPY deploy/batch_job/run_batch_job.py /app/run_batch_job.py

CMD ["python", "run_batch_job.py"]
