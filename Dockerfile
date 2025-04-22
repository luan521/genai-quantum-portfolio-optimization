FROM python:3.10-slim-bullseye

WORKDIR /code

COPY requirements.txt .
COPY setup.py .
COPY README.md .
COPY ai_quantum/ ./ai_quantum/

RUN pip install --no-cache-dir --no-deps -r requirements.txt
RUN pip install .