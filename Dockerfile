FROM python:3.10-slim-bullseye

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir --no-deps -r requirements.txt