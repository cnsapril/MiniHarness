# Fixed test-runtime image. Project sources are mounted read-only at runtime.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir 'openai>=1.0,<4'

WORKDIR /workspace
