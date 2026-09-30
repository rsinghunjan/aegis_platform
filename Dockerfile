FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AEGIS_AGENT_DATABASE_URL=sqlite:////data/aegis_agent.db \
    AEGIS_MEMORY_DATABASE_URL=sqlite:////data/aegis_agent.db

WORKDIR /app

RUN useradd --create-home --uid 10001 aegis \
    && mkdir -p /data \
    && chown aegis:aegis /data

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY --chown=aegis:aegis . /app
USER aegis

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/readyz', timeout=2)"

CMD ["uvicorn", "production:app", "--host", "0.0.0.0", "--port", "8000"]
