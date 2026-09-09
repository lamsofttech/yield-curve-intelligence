FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0

RUN useradd --create-home --uid 10001 researcher
WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=researcher:researcher src ./src
COPY --chown=researcher:researcher data/sample ./data/sample
COPY --chown=researcher:researcher data/historical ./data/historical
COPY --chown=researcher:researcher app.py ./app.py
COPY --chown=researcher:researcher pyproject.toml ./pyproject.toml
USER researcher

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/')"

CMD ["python", "app.py"]
