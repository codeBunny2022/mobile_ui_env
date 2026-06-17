FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md run_eval.py ./
COPY mobile_ui_env ./mobile_ui_env
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["pytest", "-q"]
