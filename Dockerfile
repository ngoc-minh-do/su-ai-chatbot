FROM python:3.13-slim-bookworm

WORKDIR /app

COPY pyproject.toml poetry.lock .
RUN pip install poetry && poetry install --only main --no-root --no-directory

COPY . .
RUN poetry install --only main

ENV prod=true

EXPOSE 7860

ENTRYPOINT ["poetry", "run", "python", "main.py"]