FROM python:3.13-bookworm

WORKDIR /app

RUN pip install \
    poetry \
    # For building sentencepiece
    cmake

COPY pyproject.toml poetry.lock .
RUN poetry install --only main --no-root --no-directory

COPY . .
RUN poetry install --only main

ENV prod=true

EXPOSE 7860

ENTRYPOINT ["poetry", "run", "python", "main.py"]