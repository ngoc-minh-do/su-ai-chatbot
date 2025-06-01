FROM python:3.13-bookworm

WORKDIR /app

RUN pip install \
    poetry \
    # For building sentencepiece
    cmake


RUN wget https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp312-cp312-linux_x86_64.whl -P local_packages

COPY pyproject.toml poetry.lock .
RUN poetry install --only main --no-root --no-directory

COPY . .
RUN poetry install --only main

ENV prod=true

EXPOSE 7860

ENTRYPOINT ["poetry", "run", "python", "main.py"]