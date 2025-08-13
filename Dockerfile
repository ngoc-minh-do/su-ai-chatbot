FROM python:3.13-slim

WORKDIR /app

# Install uv first (no build deps needed)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache

ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./

# Install build dependencies temporarily for deps compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
    && uv sync --no-install-project \
    && uv cache clean \
    && rm -rf /root/.cache/pip /tmp/* \
    # Remove build dependencies
    && apt-get purge -y build-essential cmake pkg-config \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy full project
COPY . .

# Install remaining dependencies (if any) without build deps
RUN uv sync \
    && uv cache clean \
    && rm -rf /root/.cache/pip /tmp/*

ENV prod=true
EXPOSE 7860

CMD ["uv", "run", "python", "main.py"]
