FROM astral/uv:latest AS uv

FROM nvidia/cuda:13.4.3-base-ubuntu26.04

WORKDIR /app

# Install runtime dependencies (libpq for psycopg, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# Copy uv binary from the official image (no curl-based install needed)
COPY --from=uv /uv /usr/local/bin/uv

ENV GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
ENV LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Sync dependencies only (cached layer)
RUN uv sync --frozen --no-install-project --no-dev \
    && uv cache clean \
    && rm -rf /tmp/*

# Copy project and install it (deps already cached)
COPY . .
RUN uv sync --frozen --no-dev \
    && uv cache clean \
    && rm -rf /tmp/*

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

CMD ["python", "main.py"]