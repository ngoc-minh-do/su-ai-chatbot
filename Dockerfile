# =========================
# Build stage
# =========================
FROM python:3.13-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    # For building sentencepiece
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install only production dependencies first (no project code yet)
RUN uv sync --no-install-project && uv cache clean

# Copy full project and install remaining dependencies
COPY . .
RUN uv sync && uv cache clean

# Remove any temporary files to reduce size
RUN rm -rf /root/.cache /tmp/*

# # =========================
# # Runtime stage
# # =========================
# FROM python:3.13-slim AS runtime

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Install uv
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# ENV PATH="/root/.local/bin:${PATH}"

# WORKDIR /app

# Copy prebuilt virtualenv from builder
# COPY --from=builder /app/.venv .venv

# COPY . .

ENV prod=true

EXPOSE 7860

# Final cleanup (just in case)
# RUN rm -rf /root/.cache /tmp/* /var/lib/apt/lists/*

CMD ["uv", "run", "python", "main.py"]
