# Stage 1: Builder with GPU access
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

WORKDIR /app

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        cmake \
        pkg-config \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/* /root/.cache

ENV PATH="/root/.local/bin:${PATH}"
ENV CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=1"
ENV GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
ENV LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Sync dependencies
RUN uv sync --no-install-project \
    && uv cache clean \
    && rm -rf /root/.cache/pip /tmp/*

# Copy project and build
COPY . .
RUN uv sync --force \
    && uv cache clean \
    && rm -rf /root/.cache/pip /tmp/* \
    && apt-get purge -y build-essential cmake pkg-config \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Runtime image
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

WORKDIR /app

ENV PATH="/root/.local/bin:${PATH}"
ENV LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV prod=true

# Copy built files from builder
COPY --from=builder /app /app

EXPOSE 7860

CMD ["uv", "run", "python", "main.py"]
