FROM python:3.13-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
        build-essential \
        cmake \
        pkg-config \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/* /root/.cache

ENV PATH="/root/.local/bin:${PATH}"

RUN curl -O -L wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run \
    && sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit \
    && rm cuda_12.8.1_570.124.06_linux.run

ENV PATH=/usr/local/cuda-12.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Copy dependency files for caching
COPY pyproject.toml uv.lock ./

# Install dependencies with build tools
RUN CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON" \
    uv sync --no-install-project \
    && uv cache clean \
    && rm -rf /root/.cache/pip /tmp/*

# Copy full project
COPY . .

# Install project dependencies without build tools
RUN CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON" \
    uv sync --force \
    && uv cache clean \
    && rm -rf /root/.cache/pip /tmp/* \
    && apt-get purge -y build-essential cmake pkg-config \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV prod=true

EXPOSE 7860

CMD ["uv", "run", "python", "main.py"]
