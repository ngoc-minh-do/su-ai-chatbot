# Building a CUDA-enabled Wheel (`.whl`) for `llama-cpp-python`

## 1. Prepare GPU-enabled Environment

Make sure you have:

- NVIDIA GPU + driver
- CUDA Toolkit installed (e.g., 12.8)
- Python 3.13
- Build tools: `cmake`, `build-essential`, `pkg-config`

> Recommended: use a GPU Docker container:

```bash
docker run --rm --gpus all -it -v /docker/su-ai-chat/build:/dist -w /dist nvidia/cuda:12.8.1-devel-ubuntu24.04 bash
```

---

## 2. Install Python Dependencies

Inside the container:

```bash
apt update -y
apt install -y \
    software-properties-common

add-apt-repository ppa:deadsnakes/ppa

apt install -y \
    python3.13 python3.13-dev curl build-essential cmake pkg-config git

curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

python3.13 -m pip install --upgrade pip setuptools wheel build
```

---

## 3. Clone `llama-cpp-python`

```bash
git clone --recursive https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
```

- `--recursive` ensures submodules (like `llama.cpp`) are cloned.

---

## 4. Set CUDA Build Environment

```bash
export CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=1"
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

---

## 5. Build the Wheel

```bash
python3.13 -m build --wheel
```

- Generates `.whl` in `dist/` folder.
- Uses `pyproject.toml` and modern PEP 517 build system.
- Compiles `libggml-cuda.so` for CUDA support.

---

## 6. Install and Test

```bash
pip install dist/llama_cpp_python-*.whl
python3.13 -c "from llama_cpp import Llama; print(Llama().get_device())"
```

- Should detect a CUDA-enabled GPU.

---

## 7. Notes / Tips

- Building **requires GPU libraries**; cannot link CUDA fully with stubs only.
- Environment variables like `CMAKE_ARGS` are critical for CUDA backend.
- Use a **GPU container or GPU-enabled host** to avoid linker errors.
- `.whl` can be reused in Docker runtime images or private PyPI.