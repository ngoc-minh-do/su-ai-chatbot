# Llama.cpp on CUDA: wheel, knobs, and troubleshooting

## How the CUDA-enabled wheel is provided

The project uses the **prebuilt** `llama-cpp-python` CUDA wheel — there is no
from-source build anymore.

- `pyproject.toml` points uv at a dedicated wheel index:
  `llama-cpp-python-wheel` → `https://abetlen.github.io/llama-cpp-python/whl/cu132`
- `uv.lock` pins the exact wheel version (currently `0.3.36`, built for CUDA 13.2)
- The Dockerfile installs it with a plain `uv sync --frozen` on an
  `nvidia/cuda:13.2.0-base-ubuntu24.04` base image — no `cmake`/CUDA Toolkit needed

To move to another CUDA version, change the index URL (e.g. `cu124`, `cu121`)
and refresh the lock:

```bash
uv lock --upgrade llama-cpp-python
uv sync
```

## Runtime knobs

| Variable | Default | Effect |
|---|---|---|
| `N_GPU_LAYERS` | 32 | Transformer layers offloaded to GPU; `32` = full offload for a 32-layer model (`-1` = everything incl. embeddings/output) |
| `N_BATCH` | 256 | Prompt-processing (prefill) batch size. Larger = faster prefill but more VRAM; decode is always 1 token at a time |
| `CONTEXT_WINDOW` | 8192 | Context length in tokens |
| `GGML_CUDA_DISABLE_GRAPHS` | 1 | Keeps llama.cpp CUDA graph capture/replay off (see below) |

## Troubleshooting: generation hangs with `CUDA Graph id N reused`

**Symptom:** the app starts generating ("Generating response…"), the log spams
`CUDA Graph id N reused` (sometimes with `CUDA graph warmup reset` /
`CUDA graph warmup complete`), then stalls indefinitely — Gradio shows a spinner
forever.

**Cause:** on consumer NVIDIA GPUs (e.g. RTX 3060/Ampere), CUDA graph
capture/replay — especially with partial CPU/GPU offload, which fragments the
graph (`splits = 145` in `sched_reserve` logs) — can stall mid-generation.

**Fix (already applied in this repo):**

1. `GGML_CUDA_DISABLE_GRAPHS=1` — set in the `Dockerfile` `ENV` and in
   `.env.prod`. Runtime-only, no wheel rebuild needed; the shipped
   `libggml-cuda.so` reads it (symbol `disable_cuda_graphs_due_to_env`).
2. `N_GPU_LAYERS=32` — full offload removes the CPU↔GPU split, collapsing the
   graph/`split` count (~2) and eliminating per-token host copies.

**Verify:** after redeploying, the log must contain **no** `CUDA Graph` lines at
all — the banner still prints `USE_GRAPHS = 1`, but that is a compile-time
attribute of the wheel and is not a runtime toggle.

> Do **not** enable `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` — it was a leftover CMake
> build flag from the from-source era. At runtime it only reserves VRAM and
> contributes to the graph-related stalls above.

## Performance: flash attention

The backend enables flash attention (`model_kwargs={"flash_attn": True}` in
`src/SuAi/utils/model.py`) because llama-cpp-python ships it **off by default**.
Without it, every generated token pays a full `O(ctx)` attention pass — at
`CONTEXT_WINDOW=8192` this dominates decode latency (your log would show
`llama_context: flash_attn = disabled`).

After a redeploy the startup log must show **`flash_attn = 1`**; token timing
lines (`t/tok`, `ms/tok`) appear in the app log because the langchain wrapper
runs with `verbose=True`. If decode is still slow at longer contexts, lowering
`CONTEXT_WINDOW` to 4096 (~half the KV cache) is the next lever — this bot's
RAG answers are short, so 8192 is usually more than needed.