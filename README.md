# Su AI Chatbot

AI-powered Japanese-language assistant for the [Sharefull Shift](https://sharefullshift.com/) service, built with RAG (Retrieval-Augmented Generation), Gradio UI, and multi-model support.

[![CI](https://github.com/ngoc-minh-do/su-ai-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/ngoc-minh-do/su-ai-chatbot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](.python-version)

**Status:** production — actively maintained, deployed via Docker with NVIDIA GPU support.

## Features

- **RAG chatbot** for the Sharefull Shift platform — answers questions about shift scheduling, creation, editing, and management
- **Multi-model support** — switch between OpenAI/LiteLLM, Ollama, HuggingFace, and Llama.cpp backends at runtime
- **Training data generation** — auto-generate QA pairs for fine-tuning, with human preference selection
- **Streaming responses** — real-time token streaming with `<think>` tag filtering for reasoning models
- **Dockerized** — GPU-accelerated deployment with NVIDIA CUDA support
- **Persistent chat history** via Gradio

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────┐
│  Gradio UI  │────▶│  RAG Chain   │────▶│   LLM    │
│  (home.py)  │     │  (LangChain) │     │ (multi)  │
└─────────────┘     └──────┬───────┘     └──────────┘
                           │
                    ┌──────▼───────┐     ┌──────────┐
                    │   Retriever  │────▶│  Qdrant  │
                    │  + Pipeline  │     │ Vector DB│
                    └──────────────┘     └──────────┘
                           │
                    ┌──────▼───────┐
                    │  PostgreSQL  │
                    │ (training)   │
                    └──────────────┘
```

## Prerequisites

- Python 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL (for training data)
- Qdrant (vector database)
- NVIDIA GPU + CUDA (optional, for HuggingFace/Llama.cpp backends)

## Quick Start

```bash
# Clone the repo
git clone git@github.com:ngoc-minh-do/su-ai-chatbot.git
cd su-ai-chatbot

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your API keys and service URLs

# Run
python main.py
```

The Gradio interface opens at `http://localhost:7860`.

## LLM Backends

| Backend | Env Variables | GPU |
|---|---|---|
| OpenAI / LiteLLM | `OPENAI_API_KEY`, `OPENAI_API_URL`, `OPENAI_MODEL` | No |
| Ollama | `OLLAMA_URL`, `OLLAMA_MODEL` | Optional |
| HuggingFace | `HF_TOKEN` (for gated models) | Yes |
| Llama.cpp | `HF_TOKEN` (for model download), `N_GPU_LAYERS`, `N_BATCH` | Yes |

Set `SELECTED_MODEL=openai` (or `ollama`, `huggingface`, `llama_cpp`) in your `.env` to choose the default backend. You can also switch models at runtime via the dropdown in the UI.

## Environment Variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Description |
|---|---|
| `SELECTED_MODEL` | Default LLM backend (`openai`, `ollama`, `huggingface`, `llama_cpp`) |
| `OPENAI_API_KEY` | OpenAI / LiteLLM API key |
| `OPENAI_API_URL` | LiteLLM proxy URL |
| `OPENAI_MODEL` | Model name |
| `OLLAMA_URL` | Ollama server URL |
| `OLLAMA_MODEL` | Ollama model name |
| `QDRANT_URL` | Qdrant server URL |
| `DATABASE_URL` | PostgreSQL connection string |
| `HF_TOKEN` | HuggingFace token for gated models |
| `MAX_TOKENS` | Max output tokens (default: 1024) |
| `TEMPERATURE` | Generation temperature (default: 0.8) |
| `CONTEXT_WINDOW` | Context length in tokens (default: 8192) |
| `N_GPU_LAYERS` | Llama.cpp layers offloaded to GPU; `-1` (default) = all layers incl. embeddings/output. `32` = all 32 blocks but leaves block 0 on CPU, which slows decode |
| `N_BATCH` | Llama.cpp prompt-processing batch size (default: 256) |
| `GGML_CUDA_DISABLE_GRAPHS` | Keep llama.cpp CUDA graphs off to avoid stalls on consumer GPUs |
| `STREAM` | Enable streaming (default: true) |
| `QUERY_ENHANCEMENT` | Enable MultiQueryRetriever (default: false) |

## Docker

```bash
# Build and run
docker compose up -d

# Requires NVIDIA Container Toolkit for GPU support
```

The container exposes port `7860`. A healthcheck verifies the service is running.

CUDA graphs are disabled in the image (`GGML_CUDA_DISABLE_GRAPHS=1`) and the
Llama.cpp backend defaults to full GPU offload (`N_GPU_LAYERS=32`) — graph
capture/replay can stall generation on consumer GPUs. See
[docs/build-llama-cpp-python.md](docs/build-llama-cpp-python.md) for details.

## Loading Documents

Place URLs in `scripts/source.txt` (one per line), then run:

```bash
uv run python -m scripts.doc_loader
# or
make load-docs
```

`doc_loader.py` loads its configuration from `.env` (or `.env.prod` when `prod=true`) and resolves `source.txt` relative to the script itself, so it can be run from anywhere in the repo. Documents are embedded and stored in Qdrant for retrieval.

> **Warning:** the loader runs with `force_recreate=True` — it **deletes and rebuilds the target Qdrant collection from scratch**. Before running, make sure `QDRANT_URL` / `QDRANT_COLLECTION_NAME` point at the collection you intend to replace (defaults: `http://localhost:6333` / `su-ai-chatbot`). The embedding model is downloaded from Hugging Face on first run.

## Development

```bash
# Install dev dependencies and pre-commit hooks
make install

# Run all checks
make check

# Individual commands
make lint        # ruff check
make format      # ruff format
make typecheck   # ty
make test        # pytest
make fix         # auto-fix lint/format issues
make load-docs   # (re)build the Qdrant knowledge base from scripts/source.txt
```

Pre-commit hooks run ruff (lint + format), ty (type checking), and uv-lock (keeps `uv.lock` in sync) on every commit.

## Project Structure

```
su-ai-chatbot/
├── src/
│   └── SuAi/
│       ├── pages/
│       │   ├── home.py          # Main chatbot UI + RAG chain
│       │   └── training.py      # QA pair generation UI
│       ├── db/
│       │   ├── engine.py        # PostgreSQL connection
│       │   ├── models.py        # SQLAlchemy models
│       │   ├── operations.py    # CRUD operations
│       │   └── utils.py         # DB management helpers
│       └── utils/
│           ├── constants.py     # Hardcoded model IDs + env fallbacks
│           ├── env.py           # Environment loader
│           ├── logging.py       # Logger setup
│           ├── model.py         # LLM/embedding/retriever factory
│           └── settings.py      # SelectedModel enum
├── scripts/
│   └── doc_loader.py        # Web scraper → Qdrant ingestion
├── tests/
│   └── test_utils.py        # Unit tests
├── docs/
│   └── build-llama-cpp-python.md  # Llama.cpp CUDA wheel, knobs & troubleshooting
├── main.py                  # Entry point
├── Dockerfile
├── docker-compose.yaml
└── Makefile
```

## Known Limitations

- **Answers are limited to the loaded documents** — the RAG chain only knows what's in the Qdrant collection. Keep `scripts/source.txt` current and rebuild the collection (`make load-docs`) whenever the source docs change.
- **Potential hallucinations** — as with any LLM, responses can be inaccurate or invented. The bot is a support assistant, not a substitute for official documentation or human judgment.
- **Japanese-first** — prompts, training data, and the UI are in Japanese; responses are most reliable in Japanese.
- **Local models are resource-heavy** — the Llama.cpp and HuggingFace backends need a GPU with enough VRAM and download several GB on first load.
- **Destructive document loader** — `make load-docs` force-recreates the Qdrant collection, deleting existing vectors.

## License

MIT — see [LICENSE](LICENSE) for details.
