# Su AI Chatbot

AI-powered Japanese-language assistant for the [Sharefull Shift](https://sharefullshift.com/) service, built with RAG (Retrieval-Augmented Generation), Gradio UI, and multi-model support.

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
| Llama.cpp | `HF_TOKEN` (for model download) | Yes |

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
| `STREAM` | Enable streaming (default: true) |
| `QUERY_ENHANCEMENT` | Enable MultiQueryRetriever (default: false) |

## Docker

```bash
# Build and run
docker compose up -d

# Requires NVIDIA Container Toolkit for GPU support
```

The container exposes port `7860`. A healthcheck verifies the service is running.

## Loading Documents

Place URLs in `scripts/source.txt` (one per line), then run:

```bash
python scripts/doc_loader.py
```

Documents are embedded and stored in Qdrant for retrieval.

## Development

```bash
# Install dev dependencies and pre-commit hooks
make install

# Run all checks
make check

# Individual commands
make lint        # ruff check
make format      # ruff format
make typecheck   # pyright
make test        # pytest
make fix         # auto-fix lint/format issues
```

Pre-commit hooks run ruff (lint + format) and pyright on every commit.

## Project Structure

```
su-ai-chatbot/
├── SuAi/
│   ├── pages/
│   │   ├── home.py          # Main chatbot UI + RAG chain
│   │   └── training.py      # QA pair generation UI
│   ├── db/
│   │   ├── engine.py        # PostgreSQL connection
│   │   ├── models.py        # SQLAlchemy models
│   │   ├── operations.py    # CRUD operations
│   │   └── utils.py         # DB management helpers
│   └── utils/
│       ├── constants.py     # Hardcoded model IDs + env fallbacks
│       ├── env.py           # Environment loader
│       ├── logging.py       # Logger setup
│       ├── model.py         # LLM/embedding/retriever factory
│       └── settings.py      # SelectedModel enum
├── scripts/
│   └── doc_loader.py        # Web scraper → Qdrant ingestion
├── tests/
│   └── test_utils.py        # Unit tests
├── docs/
│   └── build-llama-cpp-python.md  # CUDA wheel build guide
├── main.py                  # Entry point
├── Dockerfile
├── docker-compose.yaml
└── Makefile
```

## License

MIT — see [LICENSE](LICENSE) for details.
