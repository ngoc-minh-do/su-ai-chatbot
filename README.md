# su-ai-chatbot

AI chatbot for Sharefull Shift service — RAG-powered Japanese-language assistant.

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync
```

## Environment

Copy `.env.example` to `.env` and fill in the required values:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI/LiteLLM API key |
| `OPENAI_API_URL` | LiteLLM proxy URL |
| `OPENAI_MODEL` | Model name |
| `QDRANT_URL` | Qdrant server URL |
| `DATABASE_URL` | PostgreSQL connection string |
| `HF_TOKEN` | HuggingFace token for gated models |

## Running

```bash
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1  # if using llama.cpp with CUDA
python main.py
```

The Gradio interface opens at `http://localhost:7860`.

## Testing

```bash
uv run pytest tests/ -v
```
