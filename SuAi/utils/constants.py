import os

import torch

huggingface_model_id = "llm-jp/llm-jp-3.1-1.8b-instruct4"

gguf_model_id = "TheBloke/japanese-stablelm-instruct-gamma-7B-GGUF"
gguf_model_file = "japanese-stablelm-instruct-gamma-7b.Q4_K_M.gguf"

embedding_model_id = "retrieva-jp/amber-large"

qdrant_collection_name = os.environ.get("QDRANT_COLLECTION_NAME", "su-ai-chatbot")
qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

db_connection_string = os.environ.get(
    "DATABASE_URL", ""
)

max_tokens = int(os.environ.get("MAX_TOKENS", "1024"))
temperature = float(os.environ.get("TEMPERATURE", "0.8"))
context_window = int(os.environ.get("CONTEXT_WINDOW", "8192"))
n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", "20"))
stream = os.environ.get("STREAM", "true").lower() == "true"

query_enhancement = os.environ.get("QUERY_ENHANCEMENT", "false").lower() == "true"