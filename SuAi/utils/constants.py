import torch

huggingface_model_id = "llm-jp/llm-jp-3.1-1.8b-instruct4"

gguf_model_id = "TheBloke/japanese-stablelm-instruct-gamma-7B-GGUF"
gguf_model_file = "japanese-stablelm-instruct-gamma-7b.Q4_K_M.gguf"

embedding_model_id = "retrieva-jp/amber-large"

qdrant_collection_name = "su-ai-chatbot"
qdrant_url = "http://localhost:6333"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

db_connection_string = (
    "postgresql://postgres:REDACTED@localhost:5432/su_ai_chatbot"
)

max_tokens = 1024
temperature = 0.8
stream = False

query_enhancement = False