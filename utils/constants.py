import torch

model_id = "llm-jp/llm-jp-3-1.8b-instruct3"
# model_id = "tensorblock/Llama-3-ELYZA-JP-8B-GGUF"
gguf_model_file = "Llama-3-ELYZA-JP-8B-Q4_K_M.gguf"
embedding_model_id = "retrieva-jp/amber-large"
qdrant_collection_name = "su-ai-chatbot"
qdrant_url = "http://REDACTED_IP:6333"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
db_connection_string = (
    "postgresql://postgres:REDACTED_DB_PASSWORD@REDACTED_IP:5432/su_ai_chatbot"
)
number_of_retrieve_documents = 5