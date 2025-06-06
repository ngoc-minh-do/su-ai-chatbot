import torch

model_id = "llm-jp/llm-jp-3.1-1.8b-instruct4"
# model_id = "TheBloke/japanese-stablelm-instruct-gamma-7B-GGUF"
gguf_model_file = "japanese-stablelm-instruct-gamma-7b.Q4_K_M.gguf"
embedding_model_id = "retrieva-jp/amber-large"
qdrant_collection_name = "su-ai-chatbot"
qdrant_url = "http://REDACTED_IP:6333"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
db_connection_string = (
    "postgresql://postgres:REDACTED_DB_PASSWORD@REDACTED_IP:5432/su_ai_chatbot"
)
number_of_retrieve_documents = 3
max_tokens = 512