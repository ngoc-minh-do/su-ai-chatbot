import logging
import os
import pathlib
import re

import torch
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv(".env.prod" if os.environ.get("prod") else ".env")
pathlib.Path(os.environ.get("TRANSFORMERS_CACHE")
             ).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ.get("HF_HUB_CACHE")).mkdir(parents=True, exist_ok=True)


LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model_id = "retrieva-jp/amber-large"
qdrant_collection_name = "su-ai-chatbot"
vector_store = None

logging.info("Initializing the embedding model...")
model_kwargs = {"device": device}
embedding = HuggingFaceEmbeddings(
    model_name=embedding_model_id,
    model_kwargs=model_kwargs,
)

logging.info("Creating the Qdrant vector store...")
vector_store = QdrantVectorStore.from_texts(
    texts=[
        "Su AI Chatbot is a conversational AI system. It is designed to assist users with various tasks and provide information. It can answer questions, provide recommendations, and engage in general conversation.",
        "Su AI Chatbot is built using advanced natural language processing techniques. It can understand and generate human-like responses, making interactions more natural and intuitive.",
        "Su AI Chatbot was developed by Do Minh Ngoc, a software engineer with expertise in AI and machine learning. The chatbot is part of a larger project to create intelligent systems that can assist users in their daily lives.",
    ],
    embedding=embedding,
    collection_name=qdrant_collection_name,
    url="http://REDACTED_IP:6333",
    force_recreate=True,
)


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"[\n\s]{2,}", "\n\n", soup.text).strip()


logging.info("Loading web pages...")
web_paths = [
    # {"url": "https://www.sync-up.jp", "max_depth": 2},
    {"url": "https://knowledge.sync-up.jp/knowledge", "max_depth": 10},
]

for web_path in web_paths:
    logging.info("Loading web page: %s", web_path.get("url"))

    loader = RecursiveUrlLoader(
        url=web_path.get("url"),
        max_depth=web_path.get("max_depth"),
        extractor=bs4_extractor
    )
    docs = loader.load()
    logging.info("Loaded %d documents from web pages.", len(docs))

    for doc in docs:
        print(doc.metadata.get("source"))

    for doc in docs:
        print(doc.page_content[:1000])

    # logging.info("Adding documents to the vector store...")
    # vector_store.add_documents(docs)
