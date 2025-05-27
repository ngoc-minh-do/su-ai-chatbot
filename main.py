import os
import pathlib

from dotenv import load_dotenv

load_dotenv(".env.prod" if os.environ.get("prod") else ".env")
pathlib.Path(os.environ.get("TRANSFORMERS_CACHE")).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ.get("HF_HUB_CACHE")).mkdir(parents=True, exist_ok=True)

import logging

import gradio as gr
import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from transformers import BitsAndBytesConfig

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "llm-jp/llm-jp-3-1.8b-instruct3"
embedding_model_id = "retrieva-jp/amber-large"
qdrant_collection_name = "su-ai-chatbot"
llm = None
vector_store = None
prompt = None

logging.info("Starting the app...")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def response_fn(message, history):
    global llm, vector_store, prompt
    if llm is None:
        logging.info("Loading chat model...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )

        llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs=dict(
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
                return_full_text=False,
            ),
            device=device,
            model_kwargs={"quantization_config": quantization_config},
        )

        model_kwargs = {"device": device}
        embedding = HuggingFaceEmbeddings(
            model_name=embedding_model_id,
            model_kwargs=model_kwargs,
        )

        template = """User:
あなたはSu AI、質問応答タスクのアシスタントです。
以下のコンテキストに基づいて質問に答えます。答えがわからない場合は、わからないと言ってください。最大 3 つの文を使用し、回答は簡潔にしてください。
Question : {question}
Context : {context}
Answer :
"""
        prompt = PromptTemplate.from_template(template)

        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embedding,
            collection_name=qdrant_collection_name,
            url="http://REDACTED_IP:6333",
        )

    logging.info(f"Generating response for message: {message}")

    qa_chain = (
        {
            "context": vector_store.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    output = qa_chain.invoke(message)

    return output


demo = gr.ChatInterface(
    response_fn,
    type="messages",
    save_history=True,
    css="footer{display:none !important}",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
