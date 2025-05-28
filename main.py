import os
import pathlib

from dotenv import load_dotenv

load_dotenv(".env.prod" if os.environ.get("prod") else ".env")
pathlib.Path(os.environ.get("TRANSFORMERS_CACHE")).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ.get("HF_HUB_CACHE")).mkdir(parents=True, exist_ok=True)

import logging

import gradio as gr
import torch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import CharacterTextSplitter
from transformers import BitsAndBytesConfig

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "llm-jp/llm-jp-3-1.8b-instruct3"
embedding_model_id = "retrieva-jp/amber-large"
qdrant_collection_name = "su-ai-chatbot"
rag_chain = None
logging.info("Starting the app...")


def format_docs(docs):
    pretty_print_docs(docs)

    return "\n\n".join(doc.page_content for doc in docs)


def pretty_print_docs(docs):
    for i, d in enumerate(docs):
        print(
            f"{'-' * 100}\nDocument {i + 1}:\nUrl: {d.metadata.get('source')}\n\n{d.page_content}\n"
        )


def create_rag_chain():
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

    logging.info("Creating rag chain...")

    model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs=model_kwargs,
    )

    template = """System:
あなたはSu AI、質問応答タスクのアシスタントです。
以下のコンテキストに基づいて質問に答えます。答えがわからない場合は、わからないと言ってください。最大 3 つの文を使用し、回答は簡潔にしてください。
Question : {question}
Context : {context}
Answer :
"""
    prompt = PromptTemplate.from_template(template)

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=qdrant_collection_name,
        url="http://REDACTED_IP:6333",
    )

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.2,
    )
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=vector_store.as_retriever(),
    )

    rag_chain = (
        {
            "context": compression_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(rag_chain.get_graph())

    return rag_chain


def response_fn(message, history):
    global rag_chain
    if rag_chain is None:
        rag_chain = create_rag_chain()

    logging.info(f"Generating response for message: {message}")

    output = rag_chain.invoke(message)

    return output


demo = gr.ChatInterface(
    response_fn,
    type="messages",
    save_history=True,
    css="footer{display:none !important}",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
