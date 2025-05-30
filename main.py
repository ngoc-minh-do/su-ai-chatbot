from utils import constants, env, logging

env.load_env()

import gradio as gr
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import CharacterTextSplitter
from transformers import BitsAndBytesConfig

from pages import training

logger = logging.get_logger(__name__)

rag_chain: Runnable = None


def format_docs(docs):
    pretty_print_docs(docs)

    return "\n\n".join(doc.page_content for doc in docs)


def pretty_print_docs(docs):
    logger.debug(f"Number of documents retrieved: {len(docs)}")
    for i, d in enumerate(docs):
        logger.debug(
            f"{'-' * 100}\nDocument {i + 1}:\nUrl: {d.metadata.get('source')}\n\n{d.page_content}\n"
        )


def create_rag_chain():
    logger.info("Loading chat model...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    llm = HuggingFacePipeline.from_model_id(
        model_id=constants.model_id,
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=100,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.05,
            return_full_text=False,
        ),
        device=constants.device,
        model_kwargs={"quantization_config": quantization_config},
    )

    logger.info("Creating rag chain...")

    model_kwargs = {"device": constants.device}
    embeddings = HuggingFaceEmbeddings(
        model_name=constants.embedding_model_id,
        model_kwargs=model_kwargs,
    )

    template = """
### 命令:
あなたはシェアフルシフトのAIアシスタントです。シェアフルシフトは、ユーザーが勤務シフトを作成・管理するためのプラットフォームです。あなたの役割は、ユーザーがシステムを効果的に使えるようにサポートし、質問に答え、シフトスケジュールの管理を手助けすることです。

あなたが行うべきこと:
- シフトの作成、編集、キャンセル、割り当て方法の説明
- シフトスケジュールの確認方法や通知機能の使い方の案内
- よくある問題（例：シフトの重複、スケジュールが表示されない等）のサポート
- やさしく、丁寧で、分かりやすい言葉遣いで対応すること

ユーザーの質問があいまいな場合は、必ず確認の質問をし、必要に応じてステップバイステップで案内してください。
### コンテキスト:
{context}

### ユーザーからの質問:
{question}

### アシスタントの返答:
"""
    prompt = PromptTemplate.from_template(template)

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=constants.qdrant_collection_name,
        url=constants.qdrant_url,
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(), llm=llm
    )

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.5,
    )
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=multi_query_retriever,
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

    rag_chain.get_graph().print_ascii()

    return rag_chain


def response_fn(message, history):
    global rag_chain
    if rag_chain is None:
        rag_chain = create_rag_chain()

    logger.info(f"Generating response for message: {message}")

    output = rag_chain.invoke(message)

    chunks = []
    for chunk in rag_chain.stream(message):
        chunks.append(chunk)
        yield "".join(chunks)

    return output


def main():
    logger.info("Starting the app...")
    with gr.ChatInterface(
        response_fn,
        type="messages",
        save_history=True,
        css="footer{display:none !important}",
    ) as demo:
        demo.title = "Shareful Shift AI Assistant"
        demo.description = "シェアフルシフトのAIアシスタントです。シフトの作成・管理について質問してください。"

    with demo.route(name="トレーニング", path="/training"):
        training.render()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
