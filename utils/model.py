from utils import constants, env, logging

env.load_env()


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.language_models.llms import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import CharacterTextSplitter
from transformers import BitsAndBytesConfig

logger = logging.get_logger(__name__)

_llm: BaseLLM = None
_vector_store: VectorStore = None
_retriever: ContextualCompressionRetriever = None


def get_llm() -> BaseLLM:
    global _llm
    if _llm is not None:
        logger.info("LLM already loaded, returning existing instance.")
        return _llm

    logger.info("Loading llm...")
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

    _llm = llm

    return llm


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is not None:
        logger.info("Vector store already loaded, returning existing instance.")
        return _vector_store

    logger.info("Loading embeddings...")
    model_kwargs = {"device": constants.device}
    embeddings = HuggingFaceEmbeddings(
        model_name=constants.embedding_model_id,
        model_kwargs=model_kwargs,
    )

    logger.info("Creating vector store...")
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=constants.qdrant_collection_name,
        url=constants.qdrant_url,
    )
    _vector_store = vector_store

    return vector_store


def get_retriever(vector_store: VectorStore, llm: BaseLLM) -> BaseRetriever:
    global _retriever
    if _retriever is not None:
        logger.info("Retriever already loaded, returning existing instance.")
        return _retriever

    logger.info("Creating retriever...")
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(), llm=llm
    )

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=vector_store.embeddings)
    relevant_filter = EmbeddingsFilter(
        embeddings=vector_store.embeddings,
        similarity_threshold=0.5,
    )
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=multi_query_retriever,
    )

    _retriever = compression_retriever

    return compression_retriever


def format_docs(docs):
    pretty_print_docs(docs)

    return "\n\n".join(doc.page_content for doc in docs)


def pretty_print_docs(docs):
    logger.debug(f"Number of documents retrieved: {len(docs)}")
    for i, d in enumerate(docs):
        logger.debug(
            f"{'-' * 100}\nDocument {i + 1}:\nUrl: {d.metadata.get('source')}\n\n{d.page_content}\n"
        )
