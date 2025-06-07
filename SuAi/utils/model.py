import os

from huggingface_hub import hf_hub_download
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.documents import Document
from langchain_core.language_models.llms import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import CharacterTextSplitter
from qdrant_client import models
from transformers import BitsAndBytesConfig

from ..utils import constants, logging, settings
from ..utils.settings import SelectedModel

logger = logging.get_logger(__name__)

_llm: BaseLLM = {}
_vector_store: VectorStore = None
_retriever: ContextualCompressionRetriever = None


def get_llm() -> BaseLLM:
    global _llm
    model_name = settings.selected_model.name

    if _llm.get(model_name) is not None:
        logger.info(f"LLM {model_name} already loaded, returning existing instance.")
        return _llm.get(model_name)

    logger.info(f"Loading LLM {model_name}...")

    if settings.selected_model == SelectedModel.ollama:
        _llm[model_name] = _load_ollama_llm()
    elif settings.selected_model == SelectedModel.gguf:
        _llm[model_name] = _load_gguf_llm()
    elif settings.selected_model == SelectedModel.huggingface:
        _llm[model_name] = _load_huggingface_llm()
    elif settings.selected_model == SelectedModel.litellm:
        _llm[model_name] = _load_litellm_llm()
    else:
        pass

    return _llm.get(model_name)


def _load_ollama_llm() -> BaseLLM:
    llm = ChatOllama(
        base_url=os.environ.get("OLLAMA_URL"),
        model="deepseek-r1:8b",
        temperature=constants.temperature,
        num_predict=constants.max_tokens,
        num_gpu=10000,
    )

    return llm


def _load_litellm_llm() -> BaseLLM:
    llm = ChatOpenAI(
        base_url=os.environ.get("LITELLM_API_URL"),
        api_key=os.environ.get("LITELLM_API_KEY"),
        model="mistral/devstral-small-2505",
        temperature=constants.temperature,
        max_tokens=constants.max_tokens,
    )

    return llm


def _load_huggingface_llm() -> BaseLLM:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    llm = HuggingFacePipeline.from_model_id(
        model_id=constants.huggingface_model_id,
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=constants.max_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=constants.temperature,
            repetition_penalty=1.05,
            return_full_text=False,
        ),
        device=constants.device,
        model_kwargs={"quantization_config": quantization_config},
    )

    return llm


def _load_gguf_llm() -> BaseLLM:
    model_path = hf_hub_download(
        repo_id=constants.gguf_model_id, filename=constants.gguf_model_file
    )

    logger.info(f"Loading GGUF model from {model_path}")
    n_gpu_layers = -1
    n_batch = 512

    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=8192,
        max_tokens=constants.max_tokens,
        temperature=constants.temperature,
    )

    return llm


def get_vector_store() -> QdrantVectorStore:
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


def get_random_docs() -> list[Document]:
    logger.info("Retrieving random documents from vector store...")
    vector_store = get_vector_store()

    response = vector_store.client.query_points(
        collection_name=vector_store.collection_name,
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        limit=1,
    )

    documents = [
        vector_store._document_from_point(
            result,
            vector_store.collection_name,
            vector_store.content_payload_key,
            vector_store.metadata_payload_key,
        )
        for result in response.points
    ]

    return documents


def get_retriever(vector_store: VectorStore, llm: BaseLLM) -> BaseRetriever:
    global _retriever
    if _retriever is not None:
        logger.info("Retriever already loaded, returning existing instance.")
        return _retriever

    logger.info("Creating retriever...")
    base_retriever = (
        MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(), llm=llm)
        if constants.query_enhancement
        else vector_store.as_retriever()
    )
    logger.info(f"Using {base_retriever.__class__.__name__} as base retriever")

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=vector_store.embeddings)
    relevant_filter = EmbeddingsFilter(
        embeddings=vector_store.embeddings,
        similarity_threshold=0.5,
        k=3,
    )
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever,
    )

    _retriever = compression_retriever

    return compression_retriever


def format_docs(docs: list[Document]) -> str:
    pretty_print_docs(docs)

    return "\n\n".join(doc.page_content for doc in docs)


def pretty_print_docs(docs: list[Document]) -> None:
    logger.debug(f"Number of documents retrieved: {len(docs)}")
    for i, d in enumerate(docs):
        logger.debug(
            f"{'-' * 100}\nDocument {i + 1}:\nUrl: {d.metadata.get('source')}\n\n{d.page_content}\n"
        )
