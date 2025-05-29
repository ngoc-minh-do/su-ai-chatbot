import logging
import os
import pathlib
import re

from dotenv import load_dotenv

import constants

load_dotenv(".env.prod" if os.environ.get("prod") else ".env")
pathlib.Path(os.environ.get("TRANSFORMERS_CACHE")).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ.get("HF_HUB_CACHE")).mkdir(parents=True, exist_ok=True)

from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)

logging.info("Initializing the embedding model...")
model_kwargs = {"device": constants.device}
embedding = HuggingFaceEmbeddings(
    model_name=constants.embedding_model_id,
    model_kwargs=model_kwargs,
)

logging.info("Creating the Qdrant vector store...")
vector_store = QdrantVectorStore.from_texts(
    texts=[
        "Su AIチャットボットは、あなたの期待を超える、誠実で優秀なAIアシスタントです。",
        "シェアフルシフトサービスを瞬時に理解。\nシェアフルシフトサービスについて知りたいですか？ Su AIチャットボットを使えば、まるで親しい友人と話すように、自然な会話でそのすべてを今すぐ、そして簡単に学ぶことができます。もう、複雑な説明書を読んだり、どこを探せばいいか迷ったりする必要はありません。知りたいことがあれば、Su AIチャットボットに尋ねるだけ。",
        "あなたの毎日をサポートする、万能な会話パートナー。\nSu AIチャットボットは単なる情報提供ツールではありません。様々な質問に答えたり、パーソナルなおすすめを提案したり、時には気さくな会話であなたを楽しませることもできます。まるであなたのそばに、いつでも頼れるエキスパートがいるようなものです。",
        "圧倒的な自然さと直感的な操作性。\n最先端の自然言語処理技術を駆使して開発されたSu AIチャットボットは、あなたの言葉を正確に理解し、人間が話すような自然な応答を生成します。そのため、まるで本当に人と話しているかのような、スムーズで直感的な対話が可能です。複雑な操作は一切ありません。",
        "未来を創るAI技術の結晶。\nSu AIチャットボットは、AIと機械学習の分野で深い専門知識を持つソフトウェアエンジニア、ドゥ・ミン・ゴックによって生み出されました。これは、あなたの日常生活をより豊かに、より便利にするための、インテリジェントなシステムを創造するという大きなビジョンの一部です。",
        "今すぐSu AIチャットボットを体験して、その驚くべき能力を実感してください。",
    ],
    embedding=embedding,
    collection_name=constants.qdrant_collection_name,
    url=constants.qdrant_url,
    force_recreate=True,
)

seen_paras = set()


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    html_string = re.sub(r"(\n\s*){2,}", "\n\n", soup.text).strip()

    html_string_paras = []
    for para in html_string.split("\n\n"):
        if para not in seen_paras:
            seen_paras.add(para)
            html_string_paras.append(para)

    return "\n\n".join(html_string_paras)


logging.info("Loading web pages...")

with open("source.txt") as f:
    lines = f.read().splitlines()

web_paths = map(
    lambda x: {"url": x},
    lines,
)

docs = []
for web_path in web_paths:
    logging.info("Loading web page: %s", web_path.get("url"))

    loader = RecursiveUrlLoader(
        url=web_path.get("url"),
        max_depth=web_path.get("max_depth", 1),
        extractor=bs4_extractor,
        exclude_dirs=web_path.get("exclude_dirs"),
    )

    text_splitter = RecursiveCharacterTextSplitter()

    docs.extend(loader.load_and_split(text_splitter=text_splitter))

redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
docs = redundant_filter.transform_documents(
    documents=docs,
)

logging.info("Loaded %d documents from web pages.", len(docs))

# for doc_idx, doc in enumerate(docs):
#     print(doc.metadata.get("source"))
#     with open(f"v4-{doc_idx}.txt", "a") as f:
#         f.write(doc.page_content)

logging.info("Adding documents to the vector store...")
vector_store.add_documents(docs)
