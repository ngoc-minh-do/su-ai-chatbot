import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable

from ..pages import training
from ..utils import logging, model, settings

logger = logging.get_logger(__name__)

_rag_chain: RunnableSerializable = None


def create_rag_chain():
    global _rag_chain
    if _rag_chain is not None:
        logger.info("RAG chain already created, returning existing instance.")
        return _rag_chain

    logger.info("Creating rag chain...")

    llm = model.get_llm()
    vector_store = model.get_vector_store()
    retriever = model.get_retriever(vector_store, llm=llm)

    template = """
### 命令:
あなたはシェアフルシフトのAIアシスタントです。シェアフルシフトは、ユーザーが勤務シフトを作成・管理するためのプラットフォームです。あなたの役割は、ユーザーがシステムを効果的に使えるようにサポートし、質問に答え、シフトスケジュールの管理を手助けすることです。

あなたが行うべきこと:
- シフトの作成、編集、キャンセル、割り当て方法の説明
- シフトスケジュールの確認方法や通知機能の使い方の案内
- よくある問題（例：シフトの重複、スケジュールが表示されない等）のサポート
- やさしく、丁寧で、分かりやすい言葉遣いで対応すること

ユーザーの質問があいまいな場合は、わからないと伝えてください。
### コンテキスト:
{context}

### ユーザーからの質問:
{question}

### アシスタントの返答:
"""
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": retriever | model.format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.debug("\n" + rag_chain.get_graph().draw_ascii())

    _rag_chain = rag_chain

    return rag_chain


def response_fn(message, history):
    rag_chain = create_rag_chain()

    logger.info(f"Generating response for message: {message}")

    chunks = []
    for chunk in rag_chain.stream(message):
        chunks.append(chunk)
        yield "".join(chunks)


def model_change(value):
    settings.selected_model = value

css = """
footer{display:none !important}
#component-15 { height: 80vh !important; }
"""

def main():
    logger.info("Starting the app...")

    with gr.Blocks(
        css=css,
    ) as demo:
        demo.title = "Shareful Shift AI Assistant"
        demo.description = "シェアフルシフトのAIアシスタントです。シフトの作成・管理について質問してください。"

        model_selector = gr.Dropdown(
            choices=[m.value for m in settings.Model],
            value=settings.selected_model,
            label="Model:",
        )
        model_selector.change(model_change, [model_selector])

        gr.ChatInterface(
            response_fn,
            type="messages",
            save_history=True,
        )

    with demo.route(name="トレーニング", path="/training"):
        training.render()

    demo.launch(server_name="0.0.0.0", server_port=7860)
