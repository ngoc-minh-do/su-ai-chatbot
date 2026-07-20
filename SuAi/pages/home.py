import re

import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable

from ..pages import training
from ..utils import constants, logging, model, settings

logger = logging.get_logger(__name__)


def create_rag_chain() -> RunnableSerializable:
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

    return rag_chain

think_re = re.compile(r"<think>.*?</think>", re.DOTALL)

def response_fn(message, history, selected_model):
    if not message or not message.strip():
        yield "質問を入力してください。"
        return

    if len(message) > 2000:
        yield "質問が長すぎます。2000文字以内で入力してください。"
        return

    settings.selected_model = settings.SelectedModel(selected_model)

    yield "Loading the models..."
    rag_chain = create_rag_chain()

    yield "Generating the response..."
    logger.info(f"Generating response for message: {message}")

    if constants.stream:
        buffer = ""
        cleaned_buffer = ""
        stream = rag_chain.stream(message)

        while True:
            try:
                chunk = next(stream)

                if not chunk:
                    continue

                buffer += chunk
                cleaned_buffer = think_re.sub("", buffer).strip()

                if "<think>" not in cleaned_buffer:
                    yield cleaned_buffer
            except StopIteration:
                print(f"Thinking: {'\n'.join(think_re.findall(buffer)).strip()}")
                yield cleaned_buffer
                break
    else:
        yield rag_chain.invoke(message)


def model_change(value):
    settings.selected_model = settings.SelectedModel(value)


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
            choices=settings.model_choices,
            value=settings.selected_model.value,
            label="Model:",
        )
        model_selector.change(model_change, [model_selector])

        gr.ChatInterface(
            response_fn,
            type="messages",
            save_history=True,
            additional_inputs=[model_selector],
        )

    with demo.route(name="トレーニング", path="/training"):
        training.render()

    demo.launch(server_name="0.0.0.0", server_port=7860)
