import re

import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from utils import logging, model

logger = logging.get_logger(__name__)

_question_chain: Runnable = None


def extract_question_only(output: str) -> str:
    output = re.sub(r"(回答|応答|答え).*$", "", output, flags=re.DOTALL)
    return output.replace("###", "").strip()


def create_question_chain() -> Runnable:
    global _question_chain
    if _question_chain is not None:
        logger.info("Question chain already created, returning existing instance.")
        return _question_chain

    logger.info("Creating question chain...")

    llm = model.get_llm()

    template = """
### 命令:
あなたは、与えられたコンテキストに基づいて質問のみを生成してください。回答は生成しないでください。

### コンテキスト:
{context}

### 質問:
"""
    prompt = PromptTemplate.from_template(template)

    question_chain = (
        {
            "context": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
        | (lambda output: extract_question_only(output))
    )

    question_chain.get_graph().print_ascii()

    _question_chain = question_chain

    return question_chain


def generate_qa():
    logger.info("Generating question and answers...")

    vector_store = model.get_vector_store()
    sample_docs = vector_store.similarity_search("", k=5)
    context = model.format_docs(sample_docs)

    question_chain = create_question_chain()

    question = question_chain.invoke(context)
    answer1 = "これは回答1のサンプルです。"
    answer2 = "これは回答2のサンプルです。"
    return question, answer1, answer2


def submit(question, answer):
    print(f"Submitted Question: {question}")
    print(f"Submitted Answer: {answer}")


def render():
    with gr.Blocks() as demo:
        with gr.Row():
            generateQA = gr.Button("質問と回答を生成")

        with gr.Row():
            gr.Markdown(
                """
    このボタンを押すと、質問と2つの回答が自動で生成されます。
    生成された内容が不自然な場合は、自由に編集してから使用できます。
    2つのうち、より適切だと思う回答を選んでください。
    また、「質問」や「回答1」のテキストボックスに任意の内容を入力して、保存することもできます。
    モデルをトレーニングするタイミングはゴックさんによる。
    """,
                line_breaks=True,
            )

        with gr.Row():
            with gr.Column():
                question = gr.Textbox(
                    label="質問",
                    lines=5,
                    placeholder="ここに質問を入力するか、上のボタンを押して自動生成してください",
                )

        with gr.Row():
            with gr.Column():
                answer1 = gr.Textbox(
                    label="回答1",
                    lines=10,
                    placeholder="ここに回答を入力するか、上のボタンを押して自動生成してください",
                )
                accept1 = gr.Button("この回答を選ぶ")
            with gr.Column():
                answer2 = gr.Textbox(label="回答2", lines=10, placeholder="")
                accept2 = gr.Button("この回答を選ぶ")

        generateQA.click(
            fn=generate_qa, inputs=[], outputs=[question, answer1, answer2]
        )
        accept1.click(fn=submit, inputs=[question, answer1], outputs=[])
        accept2.click(fn=submit, inputs=[question, answer2], outputs=[])

    return demo
