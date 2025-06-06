import re

import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSerializable,
)

from db.operations import create_training_qa_data
from utils import logging, model

logger = logging.get_logger(__name__)

_question_chain: RunnableSerializable = None
_answer_chain: RunnableSerializable = None


def extract_question_only(output: str) -> str:
    output = re.sub(r"(回答|応答|答え|Answer).*$", "", output, flags=re.DOTALL)
    return output.replace("###", "").strip()


def create_question_chain() -> RunnableSerializable:
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

    logger.debug("\n" + question_chain.get_graph().draw_ascii())

    _question_chain = question_chain

    return question_chain


def create_answer_chain() -> RunnableSerializable:
    global _answer_chain
    if _answer_chain is not None:
        logger.info("Answer chain already created, returning existing instance.")
        return _answer_chain

    logger.info("Creating answer chain...")

    llm = model.get_llm()

    template = """
### 命令:
あなたは丁寧で正確なアシスタントです。以下のコンテキストに基づいて、質問に対して的確かつ簡潔に答えてください。  
必要に応じて、コンテキストの情報を活用してください。わからない場合は、無理に作り話をしないでください。

### コンテキスト:
{context}

### 質問:
{question}

### 回答:
"""

    prompt = PromptTemplate.from_template(template)

    answer_chain = prompt | llm | StrOutputParser()

    logger.debug("\n" + answer_chain.get_graph().draw_ascii())

    _answer_chain = answer_chain

    return answer_chain


def generate_qa():
    logger.info("Generating question and answers...")

    sample_docs = model.get_random_docs()

    context = model.format_docs(sample_docs)

    question_chain = create_question_chain()

    question = question_chain.invoke(context)

    yield (
        gr.update(value=question, interactive=False),
        gr.update(value="", interactive=False),
        gr.update(value="", interactive=False),
    )

    answer_chain = create_answer_chain()

    answer1 = answer_chain.invoke({"context": context, "question": question})

    yield (
        gr.update(),
        gr.update(value=answer1),
        gr.update(),
    )

    answer2 = answer_chain.invoke({"context": context, "question": question})

    yield (
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(value=answer2, interactive=True),
    )


def submit1(question, answer1, answer2):
    return submit(question, answer1, answer2, 1)


def submit2(question, answer1, answer2):
    return submit(question, answer1, answer2, 2)


def submit(question: str, answer1: str, answer2: str, selected_answer: int):
    logger.info(
        f"Submitting question: {question}, answer1: {answer1}, answer2: {answer2}, selected_answer: {selected_answer}"
    )

    if question.strip() and answer1.strip():
        create_training_qa_data(question, answer1, 1 if selected_answer == 1 else 0)

    if question.strip() and answer2.strip():
        create_training_qa_data(question, answer2, 1 if selected_answer == 2 else 0)

    return gr.update(value=""), gr.update(value=""), gr.update(value="")


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
        accept1.click(
            fn=submit1,
            inputs=[question, answer1, answer2],
            outputs=[question, answer1, answer2],
        )
        accept2.click(
            fn=submit2,
            inputs=[question, answer1, answer2],
            outputs=[question, answer1, answer2],
        )

    return demo
