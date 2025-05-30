import gradio as gr

from utils import logging

logger = logging.get_logger(__name__)


def generate_qa():
    question = "これはサンプルの質問です。"
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
