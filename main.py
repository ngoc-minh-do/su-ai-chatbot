import os
import pathlib

from dotenv import load_dotenv

load_dotenv(".env.prod" if os.environ.get("prod") else ".env")
pathlib.Path(os.environ.get("TRANSFORMERS_CACHE")).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ.get("HF_HUB_CACHE")).mkdir(parents=True, exist_ok=True)

import logging

import gradio as gr
import torch
from huggingface_hub import login
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "llm-jp/llm-jp-3-1.8b-instruct3"
chat_model = None

logging.info("Starting the app...")


def response_fn(message, history):
    DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"

    global chat_model
    if chat_model is None:
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

        chat_model = ChatHuggingFace(llm=llm)

    logging.info(f"Generating response for message: {message}")

    messages = [
        SystemMessage(content=DEFAULT_SYSTEM_PROMPT),
        HumanMessage(content=message),
    ]
    output = chat_model.invoke(messages).content

    return output


demo = gr.ChatInterface(
    response_fn,
    type="messages",
    save_history=True,
    css="footer{display:none !important}",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
