import os
import pathlib
from dotenv import load_dotenv

load_dotenv(".env.prod" if os.environ.get("prod") else ".env")
pathlib.Path(os.environ.get("TRANSFORMERS_CACHE")).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ.get("HF_HUB_CACHE")).mkdir(parents=True, exist_ok=True)

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = "llm-jp/llm-jp-3-1.8b-instruct3"
model = None
tokenizer = None


def response_fn(message, history):
    DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"
    
    global model, tokenizer
    if model is None:
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)

    chat = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    tokenized_input = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            tokenized_input,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.05,
        )[0]

    output = tokenizer.decode(
        output[tokenized_input.size(1) :], skip_special_tokens=True
    )
    return output


demo = gr.ChatInterface(
    response_fn,
    type="messages",
    save_history=True,
    css="footer{display:none !important}",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
