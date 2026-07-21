import os
from enum import Enum


class SelectedModel(Enum):
    openai = "OpenAI"
    llama_cpp = "Llama.cpp"
    huggingface = "Hugging Face"
    ollama = "Ollama"


_selected = os.environ.get("SELECTED_MODEL", "openai")
selected_model = (
    SelectedModel[_selected]
    if _selected in SelectedModel._member_names_
    else SelectedModel.openai
)
model_choices = [m.value for m in SelectedModel]
