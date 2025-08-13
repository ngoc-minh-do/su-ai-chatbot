from enum import Enum

from ..utils import constants


class SelectedModel(Enum):
    openai = "OpenAI"
    ollama = "Ollama"
    huggingface = "Hugging Face"
    llama_cpp = "Llama.cpp"

selected_model = SelectedModel.openai
model_choices = (
    [m.value for m in SelectedModel]
)
