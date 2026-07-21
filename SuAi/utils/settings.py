from enum import Enum


class SelectedModel(Enum):
    openai = "OpenAI"
    llama_cpp = "Llama.cpp"
    huggingface = "Hugging Face"
    ollama = "Ollama"


selected_model = SelectedModel.openai
model_choices = [m.value for m in SelectedModel]
