from enum import Enum


class SelectedModel(Enum):
    litellm = "Model 1"
    ollama = "Model 2"
    huggingface = "Model 3"
    gguf = "Model 4"

selected_model = SelectedModel.litellm