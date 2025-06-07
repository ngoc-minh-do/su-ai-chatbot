from enum import Enum

from ..utils import constants


class SelectedModel(Enum):
    litellm = "Model 1"
    ollama = "Model 2 (Local)"
    huggingface = "Model 3 (Local)"
    gguf = "Model 4 (Local)"

selected_model = SelectedModel.litellm
model_choices = (
    [m.value for m in SelectedModel]
    if constants.is_dev
    else [SelectedModel.litellm.value, SelectedModel.ollama.value]
)
