import pytest

from SuAi.db.engine import _validate_identifier
from SuAi.pages.training import extract_question_only
from SuAi.utils.settings import SelectedModel


# --- extract_question_only ---


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("これは質問です。", "これは質問です。"),
        ("質問を生成してください。回答は不要です。", "質問を生成してください。"),
        ("これは回答です。\n応答を確認してください。", "これは"),
        ("### 質問: シフトの作成方法は？", "質問: シフトの作成方法は？"),
        ("シフトについて教えてください。答えは簡単です。", "シフトについて教えてください。"),
        ("", ""),
    ],
)
def test_extract_question_only(input_text, expected):
    assert extract_question_only(input_text) == expected


# --- _validate_identifier ---


@pytest.mark.parametrize(
    "identifier,should_pass",
    [
        ("my_db", True),
        ("_private", True),
        ("MY_DB", True),
        ("db_123", True),
        ("my-db", False),
        ("123db", False),
        ("db;drop", False),
        ("db'x", False),
        ("", False),
        ("db name", False),
        ("--comment", False),
        ("db/*x", False),
        ("select", True),
    ],
)
def test_validate_identifier(identifier, should_pass):
    if should_pass:
        assert _validate_identifier(identifier) == identifier
    else:
        with pytest.raises(ValueError):
            _validate_identifier(identifier)


# --- SelectedModel ---


def test_selected_model_values():
    assert SelectedModel.openai.value == "OpenAI"
    assert SelectedModel.ollama.value == "Ollama"
    assert SelectedModel.huggingface.value == "Hugging Face"
    assert SelectedModel.llama_cpp.value == "Llama.cpp"


def test_selected_model_from_value():
    assert SelectedModel("OpenAI") == SelectedModel.openai
    assert SelectedModel("Ollama") == SelectedModel.ollama
    assert SelectedModel("Hugging Face") == SelectedModel.huggingface
    assert SelectedModel("Llama.cpp") == SelectedModel.llama_cpp


def test_selected_model_invalid():
    with pytest.raises(ValueError):
        SelectedModel("invalid")
