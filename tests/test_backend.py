from unittest.mock import MagicMock, patch

import pytest

from backend import generate_text, get_pipeline


@patch("backend.pipeline")
def test_get_pipeline(mock_pipeline):
    """Test get_pipeline calls transformers.pipeline correctly."""
    mock_pipe_instance = MagicMock()
    mock_pipeline.return_value = mock_pipe_instance

    model_name = "test-model"
    device_id = 0

    pipe = get_pipeline(model_name, device_id)

    mock_pipeline.assert_called_once_with(
        "text-generation",
        model=model_name,
        device=device_id,
        torch_dtype=pytest.importorskip("torch").bfloat16,
        trust_remote_code=True,
    )
    assert pipe == mock_pipe_instance


def test_generate_text():
    """Test generate_text returns valid output."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": "This is a test response."}]

    prompt = "Hello"
    output = generate_text(mock_pipe, prompt, max_new_tokens=50)

    assert output == "This is a test response."

    # Verify default kwargs
    mock_pipe.assert_called_with(
        prompt,
        max_new_tokens=50,
        min_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
    )


def test_generate_text_error():
    """Test generate_text handles exceptions."""
    mock_pipe = MagicMock()
    mock_pipe.side_effect = Exception("Model Error")

    output = generate_text(mock_pipe, "Hello")

    assert "Error generating text" in output
    assert "Model Error" in output
