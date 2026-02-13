import pytest
import sys
import os
import torch
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import get_pipeline, generate_text
from config import MODELS_DB, EXAMPLE_PROMPTS

# Skip if no GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for integration test")
def test_swedish_generation_integration():
    """
    Integration test verifying Swedish model loading and generation on GPU.
    This attempts to reproduce the CUBLAS_STATUS_INVALID_VALUE error with specific Swedish models.
    """
    language = "Swedish"
    models = MODELS_DB[language]
    prompts = EXAMPLE_PROMPTS[language]
    
    # Test Model A (MultiSynt)
    # Using the first one in the list, or we could iterate
    model_a_name = models["multisynt"][0]
    device_a = 0
    
    # Test Model B (HPLT)
    model_b_name = models["hplt"]
    device_b = 1 if torch.cuda.device_count() > 1 else 0
    
    prompt = prompts[0]
    
    print(f"\nTesting Swedish models: {model_a_name} (GPU {device_a}) and {model_b_name} (GPU {device_b})")
    
    try:
        # Load Model A
        print(f"Loading {model_a_name}...")
        pipe_a = get_pipeline(model_a_name, device_a)
        output_a = generate_text(pipe_a, prompt, max_new_tokens=20)
        print(f"Model A Output: {output_a}")
        assert len(output_a) > len(prompt)
        
        # Load Model B
        print(f"Loading {model_b_name}...")
        pipe_b = get_pipeline(model_b_name, device_b)
        output_b = generate_text(pipe_b, prompt, max_new_tokens=20)
        print(f"Model B Output: {output_b}")
        assert len(output_b) > len(prompt)

        # Clean up
        del pipe_a, pipe_b
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        pytest.fail(f"Swedish Model Integration test failed with error: {e}")
