import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import get_pipeline, generate_text
import torch
import gc

# Skip if no GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for integration test")
def test_gpu_generation_integration():
    """
    Integration test to verifying actual model loading and generation on GPU.
    This attempts to reproduce the CUBLAS_STATUS_INVALID_VALUE error.
    """
    # Use a small consistent model for testing if possible, 
    # but to reproduce the error we might need the actual models from config.
    # Let's try to load one of the reported failing models or a similar one.
    
    # We'll use a known model key from the app's config
    model_id = "MultiSynt/nemotron-cc-icelandic-tower9b" 
    device_id = 0
    
    print(f"\nTesting load and generation for {model_id} on GPU {device_id}")
    
    try:
        pipe = get_pipeline(model_id, device_id)
        assert pipe is not None
        
        prompt = "Einu sinni var"
        output = generate_text(pipe, prompt, max_new_tokens=20)
        
        print(f"Generated: {output}")
        assert len(output) > len(prompt)
        assert isinstance(output, str)
        
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        pytest.fail(f"GPU Integration test failed with error: {e}")
