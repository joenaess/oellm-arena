import torch
from backend import get_pipeline, generate_text
from config import MODELS_DB, EXAMPLE_PROMPTS
import gc
import traceback

def reproduce_stress():
    language = "Swedish"
    ms_models = MODELS_DB[language]["multisynt"]
    hplt_models = MODELS_DB[language]["hplt"]
    
    # Iterate through ALL combinations to find the killer
    for model_a in ms_models:
        model_b_list = [hplt_models] if isinstance(hplt_models, str) else hplt_models
        for model_b in model_b_list:
            print(f"\n\n=== Testing {model_a} vs {model_b} ===")
            
            prompt = "Det var en gång en" 
            
            # Mimic App Parameters exactly
            params = {
                "min_new_tokens": 70,
                "max_new_tokens": 112,
                "repetition_penalty": 1.2,
                "temperature": 0.7
            }
            
            # Run multiple times to stress memory/state
            for i in range(5):
                print(f"  Iteration {i+1}/5...")
                try:
                    print(f"    Loading {model_a} on GPU 0...")
                    pipe_a = get_pipeline(model_a, 0)
                    print("    Generating A...")
                    res_a = generate_text(pipe_a, prompt, **params)
                    print(f"    Result A length: {len(res_a)}")
                    
                    print(f"    Loading {model_b} on GPU 1...") # Assuming 2 GPUs
                    pipe_b = get_pipeline(model_b, 1)
                    print("    Generating B...")
                    res_b = generate_text(pipe_b, prompt, **params)
                    print(f"    Result B length: {len(res_b)}")
                    
                    # Cleanup
                    del pipe_a
                    del pipe_b
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    print(f"\nCRASHED at Iteration {i+1}: {e}")
                    traceback.print_exc()
                    return # Stop on first crash

if __name__ == "__main__":
    reproduce_stress()
