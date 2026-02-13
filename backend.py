import torch
from transformers import pipeline


def get_pipeline(model_name, device_id):
    """
    Loads a pipeline for a specific model.
    Ignores device_id and uses device_map="auto" to handle resource contention (e.g. vLLM on GPU 1).
    """
    print(f"Loading {model_name} with device_map='auto'...")

    pipe = pipeline(
        "text-generation", 
        model=model_name, 
        device_map="auto", 
        torch_dtype=torch.float32, 
        model_kwargs={"attn_implementation": "sdpa"},
        trust_remote_code=True
    )
    
    # Critical fix for models without pad_token
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        
    # FORCE LEFT PADDING for decoder-only generation to avoid shape errors
    pipe.tokenizer.padding_side = "left"
        
    return pipe


def generate_text(pipe, prompt, **kwargs):
    """
    Generates text using the provided pipeline with dynamic arguments.
    """
    try:
        # Default fallbacks if not provided in kwargs
        max_new = kwargs.get("max_new_tokens", 256)
        min_new = kwargs.get("min_new_tokens", 20)
        temp = kwargs.get("temperature", 0.7)
        rep_pen = kwargs.get("repetition_penalty", 1.15)
        
        # Ensure we don't pass 'max_length' if 'max_new_tokens' is present
        # and explicitly disable it to prevent config defaults from causing issues
        
        output = pipe(
            prompt,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            max_length=None, # Explicitly unset max_length default
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=rep_pen,
            pad_token_id=pipe.tokenizer.pad_token_id, # Explicitly pass pad_token_id
        )
        return output[0]["generated_text"]
    except Exception as e:
        return f"Error generating text: {str(e)}"
