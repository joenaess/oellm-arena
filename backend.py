import torch
from transformers import pipeline


def get_pipeline(model_name, device_id):
    """
    Loads a pipeline for a specific model onto a specific GPU.
    """
    print(f"Loading {model_name} on GPU {device_id}...")

    pipe = pipeline(
        "text-generation", model=model_name, device=device_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
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

        output = pipe(
            prompt,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=rep_pen,
        )
        return output[0]["generated_text"]
    except Exception as e:
        return f"Error generating text: {str(e)}"
