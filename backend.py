import torch
import random
from transformers import pipeline

# --- MODEL DATABASE ---
MODELS_DB = {
    "Icelandic": {
        "multisynt": ["MultiSynt/nemotron-cc-icelandic-tower9b", "MultiSynt/nemotron-cc-icelandic-opus"],
        "hplt": "HPLT/hplt2c_isl_checkpoints"
    },
    "Swedish": {
        "multisynt": ["MultiSynt/nemotron-cc-swedish-tower9b", "MultiSynt/nemotron-cc-swedish-tower72b", "MultiSynt/nemotron-cc-swedish-opus"],
        "hplt": "HPLT/hplt2c_swe_checkpoints"
    },
    "Danish": {
        "multisynt": ["MultiSynt/nemotron-cc-danish-tower9b", "MultiSynt/nemotron-cc-danish-opus"],
        "hplt": "HPLT/hplt2c_dan_checkpoints"
    },
    "Norwegian (Bokm�l)": {
        "multisynt": ["MultiSynt/nemotron-cc-norwegian-tower9b"],
        "hplt": "HPLT/hplt2c_nob_checkpoints"
    },
    "Finnish": {
        "multisynt": ["MultiSynt/nemotron-cc-finnish-tower9b"],
        "hplt": "HPLT/hplt2c_fin_checkpoints"
    },
    "German": {
        "multisynt": ["MultiSynt/nemotron-cc-german-tower9b", "MultiSynt/nemotron-cc-german-opus"],
        "hplt": "HPLT/hplt2c_deu_checkpoints"
    },
    "Dutch": {
        "multisynt": ["MultiSynt/nemotron-cc-dutch-tower9b", "MultiSynt/nemotron-cc-dutch-opus"],
        "hplt": "HPLT/hplt2c_nld_checkpoints"
    },
    "Spanish": {
        "multisynt": ["MultiSynt/nemotron-cc-spanish-tower9b", "MultiSynt/nemotron-cc-spanish-tower72b"],
        "hplt": "HPLT/hplt2c_spa_checkpoints"
    },
    "Italian": {
        "multisynt": ["MultiSynt/nemotron-cc-italian-tower72b", "MultiSynt/nemotron-cc-italian-opus"],
        "hplt": "HPLT/hplt2c_ita_checkpoints"
    },
    "Portuguese": {
        "multisynt": ["MultiSynt/nemotron-cc-portuguese-opus"],
        "hplt": "HPLT/hplt2c_por_checkpoints"
    },
    "Romanian": {
        "multisynt": ["MultiSynt/nemotron-cc-romanian-tower9b", "MultiSynt/nemotron-cc-romanian-opus"],
        "hplt": "HPLT/hplt2c_ron_checkpoints"
    },
    "Catalan": {
        "multisynt": ["MultiSynt/nemotron-cc-catalan-opus"],
        "hplt": "HPLT/hplt2c_cat_checkpoints"
    },
    "Basque": {
        "multisynt": ["MultiSynt/nemotron-cc-basque-opus"],
        "hplt": "HPLT/hplt2c_eus_checkpoints"
    }
}

def get_pipeline(model_name, device_id):
    """
    Loads a pipeline for a specific model onto a specific GPU.
    """
    print(f"Loading {model_name} on GPU {device_id}...")
    
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device=device_id,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    return pipe

def generate_text(pipe, prompt, **kwargs):
    """
    Generates text using the provided pipeline with dynamic arguments.
    """
    try:
        # Default fallbacks if not provided in kwargs
        max_new = kwargs.get('max_new_tokens', 256)
        min_new = kwargs.get('min_new_tokens', 20)
        temp = kwargs.get('temperature', 0.7)
        rep_pen = kwargs.get('repetition_penalty', 1.15)
        
        output = pipe(
            prompt, 
            max_new_tokens=max_new,
            min_new_tokens=min_new, # Force models to speak at least this much
            do_sample=True, 
            temperature=temp,
            top_p=0.9,
            repetition_penalty=rep_pen
        )
        return output[0]['generated_text']
    except Exception as e:
        return f"Error generating text: {str(e)}"