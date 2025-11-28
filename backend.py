import torch
import random
from transformers import pipeline

# --- MODEL DATABASE ---
# Structure: Language Name -> { 'multisynt': [List of A models], 'hplt': 'B model ID' }
MODELS_DB = {
    "Icelandic": {
        "multisynt": [
            "MultiSynt/nemotron-cc-icelandic-tower9b",
            "MultiSynt/nemotron-cc-icelandic-opus"
        ],
        "hplt": "HPLT/hplt2c_isl_checkpoints"
    },
    "Swedish": {
        "multisynt": [
            "MultiSynt/nemotron-cc-swedish-tower9b",
            "MultiSynt/nemotron-cc-swedish-tower72b",
            "MultiSynt/nemotron-cc-swedish-opus"
        ],
        "hplt": "HPLT/hplt2c_swe_checkpoints"
    },
    "Danish": {
        "multisynt": [
            "MultiSynt/nemotron-cc-danish-tower9b",
            "MultiSynt/nemotron-cc-danish-opus"
        ],
        "hplt": "HPLT/hplt2c_dan_checkpoints"
    },
    "Norwegian (Bokm�l)": {
        "multisynt": [
            "MultiSynt/nemotron-cc-norwegian-tower9b"
        ],
        "hplt": "HPLT/hplt2c_nob_checkpoints"
    },
    "Finnish": {
        "multisynt": [
            "MultiSynt/nemotron-cc-finnish-tower9b"
        ],
        "hplt": "HPLT/hplt2c_fin_checkpoints"
    },
    "German": {
        "multisynt": [
            "MultiSynt/nemotron-cc-german-tower9b",
            "MultiSynt/nemotron-cc-german-opus"
        ],
        "hplt": "HPLT/hplt2c_deu_checkpoints"
    },
    "Dutch": {
        "multisynt": [
            "MultiSynt/nemotron-cc-dutch-tower9b",
            "MultiSynt/nemotron-cc-dutch-opus"
        ],
        "hplt": "HPLT/hplt2c_nld_checkpoints"
    },
    "Spanish": {
        "multisynt": [
            "MultiSynt/nemotron-cc-spanish-tower9b",
            "MultiSynt/nemotron-cc-spanish-tower72b"
        ],
        "hplt": "HPLT/hplt2c_spa_checkpoints"
    },
    "Italian": {
        "multisynt": [
            "MultiSynt/nemotron-cc-italian-tower72b",
            "MultiSynt/nemotron-cc-italian-opus"
        ],
        "hplt": "HPLT/hplt2c_ita_checkpoints"
    },
    "Portuguese": {
        "multisynt": [
            "MultiSynt/nemotron-cc-portuguese-opus"
        ],
        "hplt": "HPLT/hplt2c_por_checkpoints"
    },
    "Romanian": {
        "multisynt": [
            "MultiSynt/nemotron-cc-romanian-tower9b",
            "MultiSynt/nemotron-cc-romanian-opus"
        ],
        "hplt": "HPLT/hplt2c_ron_checkpoints"
    },
    "Catalan": {
        "multisynt": [
            "MultiSynt/nemotron-cc-catalan-opus"
        ],
        "hplt": "HPLT/hplt2c_cat_checkpoints"
    },
    "Basque": {
        "multisynt": [
            "MultiSynt/nemotron-cc-basque-opus"
        ],
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

def generate_text(pipe, prompt, max_new_tokens=200):
    try:
        output = pipe(
            prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        return output[0]['generated_text']
    except Exception as e:
        return f"Error generating text: {str(e)}"