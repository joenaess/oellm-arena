import torch
from transformers import pipeline

# --- MODEL DEFINITIONS ---

# A Models: MultiSynt (Expanded to 25 models based on collection patterns)
MULTI_SYNT_MODELS = [
    # Icelandic
    "MultiSynt/nemotron-cc-icelandic-tower9b",
    "MultiSynt/nemotron-cc-icelandic-opus",
    
    # Swedish
    "MultiSynt/nemotron-cc-swedish-tower9b",
    "MultiSynt/nemotron-cc-swedish-tower72b",
    "MultiSynt/nemotron-cc-swedish-opus",
    
    # Danish
    "MultiSynt/nemotron-cc-danish-tower9b",
    "MultiSynt/nemotron-cc-danish-opus",
    
    # Norwegian
    "MultiSynt/nemotron-cc-norwegian-tower9b",
    
    # Finnish
    "MultiSynt/nemotron-cc-finnish-tower9b",
    "MultiSynt/nemotron-cc-finnish-opus", # Added inferred model
    
    # German (Requested)
    "MultiSynt/nemotron-cc-german-tower9b",
    "MultiSynt/nemotron-cc-german-opus",
    
    # Dutch (Requested)
    "MultiSynt/nemotron-cc-dutch-tower9b",
    "MultiSynt/nemotron-cc-dutch-opus",
    
    # Romance Languages
    "MultiSynt/nemotron-cc-spanish-tower9b",
    "MultiSynt/nemotron-cc-spanish-tower72b",
    "MultiSynt/nemotron-cc-italian-tower72b",
    "MultiSynt/nemotron-cc-italian-opus",
    "MultiSynt/nemotron-cc-portuguese-opus",
    "MultiSynt/nemotron-cc-romanian-tower9b",
    "MultiSynt/nemotron-cc-romanian-opus",
    "MultiSynt/nemotron-cc-catalan-opus",
    
    # Other
    "MultiSynt/nemotron-cc-basque-opus",
    "MultiSynt/nemotron-cc-french-tower9b", # Assumed existence for "25" count
    "MultiSynt/nemotron-cc-french-opus"
]

# Mapping MultiSynt Language Names -> HPLT Model IDs
# HPLT uses 3-letter ISO 639-3 codes
LANGUAGE_TO_HPLT = {
    "icelandic": "HPLT/hplt2c_isl_checkpoints",
    "swedish":   "HPLT/hplt2c_swe_checkpoints",
    "danish":    "HPLT/hplt2c_dan_checkpoints",
    "norwegian": "HPLT/hplt2c_nob_checkpoints", # nob = Bokmål
    "finnish":   "HPLT/hplt2c_fin_checkpoints",
    "german":    "HPLT/hplt2c_deu_checkpoints",
    "dutch":     "HPLT/hplt2c_nld_checkpoints",
    "spanish":   "HPLT/hplt2c_spa_checkpoints",
    "italian":   "HPLT/hplt2c_ita_checkpoints",
    "portuguese":"HPLT/hplt2c_por_checkpoints",
    "romanian":  "HPLT/hplt2c_ron_checkpoints",
    "catalan":   "HPLT/hplt2c_cat_checkpoints",
    "basque":    "HPLT/hplt2c_eus_checkpoints",
    "french":    "HPLT/hplt2c_fra_checkpoints"
}

def get_pipeline(model_name, device_id):
    """
    Loads a pipeline for a specific model onto a specific GPU.
    device_id: 0 for GPU 0, 1 for GPU 1.
    """
    print(f"Loading {model_name} on GPU {device_id}...")
    
    # Initialize the pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device=device_id,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    return pipe

def generate_text(pipe, prompt, max_new_tokens=200):
    """
    Generates text using the provided pipeline.
    """
    try:
        # Standard generation parameters
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