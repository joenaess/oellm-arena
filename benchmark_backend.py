import argparse
import gc
import os

import pandas as pd
import torch

from backend import generate_text, get_pipeline
from config import EXAMPLE_PROMPTS, MODELS_DB

RESULTS_FILE = "backend_benchmark_results.csv"


def run_benchmark(limit=None):
    results = []

    languages = list(MODELS_DB.keys())
    if limit:
        languages = languages[:limit]

    for lang in languages:
        print(f"\n--- Benchmarking Language: {lang} ---")

        # Get models
        hplt_model = MODELS_DB[lang]["hplt"]
        multisynt_options = MODELS_DB[lang]["multisynt"]

        # Benchmark against the first available MultiSynt model to save time/resources
        # In a full run, we could iterate over all options
        multisynt_model = multisynt_options[0]

        prompts = EXAMPLE_PROMPTS.get(lang, ["Hello world"])

        try:
            # Load HPLT Model on GPU 1
            print(f"Loading HPLT model: {hplt_model}")
            pipe_hplt = get_pipeline(hplt_model, 1)

            # Load MultiSynt Model on GPU 0
            print(f"Loading MultiSynt model: {multisynt_model}")
            pipe_ms = get_pipeline(multisynt_model, 0)

            for prompt in prompts:
                print(f"Generating for prompt: {prompt[:50]}...")

                # Generate HPLT
                out_hplt = generate_text(pipe_hplt, prompt, max_new_tokens=100)

                # Generate MultiSynt
                out_ms = generate_text(pipe_ms, prompt, max_new_tokens=100)

                results.append(
                    {
                        "Language": lang,
                        "Prompt": prompt,
                        "HPLT_Model": hplt_model,
                        "MultiSynt_Model": multisynt_model,
                        "HPLT_Output": out_hplt,
                        "MultiSynt_Output": out_ms,
                    }
                )

            # Cleanup
            del pipe_hplt
            del pipe_ms
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error processing {lang}: {e}")
            continue

    # Save Results
    df = pd.DataFrame(results)
    if os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)

    print(f"\nBenchmark complete. Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of languages to test")
    args = parser.parse_args()

    run_benchmark(limit=args.limit)
