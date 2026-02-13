import torch
from transformers import AutoConfig, pipeline

# The specific model to test
MODEL_ID = "MultiSynt/nemotron-cc-danish-tower9b"


def test_model():
    print(f"🔍 Inspecting: {MODEL_ID}")
    print("-" * 50)

    try:
        # 1. Print Config to check architecture
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("✅ Config loaded successfully")
        print(f"   Architecture: {config.architectures}")
        print(f"   Vocab Size: {config.vocab_size}")

        # 2. Load Pipeline
        print("\n⬇️ Loading model weights to GPU 0...")
        pipe = pipeline("text-generation", model=MODEL_ID, device=0, torch_dtype=torch.bfloat16, trust_remote_code=True)
        print("✅ Model loaded.")

        # 3. Generation Test with a strictly Danish prompt
        # If the model is actually Portuguese/Romanian, it will likely
        # switch languages immediately or hallucinate in the wrong language.
        prompt = "København er hovedstaden i Danmark og er kendt for "
        print("\n🧪 Testing Generation...")
        print(f"   Prompt: '{prompt}'")

        output = pipe(prompt, max_new_tokens=60, do_sample=True, temperature=0.7)

        generated_text = output[0]["generated_text"]
        print(f"\n📝 Output:\n{generated_text}")
        print("-" * 50)

        # 4. Simple Heuristic Check
        # Check for common Danish words vs Romance words
        lower_text = generated_text.lower()
        danish_signals = ["er", "den", "det", "og", "på", "en", "et"]
        romance_signals = ["el", "la", "de", "que", "o", "a", "é"]

        dn_score = sum(1 for w in danish_signals if f" {w} " in lower_text)
        ro_score = sum(1 for w in romance_signals if f" {w} " in lower_text)

        if ro_score > dn_score:
            print("\n⚠️ WARNING: The text contains significantly more Romance language markers than Danish.")
            print("   This suggests the model weights might be incorrect (e.g., actually Portuguese/Romanian).")
        else:
            print("\n✅ The text appears to be Danish.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")


if __name__ == "__main__":
    test_model()
