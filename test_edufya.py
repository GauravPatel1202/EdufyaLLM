
import torch
import os
from transformers import PreTrainedTokenizerFast
from models.architecture import EdufyaLLM

# ── EXACT 15M PRODUCTION MATCH ──
MODEL_PATH = "models/edufya-tiny-15m.pt"
TOKENIZER_PATH = "models/edufya-tokenizer"
device = "mps" if torch.backends.mps.is_available() else "cpu"

def test_model():
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model not found.")
        return

    # 1. Use the original 620-vocab tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    
    # 2. Load the specific weight config
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    config = checkpoint["config"]
    
    print(f"Loading OmLLM-Tiny (15M) | Vocab: {config.vocab_size} | Hidden: {config.hidden_size}")
    
    model = EdufyaLLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Use the EXACT prompt that was in the data/processed.jsonl
    # Note: "educational tutor" instead of just "tutor"
    prompt = "<|im_start|>system\nYou are a helpful educational tutor.<|im_end|>\n<|im_start|>user\nWhat is React?<|im_end|>\n<|im_start|>assistant\n"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # High-stability settings
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=40, 
            temperature=0.01, # Almost deterministic
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # 4. Also test a simple Math question to verify overall 'consciousness'
    math_prompt = "<|im_start|>system\nYou are a helpful educational tutor.<|im_end|>\n<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n"
    math_ids = tokenizer.encode(math_prompt, return_tensors="pt").to(device)
    math_output = model.generate(math_ids, max_new_tokens=10, temperature=0.01)
    math_response = tokenizer.decode(math_output[0][math_ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"\n[React Test]: {response}")
    print(f"[Math Test]: {math_response}")

if __name__ == "__main__":
    test_model()
