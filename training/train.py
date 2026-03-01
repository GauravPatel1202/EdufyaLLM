import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def train(data_path="data/processed.jsonl", num_epochs=1):
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir = "models/educational-qwen-0.5b-lora"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}. Run scraper and preprocess first.")
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Quantization config (optional, might not work on CPU as expected, but kept for pattern)
    # On CPU, we usually just use float32 or bfloat16 if supported.
    # For Intel Mac, we'll stick to float32 or float16 if possible.
    
    print(f"Loading base model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32, # CPU friendly
        device_map={"": "cpu"}, # Force CPU
    )

    # LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no", # Save manually at the end for simplicity
        push_to_hub=False,
        report_to="none",
        use_cpu=True, # Force CPU training
    )

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text", # Using the pre-formatted ChatML field
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    # Save the adapter
    print(f"Saving fine-tuned adapter to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # Ensure trl is installed (added to requirements implicitly if I missed it)
    try:
        import trl
    except ImportError:
        print("Installing trl...")
        os.system("python3.12 -m pip install trl")
        
    train()
