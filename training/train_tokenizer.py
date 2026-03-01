import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

def train_tokenizer(data_path="data/processed.jsonl", vocab_size=16384, save_path="models/edufya-tokenizer"):
    if not os.path.exists("models"):
        os.makedirs("models")

    # Initialize a tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Create a trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "<|im_start|>", "<|im_end|>"],
        show_progress=True
    )

    # Prepare iterator for training
    def get_training_corpus():
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield data["text"]

    # Train the tokenizer
    print(f"Training tokenizer on {data_path} with vocab_size={vocab_size}...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # Save tokenizer
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    
    # Wrap in transformers PreTrainedTokenizerFast for easy use
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
    )
    fast_tokenizer.save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path}")

if __name__ == "__main__":
    train_tokenizer()
