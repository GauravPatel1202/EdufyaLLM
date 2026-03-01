
"""
EdufyaLLM Production Training Script
Trains the 100M parameter Transformer model for long-term production use.
"""
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.architecture import EdufyaLLM, ModelConfig
from transformers import PreTrainedTokenizerFast
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console

console = Console()

class EdufyaDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.examples = []
        if not os.path.exists(data_path):
            console.print(f"[bold red]✗ Data file not found: {data_path}[/bold red]")
            return
            
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    encoded = tokenizer(item["text"], truncation=True, max_length=max_length, padding=False, return_tensors="pt")
                    if encoded["input_ids"].shape[1] > 1:
                        self.examples.append(encoded["input_ids"].squeeze(0))
                except: continue

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

def collate_fn(batch, pad_token_id=0):
    max_len = max(len(x) for x in batch)
    padded_batch = []
    for x in batch:
        pad_len = max_len - len(x)
        padded = torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_batch.append(padded)
    return torch.stack(padded_batch)

def train(data_path: str = "data/processed.jsonl"):
    tokenizer_path = "models/edufya-tokenizer"
    model_save_path = "models/edufya-prod.pt"
    
    # ── PRODUCTION HYPERPARAMETERS ──
    batch_size = 2 # Small batch to fit 100M parameters in RAM
    lr = 3e-5
    epochs = 40
    max_length = 512
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if not os.path.exists(tokenizer_path):
        console.print("[red]✗ Run train_tokenizer.py first.[/red]")
        return

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    pad_id = tokenizer.pad_token_id

    config = ModelConfig(
        vocab_size=len(tokenizer),
        max_context_len=max_length,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072
    )
    
    model = EdufyaLLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    console.print(f"[cyan]Training Production Model: {num_params / 1e6:.2f}M parameters[/cyan]")

    dataset = EdufyaDataset(data_path, tokenizer, max_length=max_length)
    from functools import partial
    custom_collate = partial(collate_fn, pad_token_id=pad_id)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    console.print(f"\n[bold]Starting Production Training Run — {len(dataset)} examples[/bold]\n")
    model.train()

    with Progress(TextColumn("{task.description}"), BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TimeRemainingColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Training...", total=len(dataloader)*epochs)
        for epoch in range(epochs):
            for batch in dataloader:
                batch = batch.to(device)
                inputs, targets = batch[:, :-1], batch[:, 1:]
                
                optimizer.zero_grad()
                logits, loss = model(inputs, targets)

                if torch.isnan(loss):
                    console.print("[red]!!! Loss is NaN. Aborting.[/red]")
                    return

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                progress.update(task, advance=1, description=f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    torch.save({"model_state_dict": model.state_dict(), "config": config}, model_save_path)
    console.print(f"\n[bold green]✓ Production Model Training Complete![/bold green]")

if __name__ == "__main__":
    train()
