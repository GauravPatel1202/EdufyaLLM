import os
# Enable MPS for Mac acceleration
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
# Optimize CPU performance
if not torch.backends.mps.is_available():
    torch.set_num_threads(min(os.cpu_count(), 4))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import threading
from utils.scraper import scrape_url
from utils.preprocess import preprocess_data
from training.train import train as train_model
from utils.vector_db import vector_db
import json

app = FastAPI(title="EdufyaLLM API", description="API for the private Educational LLM")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "models/educational-qwen-0.5b-lora"

# Global variables for model and tokenizer
model = None
tokenizer = None

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    time_taken: float

@app.on_event("startup")
def load_model():
    global model, tokenizer
    # Detect device: MPS for Mac GPU, else CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading base model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Load base model with float16 for performance
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)
    
    # Load adapter if it exists and is ready
    adapter_config_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print(f"Loading adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        print("Adapter loaded successfully!")
    else:
        print(f"Note: Adapter not found at {adapter_config_path} yet. Using base model.")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the EdufyaLLM API",
        "endpoints": {
            "/chat": "POST - Send a prompt to get an educational response",
            "/health": "GET - Check API and model adapter status",
            "/docs": "GET - Interactive Swagger UI documentation"
        },
        "usage": "Use /docs to test the API directly from your browser."
    }

class TrainRequest(BaseModel):
    url: str
    num_epochs: int = 1

def run_training_pipeline(url: str, num_epochs: int):
    raw_file = "data/raw_scraped.txt"
    processed_file = "data/processed_scraped.jsonl"
    
    if scrape_url(url, raw_file):
        num_examples = preprocess_data(raw_file, processed_file)
        if num_examples > 0:
            # Index into Vector DB
            print("Indexing documents into Vector DB...")
            with open(processed_file, "r") as f:
                lines = f.readlines()
                docs = []
                ids = []
                for i, line in enumerate(lines):
                    data = json.loads(line)
                    # Extract the informative part (assuming 'text' or similar after preprocessing)
                    # For now, we index the full JSON line or specific fields if available
                    docs.append(data.get("output", data.get("instruction", "")))
                    ids.append(f"doc_{i}_{hash(url)}")
            vector_db.add_documents(docs, ids)
            
            print(f"Starting fine-tuning with {num_examples} examples...")
            train_model(data_path=processed_file, num_epochs=num_epochs)
            print("Training pipeline completed successfully.")
            # Optionally reload the model here if needed, 
            # but usually, we'd restart or have a reload mechanism.
            # For simplicity, we just finish.

@app.post("/train")
async def train_endpoint(request: TrainRequest):
    # Run in background to avoid timeout
    thread = threading.Thread(target=run_training_pipeline, args=(request.url, request.num_epochs))
    thread.start()
    return {"message": "Training pipeline started in the background", "url": request.url}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Retrieve relevant context from Vector DB
    context_list = vector_db.query(request.prompt)
    context_text = "\n".join(context_list) if context_list else "No additional context found."

    # Format for Qwen ChatML with Math Specialist Prompt and Context
    input_text = f"<|im_start|>system\nYou are an expert educational mathematics tutor. Use the provided context to answer the user's question accurately. Always use LaTeX for mathematical notation (e.g., use $x^2$ for powers). Provide professional, clear, and structured explanations.\n\nContext:\n{context_text}<|im_end|>\n<|im_start|>user\n{request.prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=request.temperature > 0,
            temperature=request.temperature if request.temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    response_text = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return ChatResponse(response=response_text, time_taken=round(time_taken, 2))

@app.get("/health")
def health():
    return {"status": "ok", "adapter_loaded": os.path.exists(ADAPTER_PATH)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
