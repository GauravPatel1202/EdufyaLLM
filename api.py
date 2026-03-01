
"""
EdufyaLLM API - PRODUCTION VERSION (with RAG)
A FastAPI-based educational LLM inference server using Retrieval-Augmented Generation.
"""
import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerFast
from models.architecture import EdufyaLLM, ModelConfig
from utils.vector_db import vector_db
import logging
import time

# ── Logging Setup ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("edufya_prod")

# ── Globals ──
MODEL_PATH = "models/edufya-tiny-15m.pt"
TOKENIZER_PATH = "models/edufya-tokenizer"
device = "mps" if torch.backends.mps.is_available() else "cpu"

app = FastAPI(title="EdufyaLLM Production API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load State
tokenizer = None
model = None

def load_assets():
    global tokenizer, model
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
        logger.info(f"Tokenizer loaded ({len(tokenizer)} tokens)")
        
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            config = checkpoint["config"]
            model = EdufyaLLM(config).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            logger.info("Production model loaded.")
        else:
            logger.warning("Production model not found. API will run in RAG-only fallback mode if possible.")
    except Exception as e:
        logger.error(f"Error loading assets: {e}")

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    use_rag: bool = True
    max_tokens: int = 256

@app.on_event("startup")
async def startup():
    load_assets()

@app.get("/health")
def health():
    return {"status": "ok", "model": model is not None, "device": device}

@app.post("/chat")
async def chat(request: ChatRequest):
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer not initialized.")

    start_time = time.time()
    
    # ── RAG: Retrieve Context ──
    context = ""
    if request.use_rag:
        try:
            results = vector_db.search(request.prompt, n_results=2)
            if results and 'documents' in results and results['documents']:
                context = "\n".join(results['documents'][0])
                logger.info(f"Retrieved {len(results['documents'][0])} document snippets for context.")
        except Exception as e:
            logger.error(f"RAG Error: {e}")

    # ── Prompt Construction ──
    system_prompt = "You are a helpful educational tutor."
    if context:
        system_prompt += f" Use this verified documentation to answer: {context}"
    
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{request.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # ── Inference ──
    if model is not None:
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=request.max_tokens, 
                temperature=0.4, # Lower for production stability
                eos_token_id=tokenizer.eos_token_id
            )
        response_ids = output_ids[0][input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    else:
        response_text = "Model is currently training. (RAG Fallback: Found relevant info but LLM is offline.)"

    time_taken = time.time() - start_time
    logger.info(f"Inference complete in {time_taken:.2f}s")

    return {
        "response": response_text,
        "context_used": bool(context),
        "time": round(time_taken, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
