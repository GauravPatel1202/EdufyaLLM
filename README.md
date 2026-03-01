# EdufyaLLM

> A custom 15M parameter educational LLM built from scratch with a Transformer architecture.

## Project Structure

```
EdufyaLLM/
├── api.py                          # FastAPI production server
├── models/
│   ├── architecture.py             # Custom Transformer model (RoPE, RMSNorm, SwiGLU)
│   ├── edufya-tiny-15m.pt          # Trained model weights
│   └── edufya-tokenizer/           # BPE tokenizer (16K vocab)
├── training/
│   ├── train.py                    # Model training script
│   └── train_tokenizer.py          # Tokenizer training script
├── data/
│   ├── generate_react_data.py      # React + Math dataset generator
│   └── processed.jsonl             # Training data
├── utils/
│   ├── scraper.py                  # Website scraper
│   ├── preprocess.py               # Data preprocessor
│   └── vector_db.py                # ChromaDB vector store
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python -m data.generate_react_data
```

### 3. Train Tokenizer & Model

```bash
python -m training.train_tokenizer
python -m training.train
```

### 4. Run API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint  | Description                      |
| ------ | --------- | -------------------------------- |
| GET    | `/`       | API info                         |
| GET    | `/health` | Health check & model status      |
| POST   | `/chat`   | Chat with the model              |
| POST   | `/train`  | Scrape a URL & retrain the model |

### Chat Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is React?", "max_tokens": 128, "temperature": 0.7}'
```

### Train on a URL

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"url": "https://react.dev/"}'
```

## Model Architecture

- **Type**: Decoder-only Transformer
- **Parameters**: ~15M
- **Hidden size**: 256
- **Layers**: 6
- **Attention heads**: 8
- **Context length**: 128 tokens
- **Positional encoding**: RoPE (Rotary Position Embeddings)
- **Normalization**: RMSNorm
- **FFN**: SwiGLU activation
- **Tokenizer**: BPE (16,384 vocab)

## Docker

```bash
docker-compose up --build
```

## License

Private educational project.
# Omllm
