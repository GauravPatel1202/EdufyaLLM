# 🎓 EdufyaLLM: Private Educational AI API

EdufyaLLM is a custom-trained, small-scale Large Language Model (LLM) designed specifically for **educational mathematics tutoring**. It leverages a base Qwen model and is fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) to provide step-by-step mathematical explanations with LaTeX formatting, delivered via a production-ready REST API.

---

## 🚀 Key Features

- **REST API First**: Designed to be integrated into any frontend or mobile application.
- **Expert Mathematics Specialist**: System-prompted to provide clear, professional, and structured math solutions.
- **Contextual Memory (RAG)**: Uses **ChromaDB** to remember and retrieve relevant facts from training data for more accurate answers.
- **LaTeX Math Output**: Generates clean mathematical notation (e.g., $x^2 + y^2 = z^2$).
- **Speed Optimized (Mac)**: Uses **MPS (Metal Performance Shaders)** and `float16` precision for fast inference on Mac GPU hardware.
- **End-to-End Pipeline**: Includes tools for web scraping, data preprocessing, and model fine-tuning.
- **Docker Enabled**: Production-ready containerization for easy cloud deployment.

---

## 🛠 Dependencies & Why They are Used

| Dependency                    | Purpose                                                                                                  |
| :---------------------------- | :------------------------------------------------------------------------------------------------------- |
| **transformers**              | The core library for loading the LLM (Qwen) and handling text generation.                                |
| **peft**                      | Enables **LoRA (Low-Rank Adaptation)** training, allowing you to fine-tune the model on your laptop.     |
| **torch / accelerate**        | The deep learning backend, optimized with `accelerate` for faster weights loading and device management. |
| **fastapi / uvicorn**         | Serves the model as a high-performance web API with automatic documentation.                             |
| **chromadb**                  | The Vector Database used for long-term memory and context retrieval (RAG).                               |
| **sentence-transformers**     | Provides the embedding model used to turn text into vectors for the database.                            |
| **beautifulsoup4 / requests** | Used in the scraping logic to collect educational content from the web for training.                     |
| **datasets / trl**            | Handles the data pipeline and the supervised fine-tuning (SFT) process.                                  |
| **rich**                      | Used for beautiful and readable terminal output during training progress.                                |

---

## 🚦 Getting Started

### 1. Installation

Clone the repo and install the requirements:

```bash
pip install -r requirements.txt
```

### 2. Running the API (FastAPI)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

_Access the interactive Swagger documentation at [http://localhost:8000/docs](http://localhost:8000/docs)_

### 3. Usage Examples

#### Get a Chat Response

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain the Pythagorean Theorem"}'
```

#### Start a Training Pipeline

```bash
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/math-lesson", "num_epochs": 1}'
```

---

## 🐳 Docker Deployment

To run the API in a containerized environment:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

The API will be available at `http://localhost:8000`.
