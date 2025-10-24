import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "timeout": 60.0,
    "models": {
        "coder": "mistral",
        "assistant": "genma:2b",
        "embeddings": "nomic-embed-text"
    }
}

PROCESSING_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "supported_extensions": [".pdf", ".docx", ".txt", ".md"]
}

DATA_PATHS = {
    "raw_documents": DATA_DIR / "raw",
    "processed_documents": DATA_DIR / "processed",
    "vector_store": MODELS_DIR / "vector_store"
}

for path in DATA_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)
