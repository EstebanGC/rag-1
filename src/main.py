import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

try: 
    from config.settings import OLLAMA_CONFIG, DATA_PATHS
    from utils.helpers import setup_enviroment
except ImportError as e:
    print(f"Importing Error: {e}")
    print("Make sure you're running from src folder")
    sys.exit(1)

def main():
    """Main function"""
    print("Initializing RAG")

    setup_enviroment()

    print("Enviroment configured correctly")
    print(f"LLM Model: {OLLAMA_CONFIG['models']['llm']}")

if __name__ == "__main__":
    main()
