import sys
from pathlib import Path

# Add config to path
config_path = Path(__file__).parent / "config"
sys.path.append(str(config_path))

try:
    from settings import OLLAMA_CONFIG, DATA_PATHS, PROCESSING_CONFIG

    print("CONFIGURATION TEST")
    print("=" * 40)
    
    print("OLLAMA MODELS:")
    print(f"   Coder: {OLLAMA_CONFIG['models']['coder']}")
    print(f"   Assistant: {OLLAMA_CONFIG['models']['assistant']}")
    print(f"   Embeddings: {OLLAMA_CONFIG['models']['embeddings']}")
    
    print("\n📁 DATA PATHS:")
    for name, path in DATA_PATHS.items():
        exists = "EXISTS" if path.exists() else "MISSING"
        print(f"   {name}: {path} {exists}")
    
    print("\nPROCESSING CONFIG:")
    print(f"   Chunk Size: {PROCESSING_CONFIG['chunk_size']}")
    print(f"   Chunk Overlap: {PROCESSING_CONFIG['chunk_overlap']}")
    
    print("\nNEXT STEPS:")
    print("1. Create knowledge_base.py")
    print("2. Create requirements.txt") 
    print("3. Run: python src/main.py --build")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("💡 Check your file structure and imports")
except Exception as e:
    print(f"Error: {e}")