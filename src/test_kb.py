# src/test_kb.py
import sys
from pathlib import Path

src_path = Path(__file__).parent
project_root = src_path.parent
sys.path.append(str(project_root))

from knowledge_base import KnowledgeBase
import logging

logging.basicConfig(level=logging.INFO)

def test_knowledge_base():
    print("🧪 Testing Knowledge Base Load...")
    
    kb = KnowledgeBase()
    
    # Intentar cargar la base existente
    if kb.load_knowledge_base():
        print("Knowledge base loaded successfully!")
        print(f"Vector store: {kb.vector_store}")
        
        # Probar búsqueda
        results = kb.search_similar_documents("test", k=1)
        print(f"Search test returned {len(results)} results")
    else:
        print("Knowledge base NOT found or failed to load")
        print("Run: python src/main.py --build")

if __name__ == "__main__":
    test_knowledge_base()