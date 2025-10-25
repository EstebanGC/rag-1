import logging
from typing import List, Optional
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from config.settings import DATA_PATHS, OLLAMA_CONFIG

logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
        model=OLLAMA_CONFIG["models"]["embeddings"],
        base_url=OLLAMA_CONFIG["base_url"]
    )      
        self.vector_store = None

    def create_knowledge_base(self, documents: List[Document]) -> bool:
        try:
            logger.info("Creating embeddings and vector store... ")

            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            logger.info(f"Knowledge base created with {len(documents)} documents")
            return True
    
        except Exception as e:
            logger.error(f"Error creating knowledge base: {e}")
            return False
    
    def save_knowledge_base(self) -> bool:
        try:
            if self.vector_store:
                save_path = DATA_PATHS["vector_store"]
                save_path.mkdir(parents=True, exist_ok=True)

                self.vector_store.save_local(str(save_path))
                logger.info(f"Base saved in: {save_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
        
    def load_knowledge_base(self) -> bool:
        try:
            load_path = DATA_PATHS["vector_store"]

            if load_path.exists() and any(load_path.iterdir()):
                self.vector_store = FAISS.load_local(
                    folder_path=str(load_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Knowledge base loaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False

    def search_similar_documents(self, query: str, k: int=4) -> List[Document]:
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []
    
    def get_retriever(self, k: int =4):
        if self.vector_store:
            return self.vector_store.as_retriever(search_kwargs ={"k": k})
        return None
    
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Knowledge Base...")
    kb = KnowledgeBase()
    print(f"Knowledge Base initialized")
    print(f"Embeddings model: {OLLAMA_CONFIG['models']['embeddings']}")
    
    

        


    
