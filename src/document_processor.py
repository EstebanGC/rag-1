import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader, 
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import DATA_PATHS, PROCESSING_CONFIG

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader
        }

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PROCESSING_CONFIG["chunk_size"],
            chunk_overlap=PROCESSING_CONFIG["chunk_overlap"],
            length_function=len
        )

    def load_single_document(self, file_path: Path) -> Optional[List[Document]]:
        try: 
            extension = file_path.suffix.lower()
            if extension in self.loaders:
                loader = self.loaders[extension](str(file_path))
                return loader.load()
            else: 
                logger.warning(f"Format not supported: {extension}")
                return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def load_all_documents(self) -> List[Document]:

        documents = []
        raw_dir = DATA_PATHS["raw_documents"]

        if not raw_dir.exists():
            logger.warning(f"Document not found: {raw_dir}")
            return documents
        
        for file_path in raw_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.loaders:
                loaded_docs = self.load_single_document(file_path)
                if loaded_docs:
                    documents.extend(loaded_docs)
                    logger.info(f"Loaded: {file_path.name} ({len(loaded_docs)} documents)")

        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)

    def process_documents(self) -> List[Document]:
        logger.info("Loading documents... ")
        documents = self.load_all_documents()

        if not documents:
            logger.warning("Documents to process not found")
            return []
        
        logger.info(f"Loaded documents: {len(documents)}")
        logger.info("Dividing documents into chunks ... ")

        chunks = self.split_documents(documents)
        logger.info(f"Chunks created: {len(chunks)}")
        return chunks

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    processor = DocumentProcessor()
    chunks = processor.process_documents()

    if chunks:
        print(f"\n Summary: ")
        print(f"Total chunks: {len(chunks)}")
        print(f"First chunk: {chunks[0].page_content[:100]}...")
        print(f"Metadata: {chunks[0].metadata}")
