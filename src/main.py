import sys
import logging 
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s -%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
src_path = Path(__file__).parent
sys.path.append(str(src_path))

from config.settings import OLLAMA_CONFIG, DATA_PATHS
from document_processor import DocumentProcessor
from knowledge_base import KnowledgeBase
from specialized_agent import SpecializedAgent

def knowledge_base():
    logger.info("Building knowledge base ...")

    processor = DocumentProcessor()
    documents = processor.process_documents()

    if not documents: 
        logger.error("No documents found to process")
        print("Place your files in data/raw/")
        return false




