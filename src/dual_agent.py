import logging 
from typing import Dict, Any, List
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa

from config.settings import OLLAMA_CONFIG
from .knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

class DualAgent:
    def __init__(self):
        self.coder_llm = Ollama(
            model=OLLAMA_CONFIG["models"]["coder"],
            base_url=OLLAMA_CONFIG["base_url"],
            timeout=OLLAMA_CONFIG["timeout"],
            temperature=0.1,
            num_threads=6,
            num_gpu=1                           
        )

        self.assistant_llm = Ollama(
            model=OLLAMA_CONFIG["models"]["assistat"],
            base_url=OLLAMA_CONFIG["base_url"],
            timeout=OLLAMA_CONFIG["timeout"],
            temperature=0.3,
            num_threads=4,
            num_gpu=1
        )

        self.knowledge_base = KnowledgeBase()
        self.agents = self._setup_agents()
        self.qa_chain = {}

    



