import logging
from typing import Dict, Any, List
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from config.settings import OLLAMA_CONFIG
from .knowledge_base import KnowledgeBasee

logger = logging.getLogger(__name__)

class SpecializedAgent:
    def __init__(self, specialization: str = "research_analyst"):
        self.specialization = specialization
        self.llm = Ollama(
            model = OLLAMA_CONFIG["models"]["llm"],
            base_url = OLLAMA_CONFIG["base_url"]
            timeout = OLLAMA_CONFIG["timeout"]
            temperature = 0.3,
            num_threads = 4
        )

        self.knowledge_base = Knowledge_base()
        self.role_templates = self._get_role_templates()
        self.qa_chain = None