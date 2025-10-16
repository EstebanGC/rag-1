import logging 
from typing import Dict, Any, List
from langchain.llms import ollama
from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa

from config.settings import OLLAMA_CONFIG
from .knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

class MistralAgent:
    def __init__(self):
        self.llm = Ollama(
            model="mistral",
            base_url=OLLAMA_CONFIG["base_url"],
            timeout=OLLAMA_CONFIG["timeout"],
            temperature=0.1,
            num_threads=6,
            num_gpu=1                           
        )

        self.knowledge_base = KnowledgeBase()
        self.qa_chain = None

        self.prompt_template = """You are a specialist focused on code, optimization and practical questions.
        TECNICAL CONTEXT:
        {context}

        TECHNICAL REQUEST:
        {question}

        INSTRUCTIONS:
        -
        """



