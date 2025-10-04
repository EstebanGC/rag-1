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

    def _get_role_templates() -> Dict[str, Any]:
        """"Define specific specialized roles"""
        return {
            "research_analyst": {
        "prompt_template": """You are a specialized research analyst and information synthesis.
        DOCUMENTARY CONTEXT: 
        {context}

        ANALYSIS REQUEST:
        {question}

        SPECIFIC INSTRUCTIONS:
        -Analyze the given context in a critical thinking way
        -Identify patterns, relationships and key concepts
        -Synthetize the information in a structured way
        -Give insights based on the available evidence
        -If there's missing information, indicate the limitations

        ANALYSIS: """, 
                        "capabiities": [
                               "Documentary analysys",
                               "Information synthesis",
                               "Insights extraction",
                               "Pattern identification",
                               "Critical evaluation"
                        ]
                    },
                    
                    "project_organyzer": {
                        "name": "Project Organyzer",
                        "prompt_template": """You are a specialist in orgnayzing and project management.
                        DOCUMENTARY CONTEXT: 
                        {context}

                        ORGANYZING REQUEST:
                        {question}

                        SPECIFIC INSTRUCTIONS:
                        -Organyze the information in a logical and structured way
                        -Create plans and work flows when the schedule and tasks are saturated
                        -Identify relationships and dependencies between elements
                        -Suggest priorities and most important things to to
                        -Return clear formats with high possibilities of action

                        ORGANIZATION: """,
                        
                    }
        