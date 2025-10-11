import logging
from typing import Dict, Any, List
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from config.settings import OLLAMA_CONFIG
from .knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

class SpecializedAgent:
    def __init__(self, specialization: str = "research_analyst"):
        self.specialization = specialization
        self.llm = Ollama(
            model=OLLAMA_CONFIG["models"]["llm"],
            base_url=OLLAMA_CONFIG["base_url"],
            timeout=OLLAMA_CONFIG["timeout"],
            temperature=0.3,
            num_threads=4
            )
        
        self.knowledge_base = KnowledgeBase()
        self.role_templates = self._get_role_templates()
        self.qa_chain = None


    def _get_role_templates(self) -> Dict[str, Any]:
        """"Define specific specialized roles"""
        return {
            "research_analyst": {
                "name": "Research Analyst",
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
                        "capabilities": [
                            "Documentary analysys",
                            "Information synthesis",
                            "Insights extraction",
                            "Pattern identification",
                            "Critical evaluation"
                        ]
                    },
                    
                    "project_organizer": {
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
                        "capabilities": [
                            "Information structuration",
                            "Project planning",
                            "Schedule creation",
                            "Task management",
                            "Prioritization"
                        ]
                    },
                    
                    "advanced_summarizer": {
                        "name": "Advanced Summarizer",
                        "prompt_template": """You are a specialist in summary and syntheis of long documents.
        
        DOCUMENTARY CONTEXT:
        {context}

        SUMMARY REQUEST:
        {request}

        SPECIFIC INSTRUCTIONS:
        -Create summaries that captures the main ideas.
        -Keep the original meanings of the text.
        -Adapt the details depending on the requests you receive.
        -Structure the text in the best logical way.
        -Give important conclusions and key points.

        SUMMARY: """,

                        "capabilities": [
                            "Executive summary",
                            "Key points extraction",
                            "Content synthesis",
                            "Detail level adaptation",
                            "Logical structuration"
                            ]
                        },

                        "code_analyzer": {
                            "name": "Code Analyzer",
                            "prompt_template": """You are a senior engineer specialized on code analysis and technical documentation.
                                        
        TECHNICAL CONTEXT:
        {context}
                                        
        ANALYSIS REQUEST:
        {question}
        SPECIFIC INSTRUCTIONS:
        -Analyze code, architecture or technical documentation
        -Identify patterns, good practices, clean code, possible issues
        -Add comments explaining technical concepts in a clear way
        -Suggest improving and optimizations 
        -Give practical examples when needed
        -Changes variables into English and according to the language
        -Give the reference of documentation when you suggest or change something on the code
                                        
        TECHNICAL ANALYSIS:""",
                            "capabilities": [
                                "Code Analysis",
                                "Technical Review",
                                "Technical Documentation",
                                "Optimizations",
                                "Concepts Explanations"
                                                                
                            ]
                        }
                    }

    def initialize(self) -> bool:

        """Initialize the specialized agent"""
        if self.knowledge_base.load_knowledge_base():
            retriever = self.knowledge_base.get_retriever()
            if retriever:
                role_config = self.role_templates.get(
                    self.specialization,
                    self.role_templates["research_analyst"]
                )

                prompt = PromptTemplate(
                    template = role_config["prompt_template"],
                    input_variables=["context", "question"]
                )

                self.qa_chain = RetrievalQA.from_chain_type(
                    llm = self.llm,
                    chain_type = "stuff",
                    retriever = retriever,
                    chain_type_kwargs = {"prompt": prompt},
                    return_source_documents = True
                )

                logger.info(f"{role_config['name']} initializing")
                return True
            
            logger.warning("Knowledge base not found")
            return False
        
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Make a request to the specialized assistant"""
        if not self.qa_chain:
            return {
                "answer": "Agent not iniatialized",
                "success": False
            }
        
        try: 
            result = self.qa_chain({"query": question})

            role_config = self.role_templates.get(self.specialization)


            sources = []
            for doc in result.get("source_documents", []):
                source_name = doc.metadata.get("source", "Unknown")
                sources.append({
                    "source": source_name,
                    "preview": doc.page_content[:150] + "..."
                })

            return {
                "answer": result["result"],
                "sources": sources,
                "role": role_config["name"],
                "capabilities": role_config["capabilities"],
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error: {e}")
            return {
                "answer": f"Error {str(e)}",
                "success": False 
            }
        
    def list_specializations(self) -> List[Dict[str, Any]]:
        """List all specializations availables"""
        return [
            {
                "id": spec_id,
                "name": config["name"],
                "capabilities": config["capabilities"]
            }
            for spec_id, config in self.role_templates.items()
        ]
    
    def change_specialization(self, new_specialization: str) -> bool:
        """Change the specialized agent"""
        if new_specialization in self.role_templates:
            self.specialization = new_specialization
            return self.initialize()
        return False
            

                

        