import logging 
from typing import Dict, Any, List
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from config.settings import OLLAMA_CONFIG
from knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

class DualAgent:
    def __init__(self):
        self.coder_llm = Ollama(
            model=OLLAMA_CONFIG["models"]["coder"],
            base_url=OLLAMA_CONFIG["base_url"],
            timeout=OLLAMA_CONFIG["timeout"],
            temperature=0.1,
            num_predict=512,  
            num_gpu=1
            system="You are a senior software engineer specialized in development."
        )

        self.assistant_llm = Ollama(
            model=OLLAMA_CONFIG["models"]["assistant"],
            base_url=OLLAMA_CONFIG["base_url"],
            timeout=OLLAMA_CONFIG["timeout"],
            temperature=0.3,
            num_predict=256, 
            system="You are an intelligent and organized personal assistant."
        )

        self.knowledge_base = KnowledgeBase()
        self.agents = self._setup_agents()
        self.qa_chains = {}

    def _setup_agents(self) -> Dict[str, Any]:
        """Configure both agents with their specific models"""
        return {
            "coder": {
                "name": "💻 Code Assistant (Mistral 7B)",
                "model": self.coder_llm,
                "prompt_template": """You are a senior software engineer specialized in development. Use the context to help you.

TECHNICAL CONTEXT:
{context}

CODE REQUEST:
{question}

SPECIFIC INSTRUCTIONS:
- Analyze and write clean, efficient, and well-documented code
- Explain the logic and algorithms behind the solutions
- Suggest best practices, design patterns, and optimizations
- Provide testable examples and use cases
- If complex, break down into understandable steps
- Variables in English, comments in Spanish
- Include error handling when appropriate

TECHNICAL RESPONSE:""",
                "capabilities": [
                    "Full-stack development",
                    "Debugging and profiling", 
                    "Software architecture",
                    "Professional code review",
                    "Technical documentation",
                    "Algorithm optimization"
                ],
                "keywords": [
                    "code", "program", "function", "class", "method", "algorithm",
                    "python", "javascript", "java", "html", "css", "sql", "api",
                    "debug", "error", "variable", "import", "export", "compile",
                    "syntax", "framework", "library", "git", "docker", "database"
                ]
            },
            
            "assistant": {
                "name": "📊 General Assistant (Gemma 2B)", 
                "model": self.assistant_llm,
                "prompt_template": """You are an intelligent and organized personal assistant. Use the context to provide accurate responses.

DOCUMENT CONTEXT:
{context}

REQUEST:
{question}

SPECIFIC INSTRUCTIONS:
- Organize information in a clear, structured, and actionable way
- Create professional formats, templates, and documents
- Suggest efficient workflows and best practices
- Help with business and personal communication
- Provide concise but complete responses
- Use lists, tables, and bullet points to improve clarity
- Maintain a professional but friendly tone

ORGANIZED RESPONSE:""",
                "capabilities": [
                    "Document management",
                    "Professional communication", 
                    "Project organization",
                    "Task automation",
                    "Templates and formats",
                    "Document analysis",
                    "Executive summaries"
                ],
                "keywords": [
                    "document", "organize", "email", "letter", "report", "summary",
                    "meeting", "agenda", "format", "template", "table", "chart",
                    "plan", "strategy", "communication", "presentation", "budget",
                    "task", "project", "calendar", "reminder", "analysis"
                ]
            }
        }
    
    def initialize(self) -> bool:
        """Initialize both agents with the knowledge base"""
        if self.knowledge_base.load_knowledge_base():
            retriever = self.knowledge_base.get_retriever()
            if retriever:
                # Create QA chains for EACH agent
                for agent_id, config in self.agents.items():
                    prompt = PromptTemplate(
                        template=config["prompt_template"],
                        input_variables=["context", "question"]
                    )
                    
                    self.qa_chains[agent_id] = RetrievalQA.from_chain_type(
                        llm=config["model"],  
                        chain_type="stuff",
                        retriever=retriever,
                        chain_type_kwargs={"prompt": prompt},
                        return_source_documents=True
                    )
                
                logger.info("Dual Agent initialized - Mistral 7B + Gemma 2B")
                return True
        
        logger.warning("Knowledge base not found - using direct model responses")
        return True  
    
    def ask_question(self, question: str, agent: str = "auto") -> Dict[str, Any]:
        """Answer using the appropriate agent"""
        #If there's no QA chains, use model direct answer
        if not self.qa_chains:
            return self._direct_model_response(question, agent)
        
        # Automatic agent detection
        if agent == "auto":
            agent = self._detect_agent(question)
        
        if agent not in self.qa_chains:
            return {
                "answer": f"Invalid agent: {agent}",
                "success": False
            }
        
        try:
            result = self.qa_chains[agent]({"query": question})
            agent_config = self.agents[agent]
            
            # Format sources
            sources = []
            for doc in result.get("source_documents", []):
                source_name = doc.metadata.get("source", "Unknown")
                sources.append({
                    "source": source_name,
                    "preview": doc.page_content[:200] + "..."
                })
            
            return {
                "answer": result["result"],
                "sources": sources,
                "agent": agent_config["name"],
                "model": "Mistral 7B" if agent == "coder" else "Gemma 2B",
                "capabilities": agent_config["capabilities"],
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in {agent}: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "success": False
            }

    def _direct_model_response(self, question: str, agent: str) -> Dict[str, Any]:
        if agent == "auto":
            agent = self._detect_agent(question)
        
        if agent == "coder":
            llm = self.coder_llm
            agent_name = self.agents["coder"]["name"]
            model_name = "Mistral 7B"
        else:
            llm = self.assistant_llm
            agent_name = self.agents["assistant"]["name"]
            model_name = "Gemma 2B"
        
        try:
            logger.info(f"Starting model response with {model_name}...")
            enhanced_prompt = f"""You are {agent_name}. Answer the following question clearly and helpfully:

QUESTION: {question}

Please provide a comprehensive and useful response:"""
            
            logger.info(f"📤 Sending request to Ollama...")
            response = llm.invoke(enhanced_prompt)
            logger.info(f"✅ Received response from Ollama, length: {len(response)}")
            return {
                "success": True,
                "agent": agent_name,
                "model": model_name,
                "answer": response,
                "sources": [],
                "note": "Response from model (no documents in knowledge base)"
            }
        except Exception as e:
            logger.error(f"Error in direct model response: {e}")
            return {
                "success": False,
                "answer": f"Error: {str(e)}",
                "agent": "Error",
                "model": "N/A",
                "sources": []
            }
    
    def _detect_agent(self, question: str) -> str:
        """Automatically detect which agent to use based on keywords"""
        question_lower = question.lower()
        
        # Count keywords for each agent
        coder_score = sum(1 for keyword in self.agents["coder"]["keywords"] 
                         if keyword in question_lower)
        assistant_score = sum(1 for keyword in self.agents["assistant"]["keywords"] 
                             if keyword in question_lower)
        
        # Decide based on scores
        if coder_score > assistant_score:
            return "coder"
        elif assistant_score > coder_score:
            return "assistant" 
        else:
            # Tie or no keywords - use assistant by default
            return "assistant"
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List available agents"""
        return [
            {
                "id": agent_id,
                "name": config["name"],
                "model": "Mistral 7B" if agent_id == "coder" else "Gemma 2B",
                "capabilities": config["capabilities"]
            }
            for agent_id, config in self.agents.items()
        ]