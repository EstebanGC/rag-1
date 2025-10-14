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
        return False
    
    knowledge_base = KnowledgeBase()
    if knowledge_base.create_knowledge_base(documents):
        if knowledge_base.save_knowledge_base():
            logger.info("Knowledge base built successfully!")
            return True
    
    logger.error("Error handling knowledge base")
    return False 

def ask_question(question: str, specialization: str = "research_analyst"):
    agent = SpecializedAgent(specialization)
    if agent.initialize():
        result = agent.ask_question(question)

        print(f"\n {result['role']} Response: ")
        print(f"{result['answer']}")

        if result['sources']:
            print(f"\n Sources ({len(result['sources'])}):")
            for i, source, in enumarate(result['sources'], 1):
                print(f"{i}. {source['source']}")
                print(f" Preview: {source['preview']}\n")

        else: print("First build the knowledge base with: python src/main.py --build")

def interactive_chat():
    agent = SpecializedAgent()
    specializations = agent.list_specializations()

    print("\n Available Specializations: ")
    for i, spec in enumerate(specializations, 1):
        print(f"{i}, {spec['name']} - {', '.join(spec['capabilities'][:2])}...")

        try: 
            choice = int(input("\nSelect specialization (1-4): ")) - 1
            if 0 <= choice < len(specializations):
                selected_spec = specializations[choice]["id"]
            else: 
                selected_spec = "research_analyst"
        except:
            selected_spec = "research_analyst"

        agent = SpecializedAgent(selected_spec)
        if not agent.initialize():
            print("First build knowledge base with: python src/main.py -- build")
            return 
        
        role_info = agent.role_templates[selected_spec]
        print(f"\n {role_info['name']} ready!")
        print(f"💡 I can help with: {', '.join(role_info['capabilities'])}")
        print("\n" + "=" * 50)

        while True:
            user_input = input("\n Your question (or 'exit'): ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'salir']:
                print("Goodbye! ")
                break

            if user_input:
                print("Thinking...")
                result = agent.ask_question(user_input)

                if result["success"]:
                    print(f"\n Answer:")
                    print(result["answer"])

                    if result["sources"]:
                        print(f"\n Sources consulted: {len(result['sources'])}")

                    else:
                        print(f"Error: {result['answer']}")

        


