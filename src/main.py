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

def build_knowledge_base():
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
            for i, source, in enumerate(result['sources'], 1):
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

def main():
    parser = argparse.ArgumentParser(description = "RAG University System")
    parser.add_argument("--build", action="store_true", help="Build knowledge base")
    parser.add_argument("--ask", "a", help="Ask a specific question")
    parser.add_argument("--specialization", "-s", default="research_analyst",
                        help="Specialization: resarch_analyst , project_organizer, " \
                        "advanced_summarizer, code_analyzer")
    parser.add_argument("--chat", "-c", action="store_true", help="Interactive chat mode")

    args = parser.parse_args()

    logger.info("RAG University System - Starting ...")
    logger.info(f"LLM Model: {OLLAMA_CONFIG['models']['llm']}")
    logger.info(f"Documents: {DATA_PATHS['raw_documents']}")

    if args.build:
        build_knowledge_base()
    elif args.ask:
        ask_question(args.ask, args.specialization)
    elif args.chat:
        interactive_chat()
    else:
        print("Usage: python src/main.py [OPTIONS]")
        print("\n Examples:")
        print(" python src/main.py --build")
        print(" python src/main.py --ask \"Explain derivaties\" --specialization resarch_analyst")
        print( " python src/main.py --chat")

    if __name__ == "__main":
        main()
              




