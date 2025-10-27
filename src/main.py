import sys
import logging 
import argparse
from pathlib import Path

src_path = Path(__file__).parent
project_root = src_path.parent
sys.path.append(str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s -%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)  

from config.settings import OLLAMA_CONFIG, DATA_PATHS
from document_processor import DocumentProcessor
from knowledge_base import KnowledgeBase
from dual_agent import DualAgent

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

def ask_question(question: str, agent: str = "auto"):
    agent_handler = DualAgent()

    if agent_handler.initialize():
        print("🔄 Processing your question... (this may take 10-30 seconds)")
        result = agent_handler.ask_question(question, agent)

        print(f"🔍 Debug - Result keys: {list(result.keys())}")
        print(f"🔍 Debug - Success: {result.get('success')}")

        if result.get('success', False):
            print(f"\n{'='*50}")
            print(f"{result['agent']} ({result['model']}):")
            print(f"{'='*50}")
            print(f"{result['answer']}")
            print(f"{'='*50}")

            if result.get('sources'):
                print(f"\nSources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):  
                    print(f"{i}. {source['source']}")
                    print(f"   Preview: {source['preview']}\n")
            else:
                print("\n💡 Note: Response from model (no documents used)")
        else:
            print(f"\n❌ Error: {result.get('answer', 'Unknown error')}")
            
    else: 
        print("First build the knowledge base with: python src/main.py --build")

def interactive_chat():
    agent_handler = DualAgent()
    agents = agent_handler.list_agents()

    print("\n Available Agents: ")
    for i, agent in enumerate(agents, 1):
        print(f"{i}. {agent['name']}")
        print(f" Model: {agent['model']}")
        print(f" Skills: {', '.join(agent['capabilities'][:3])}...\n")  

    try: 
        choice = int(input("\nSelect agent (1-2) or Enter for auto-detection: ") or "0") 
        if choice == 1:
            selected_agent = "coder"
        elif choice == 2:  
            selected_agent = "assistant"
        else:
            selected_agent = "auto"
    except: 
        selected_agent = "auto"

    if not agent_handler.initialize():  
        print("First build knowledge base with: python src/main.py --build")
        return 
    
    agent_info = agent_handler.agents.get(selected_agent, agent_handler.agents['assistant'])
    print(f"\n {agent_info['name']} ready!")
    print(f" I can help with: {', '.join(agent_info['capabilities'][:3])}")
    print("\n" + "=" * 50)

    while True:
        user_input = input("\n Your question (or 'exit'): ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'salir']:
            print("Goodbye! ")
            break

        if user_input:
            print("Thinking...")
            result = agent_handler.ask_question(user_input, selected_agent)

            if result["success"]:
                print(f"\n Answer ({result['model']}):")
                print(result["answer"])

                if result["sources"]:
                    print(f"\n Sources consulted: {len(result['sources'])}")
            else:
                print(f"Error: {result['answer']}")

def main():
    parser = argparse.ArgumentParser(description = "Dual Agent RAG System")
    parser.add_argument("--build", action="store_true", help="Build knowledge base")
    parser.add_argument("--ask", "-a", help="Ask a specific question")
    parser.add_argument("--agent", "-g", default="auto",  
                        help="Agent: coder (Mistral 7B), assistant (Genma 2B), auto")
    parser.add_argument("--chat", "-c", action="store_true", help="Interactive chat mode")

    args = parser.parse_args()

    logger.info("Dual Agent RAG System - Starting ...")
    logger.info(f"coder: {OLLAMA_CONFIG['models']['coder']} (Mistral 7B)")
    logger.info(f"assistant: {OLLAMA_CONFIG['models']['assistant']} (Genma 2B)")  
    logger.info(f"Documents: {DATA_PATHS['raw_documents']}")

    if args.build:
        build_knowledge_base()
    elif args.ask:
        ask_question(args.ask, args.agent)
    elif args.chat:
        interactive_chat()
    else:
        print("Usage: python src/main.py [OPTIONS]")
        print("\n Examples:")
        print(" python src/main.py --build")
        print(" python src/main.py --ask \"Write a Python function\" --agent coder")  
        print(" python src/main.py --ask \"Organize this document\" --agent assistant") 
        print(" python src/main.py --chat")

if __name__ == "__main__":
    main()