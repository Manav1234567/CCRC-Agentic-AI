import sys
from rag_engine import ClimateRAG
import traceback

def main():
    print("⏳ Initializing Climate Agent...")
    # We pass NOTHING so it defaults to using the Env Var (MILVUS_URI)
    agent = ClimateRAG()
    print("✅ Agent Ready! (Type 'quit' to exit)")
    
    # Simple Loop
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        try:
            response = agent.ask(user_input)
            print(f"\n🤖 Agent: {response}")
        except Exception as e:
            print(f"\n❌ ERROR DETAIL:") 
            traceback.print_exc()

if __name__ == "__main__":
    main()