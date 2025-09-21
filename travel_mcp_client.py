import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp_use import MCPAgent, MCPClient

# Load environment variables
load_dotenv()

async def run_travel_chat():
    """Run a chat using MCPAgent with travel planner."""
    
    # Configure Google API key
    if "GOOGLE_API_KEY" not in os.environ:
        GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
        if GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        else:
            print("Error: GEMINI_API_KEY not found in .env file")
            return

    # Config file path
    config_file = "travel_config.json"

    print("Initializing travel chat...")

    try:
        # Create MCP client
        client = MCPClient.from_config_file(config_file)
        
        # Create ChatGoogleGenerativeAI LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # Create agent with memory
        agent = MCPAgent(
            llm=llm,
            client=client,
            max_steps=15,
            memory_enabled=True,
        )

        print("\n===== Travel Planner Chat =====")
        print("Type 'exit' or 'quit' to end the conversation")
        print("Type 'clear' to clear conversation history")
        print("================================\n")

        # Main chat loop
        while True:
            # Get user input
            user_input = input("\nYou: ")

            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation...")
                break

            # Check for clear history command
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue

            # Get response from agent
            print("\nAssistant: ", end="", flush=True)

            try:
                # Run the agent with the user input
                response = await agent.run(user_input)
                print(response)

            except Exception as e:
                print(f"\nError: {e}")

    except Exception as e:
        print(f"Failed to initialize: {e}")
        print("Make sure:")
        print("1. travel_config.json exists in current directory")
        print("2. travel_mcp_server.py path is correct in config")
        print("3. GEMINI_API_KEY is set in .env file")
    
    finally:
        # Clean up
        try:
            if 'client' in locals() and client and client.sessions:
                await client.close_all_sessions()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(run_travel_chat())