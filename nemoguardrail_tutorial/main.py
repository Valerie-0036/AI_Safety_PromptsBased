# main.py (with debug prints)
import asyncio
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.logging.verbose import set_verbose
from guardrails_config.actions import LangGraphAgentRunnable, langgraph_agent_action
from dotenv import load_dotenv
load_dotenv()

async def main():
    config = RailsConfig.from_path("./guardrails_config")



    app = LLMRails(config)
    set_verbose(True)
    # langgraph_agent_instance = LangGraphAgentRunnable()
    # app.register_action(langgraph_agent_action, "langgraph_agent_action")
    
    print("\n--- DEBUG: Actions immediately after explicit registration ---")
    registered_keys = list(app.runtime.action_dispatcher.registered_actions.keys())
    print(f"Total actions registered: {len(registered_keys)}")
    print(f"Is 'langgraph_agent_action' present? {'langgraph_agent_action' in registered_keys}")
    print("Registered actions:", registered_keys)
    print("----------------------------------------------------------------\n")
    
    print("Chat with your Guarded React Agent! (type 'exit' to quit)")
    while True:
        user_message = input("You: ")
        if user_message.lower() == "exit":
            break
            
        try:
            chat = [{"role":"user","content":user_message}]
            bot_response = await app.generate_async(messages=chat)
            print(f"Bot: {bot_response['content']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())