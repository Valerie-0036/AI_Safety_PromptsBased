#./guardrails_config/actions.py
from nemoguardrails.logging.verbose import set_verbose

from langchain_core.runnables import Runnable
import os
from nemoguardrails.actions.actions import action # Keep @action for clarity, though it might not be auto-discovered
from agent import create_react_agent_graph

# --- Initialize the LangGraph agent graph ONCE ---
react_graph = create_react_agent_graph()

global_thread_id_counter = 0

@action(name="langgraph_agent_action")
async def langgraph_agent_action(user_input: str) -> str:
    """
    This function is called by NeMo Guardrails.
    It runs the LangGraph agent and returns the final response.
    """
    global global_thread_id_counter
    set_verbose(True)

    try:
        print(f"\n--- [Guardrails] Handing off to LangGraph Agent for input: '{user_input}' ---")

        global_thread_id_counter += 1
        config_for_langgraph = {"configurable": {"thread_id": f"thread-{global_thread_id_counter}"}}
        
        initial_input = {"messages": [("user", user_input)]}

        result = await react_graph.ainvoke(initial_input, config=config_for_langgraph)

        if "messages" in result and result["messages"]:
            final_response = result["messages"][-1].content
            print(f"--- [LangGraph Agent] Final response: '{final_response}' ---")
            return final_response
        else:
            print("--- [LangGraph Agent] Returned result had no messages or empty messages list. ---")
            return "The agent did not return a valid response."

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"--- [Error] Agent execution failed: {e}\n{error_trace} ---")
        return "Sorry, I encountered an issue processing your request."
# This is your desired Runnable class
class LangGraphAgentRunnable(Runnable):
    def __init__(self):
        self.thread_id_counter = 0

    async def invoke(self, user_input: str, config=None, **kwargs) -> str:
        # --- NEW DEBUG PRINT: Verify entry into the method ---
        print(f"\n--- DEBUG: Entering LangGraphAgentRunnable.invoke with input: '{user_input}' ---")
        
        try:
            print(f"\n--- [Guardrails] Handing off to LangGraph Agent for input: '{user_input}' ---")

            self.thread_id_counter += 1
            langgraph_config = config.copy() if config else {}
            langgraph_config.setdefault("configurable", {})["thread_id"] = f"thread-{self.thread_id_counter}"
            
            initial_input = {"messages": [("user", user_input)]}

            # --- NEW DEBUG PRINT: Before ainvoke call ---
            print(f"--- DEBUG: Calling react_graph.ainvoke with config: {langgraph_config} and input: {initial_input} ---")

            result = await react_graph.ainvoke(initial_input, config=langgraph_config)
            
            # --- NEW DEBUG PRINT: After ainvoke call ---
            print(f"--- DEBUG: react_graph.ainvoke returned result: {result} ---")

            # Check if result["messages"] exists and has at least one message
            if "messages" in result and result["messages"]:
                final_response = result["messages"][-1].content
                print(f"--- [LangGraph Agent] Final response: '{final_response}' ---")
                return final_response
            else:
                # --- NEW DEBUG PRINT: Empty messages from agent ---
                print("--- [LangGraph Agent] Returned result had no messages or empty messages list. ---")
                return "The agent did not return a valid response."

        except Exception as e:
            # --- IMPROVED ERROR HANDLING ---
            import traceback
            error_trace = traceback.format_exc()
            print(f"--- [Error] Agent execution failed: {e}\n{error_trace} ---")
            return "Sorry, I encountered an issue processing your request."