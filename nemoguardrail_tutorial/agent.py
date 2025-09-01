# /agent.py
import os
from typing import Annotated
from typing_extensions import TypedDict

# Make sure to import SystemMessage!
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
import datetime
from dotenv import load_dotenv

load_dotenv()

# --- Agent Tools ---
@tool
def get_current_time() -> str:
    """Returns the current time in H:M:S format."""
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"--- [Tool Executed] get_current_time returned: {now} ---")
    return now

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# --- Agent Logic ---
def create_react_agent_graph():
    """Builds the LangGraph agent."""
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    tools = [get_current_time]
    llm_with_tools = llm.bind_tools(tools)

    def assistant_node(state: AgentState):
        """The core logic node that calls the LLM."""
        
        # 1. Define the system prompt to instruct the LLM.
        system_prompt = (
            "You are a helpful assistant that has access to tools. "
            "Use the `get_current_time` tool if the user asks for the current time."
        )
        
        # 2. Prepend the system prompt to the message history.
        messages_with_system_prompt = [SystemMessage(content=system_prompt)] + state["messages"]

        print(f"--- [Agent LLM Input] Sending to Gemini: {messages_with_system_prompt} ---")

        # 3. Invoke the LLM with the full context.
        response = llm_with_tools.invoke(messages_with_system_prompt)
        return {"messages": [response]}

    # Define the graph
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile the graph with memory
    memory = InMemorySaver()
    return builder.compile(checkpointer=memory)
  

