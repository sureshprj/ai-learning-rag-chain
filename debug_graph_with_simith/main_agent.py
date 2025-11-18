### BASIC LANGRAPH with connected tools


from typing import Annotated
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# loads environment variables from .env
load_dotenv()  
llm = ChatGroq(model_name="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))


class State(TypedDict):
    messages: Annotated[list, add_messages]


# define tools
@tool(parse_docstring=True)
def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of a and b.
    """
    return a + b

@tool(parse_docstring=True)
def multiply(a: int, b: int) -> int:
    """
    multiply two integers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The multiply of a and b.
    """
    return a * b


tools = [add, multiply]
tool_node = ToolNode(tools=tools)
llm_with_tool = llm.bind_tools(tools)

# define node
def chat_node(state:State) -> State:
    return {"messages": llm_with_tool.invoke(state["messages"])}
    

def create_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("chat", chat_node)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "chat")
    graph_builder.add_conditional_edges("chat", tools_condition)
    graph_builder.add_edge("tools", "chat")
    graph = graph_builder.compile()
    return graph

chat_agent = create_graph()