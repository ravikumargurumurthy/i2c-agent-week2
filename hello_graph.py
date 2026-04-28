# hello_graph.py
"""
Smallest possible LangGraph to verify end-to-end setup with Azure OpenAI.
Not the real agent — that's Day 1. This is a connectivity sanity check.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ---------- State ----------
class HelloState(TypedDict):
    """The data that flows through the graph."""
    user_input: str
    greeting: str
    response: str


# ---------- LLM client (reused across nodes) ----------
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)


# ---------- Nodes ----------
def greet_node(state: HelloState) -> dict:
    """Produces a greeting. Pure Python, no LLM."""
    name = state["user_input"]
    print(f"[greet_node] received user_input='{name}'")
    return {"greeting": f"Hello, {name}! I am a LangGraph node."}


def respond_node(state: HelloState) -> dict:
    """Calls Azure to produce a one-line reply to the greeting."""
    greeting = state["greeting"]
    print(f"[respond_node] greeting='{greeting}', calling LLM...")
    resp = llm.invoke(
        f"You received this greeting: '{greeting}'. "
        f"Reply in one short sentence."
    )
    return {"response": resp.content}


# ---------- Graph construction ----------
builder = StateGraph(HelloState)
builder.add_node("greet", greet_node)
builder.add_node("respond", respond_node)
builder.add_edge(START, "greet")
builder.add_edge("greet", "respond")
builder.add_edge("respond", END)

graph = builder.compile()


# ---------- Run it ----------
if __name__ == "__main__":
    initial_state = {"user_input": "Ravikumar", "greeting": "", "response": ""}
    final_state = graph.invoke(initial_state)
    print("\n" + "=" * 50)
    print("FINAL STATE:")
    print("=" * 50)
    for key, value in final_state.items():
        print(f"  {key}: {value}")
