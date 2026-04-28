# agent.py
"""
LangGraph rewrite of the I2C remittance extraction agent.

Same behavior as the Week 1 hand-rolled version (same evals should pass),
but expressed as an explicit state machine: nodes for LLM calls, tool
execution, and validation; edges for control flow.
"""

import json
import os
from typing import Optional, Annotated
from operator import add
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from langgraph.graph import StateGraph, START, END

from schemas import RemittanceAdvice
from tools import lookup_customer, lookup_open_invoices, parse_amounts_and_invoices

load_dotenv()


# ---------- State ----------
class AgentState(BaseModel):
    """
    The state that flows through the graph. Every node reads from it and
    returns partial updates that the framework merges back in.
    """

    # --- Input ---
    remittance_text: str
    """The raw remittance the user wants extracted."""

    # --- Conversation ---
    messages: Annotated[list[BaseMessage], add] = Field(default_factory=list)
    """Running conversation history. The Annotated[..., add] tells LangGraph
    to APPEND new messages to this list rather than overwrite, so each node
    can return new messages without clobbering existing ones."""

    # --- Outputs ---
    advice: Optional[RemittanceAdvice] = None
    """Final validated extraction. Set by validate_output once successful."""

    validation_error: Optional[str] = None
    """If validation failed, the error to feed back to the LLM for repair."""

    validation_retries: int = 0
    """Counter for how many repair attempts we've made."""

# ---------- LLM client ----------
TOOL_REGISTRY = {
    "lookup_customer": lookup_customer,
    "lookup_open_invoices": lookup_open_invoices,
    "parse_amounts_and_invoices": parse_amounts_and_invoices,
}

# Tool schemas in the format LangChain expects (same JSON Schema as Week 1)
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "parse_amounts_and_invoices",
            "description": (
                "Deterministic regex extraction of money amounts and invoice-number-like "
                "tokens from raw text. ALWAYS call this first to ground extraction."
            ),
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": (
                "Resolve a payer name to a customer_id via fuzzy match. "
                "Strips generic corporate suffixes (Corp, Inc, Ltd, etc.) before scoring. "
                "Returns customer_id (or null), match_score, ambiguous flag, top 3 candidates. "
                "If customer_id is null OR ambiguous=true, you MUST set payer_customer_id "
                "to null. Do NOT pick from top_candidates even if one looks plausible."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name_query": {"type": "string"}},
                "required": ["name_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_open_invoices",
            "description": (
                "Get open invoices for a customer, optionally filtered to specific "
                "invoice numbers. Use to verify invoices mentioned in the remittance "
                "exist as open AR for the resolved customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "invoice_numbers": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["customer_id"],
            },
        },
    },
]

# Construct the LangChain Azure client and bind the tools to it.
# `bind_tools` is the LangChain way to attach tool schemas to a model;
# the model now knows about them on every invocation.
_llm_base = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)
llm_with_tools = _llm_base.bind_tools(TOOLS_SCHEMA)

# ---------- System prompt ----------
SYSTEM_PROMPT = """You are a cash application assistant. Your job is to extract a structured \
RemittanceAdvice from a remittance string.

Process to follow:
1. Call `parse_amounts_and_invoices` first to get regex-detected amounts and invoice numbers.
2. Call `lookup_customer` with the payer name to resolve customer_id.
3. Call `lookup_open_invoices` with that customer_id and the detected invoice numbers.
4. After tool calls, return a final JSON object matching the RemittanceAdvice schema.

Output schema (RemittanceAdvice):
{
  "payer_name": str,
  "payer_customer_id": str | null,
  "payment_reference": str | null,
  "payment_date": "YYYY-MM-DD" | null,
  "total_amount": str (decimal as string, e.g. "4250.00"),
  "allocations": [
    {
      "invoice_number": str,
      "amount_paid": str (decimal),
      "deduction_amount": str (decimal) | null,
      "deduction_reason": "pricing"|"shortage"|"damage"|"promo"|"unauthorized"|"unknown" | null,
      "notes": str | null
    }
  ],
  "unallocated_amount": str (decimal, default "0"),
  "confidence": float between 0.0 and 1.0,
  "extraction_notes": str | null
}

Confidence guidance:
- 0.95+ ONLY if customer resolved with high score (>=90) AND all invoices verified open AND amounts reconcile.
- 0.70-0.94 if there is any ambiguity (low fuzzy score, missing invoices, partial amount match).
- Below 0.70 if customer can't be resolved confidently or invoices don't match open AR.

Rules:
- amount_paid is the cash applied to the invoice. For short-pays, this is the FULL cash amount.
- deduction_amount is informational only. It does NOT reduce amount_paid.
- sum(amount_paid) + unallocated_amount must equal total_amount. Deductions do not appear in this sum.
- Always express amounts as decimal-compatible strings (e.g. "4250.00", not 4250.00).
- If the customer can't be resolved confidently, set payer_customer_id to null.
- If sum of allocations does not equal total_amount, set unallocated_amount to absorb the difference.
- Always include extraction_notes briefly explaining your reasoning.
- If you cannot produce a valid extraction, still emit a RemittanceAdvice with confidence < 0.5
  and extraction_notes explaining why.
"""

# ---------- Nodes ----------

def call_llm_node(state: AgentState) -> dict:
    """
    The LLM call node. Constructs the messages list (using state) and invokes
    the LLM. Returns the new assistant message to be appended to state.

    On the first call, prepends the system prompt and user message.
    On subsequent calls, just adds another LLM turn given the running history.
    """
    # If this is the first call, seed the conversation. Otherwise, use existing.
    if not state.messages:
        msgs = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Extract the remittance advice from this text:\n\n{state.remittance_text}"
            ),
        ]
    else:
        msgs = list(state.messages)

    # If a validation error is flagged, append it as a user message before re-invoking
    if state.validation_error:
        msgs.append(HumanMessage(
            content=(
                f"Your previous output failed validation:\n{state.validation_error}\n\n"
                f"Fix the issues and return corrected JSON matching the RemittanceAdvice schema."
            )
        ))

    response = llm_with_tools.invoke(msgs)

    # Determine which messages to append to state.
    # If first call: append system + user + response (full seeding).
    # Otherwise: append the validation feedback (if any) + response.
    new_messages = []
    if not state.messages:
        new_messages.extend([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Extract the remittance advice from this text:\n\n{state.remittance_text}"
            ),
        ])
    if state.validation_error:
        new_messages.append(HumanMessage(
            content=(
                f"Your previous output failed validation:\n{state.validation_error}\n\n"
                f"Fix the issues and return corrected JSON matching the RemittanceAdvice schema."
            )
        ))
    new_messages.append(response)

    # Clear validation_error since we've fed it back
    return {
        "messages": new_messages,
        "validation_error": None,
    }


def execute_tools_node(state: AgentState) -> dict:
    """
    Execute every tool call in the most recent assistant message and
    append the results as ToolMessage objects.
    """
    last_msg = state.messages[-1]
    if not getattr(last_msg, "tool_calls", None):
        # Defensive — this node shouldn't be entered if there are no tool calls
        return {}

    tool_messages = []
    for tc in last_msg.tool_calls:
        name = tc["name"]
        args = tc["args"]
        if name not in TOOL_REGISTRY:
            result = json.dumps({"error": f"Unknown tool: {name}"})
        else:
            try:
                fn = TOOL_REGISTRY[name]
                output = fn(**args)
                result = json.dumps(output, default=str)
            except Exception as e:
                result = json.dumps({"error": f"Tool {name} failed: {str(e)}"})

        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tc["id"], name=name)
        )

    return {"messages": tool_messages}


def validate_output_node(state: AgentState) -> dict:
    """
    Try to parse the most recent assistant message as a RemittanceAdvice.
    On success, set state.advice. On failure, set validation_error so the
    next call_llm cycle feeds it back to the model.
    """
    last_msg = state.messages[-1]
    content = last_msg.content if isinstance(last_msg, AIMessage) else None

    if not content:
        return {"validation_error": "Final assistant message had no content to parse."}

    try:
        advice = RemittanceAdvice.model_validate_json(content)
    except Exception as e:
        return {
            "validation_error": f"Schema validation failed: {e}",
            "validation_retries": state.validation_retries + 1,
        }

    business_errors = advice.validate_amounts()
    if business_errors:
        return {
            "validation_error": f"Business rule failed: {business_errors}",
            "validation_retries": state.validation_retries + 1,
        }

    return {"advice": advice}

# ---------- Conditional edge functions ----------

def after_llm(state: AgentState) -> str:
    """
    After the LLM responded, decide where to go next.
    - If it requested tools, execute them.
    - Otherwise, validate the final output.
    """
    last_msg = state.messages[-1]
    if getattr(last_msg, "tool_calls", None):
        return "execute_tools"
    return "validate_output"


def after_validation(state: AgentState) -> str:
    """
    After validation:
    - If advice is set, we're done.
    - If validation_error is set and we have retries left, loop back.
    - If retries exhausted, end anyway (return whatever we have).
    """
    if state.advice is not None:
        return END
    if state.validation_retries >= 2:
        # Out of repair retries — end the loop. Caller can inspect state.advice (None)
        # and state.validation_error.
        return END
    return "call_llm"

# ---------- Build the graph ----------

builder = StateGraph(AgentState)

builder.add_node("call_llm", call_llm_node)
builder.add_node("execute_tools", execute_tools_node)
builder.add_node("validate_output", validate_output_node)

builder.add_edge(START, "call_llm")

builder.add_conditional_edges(
    "call_llm",
    after_llm,
    {"execute_tools": "execute_tools", "validate_output": "validate_output"},
)

# After tool execution, always go back to the LLM
builder.add_edge("execute_tools", "call_llm")

# After validation, branch on success/retry/end
builder.add_conditional_edges(
    "validate_output",
    after_validation,
    {"call_llm": "call_llm", END: END},
)

# Compile with a recursion ceiling matching Week 1's max_iterations spirit
graph = builder.compile()


# ---------- Public API ----------

def extract_remittance(remittance_text: str) -> RemittanceAdvice:
    """Run the agent on a remittance string and return the validated extraction."""
    initial = AgentState(remittance_text=remittance_text)
    final_state = graph.invoke(initial, config={"recursion_limit": 25})

    # final_state is a dict (LangGraph returns dicts even from Pydantic state)
    if not final_state.get("advice"):
        raise RuntimeError(
            f"Agent did not produce valid output. "
            f"Validation error: {final_state.get('validation_error')}"
        )
    return final_state["advice"]


# ---------- Demo ----------

if __name__ == "__main__":
    sample = (
        "Payment $4,300.00 from Acme Corporation via wire ref WIRE-789 for "
        "INV-1001 ($2,500) and INV-1002 ($1,800)."
    )
    advice = extract_remittance(sample)
    print("FINAL VALIDATED OUTPUT:")
    print(advice.model_dump_json(indent=2))
