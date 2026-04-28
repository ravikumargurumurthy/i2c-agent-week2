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
from enum import Enum

from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from langgraph.graph import StateGraph, START, END

from schemas import RemittanceAdvice
from tools import lookup_customer, lookup_open_invoices, parse_amounts_and_invoices

from tracing import start_run, end_run, trace_node

load_dotenv()

class RoutingDecision(str, Enum):
    AUTO_APPLY = "auto_apply"
    HITL_REVIEW = "hitl_review"
    EXCEPTION = "exception"

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

    # --- Routing outputs ---
    routing_decision: Optional[RoutingDecision] = None
    """One of: 'auto_apply', 'hitl_review', 'exception'. Set by route_by_confidence."""

    action_result: Optional[dict] = None
    """The structured outcome of the terminal node — what was done, where it went,
    what audit info was recorded. Mock for now; in production this would be the
    return value of writing to ledger / enqueuing / escalating."""

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
@trace_node
def call_llm_node(state: AgentState) -> dict:
    """
    Build the messages list for the LLM, invoke, and return what's new.
    """
    # Compute what new context messages to add THIS turn
    new_context: list[BaseMessage] = []

    if not state.messages:
        # First turn — seed the conversation
        new_context.extend([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Extract the remittance advice from this text:\n\n{state.remittance_text}"
            ),
        ])

    if state.validation_error:
        # Repair turn — feed the error back
        new_context.append(HumanMessage(
            content=(
                f"Your previous output failed validation:\n{state.validation_error}\n\n"
                f"Fix the issues and return corrected JSON matching the RemittanceAdvice schema."
            )
        ))

    # Full message list = existing state + new context
    full_messages = list(state.messages) + new_context

    response = llm_with_tools.invoke(full_messages)

    # Return only what's new — framework appends via the `add` reducer
    return {
        "messages": new_context + [response],
        "validation_error": None,
    }

@trace_node
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

@trace_node
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

@trace_node
def route_by_confidence_node(state: AgentState) -> dict:
    """
    Decide where to route based on confidence band.
    This node only WRITES the decision to state. The actual dispatch
    happens in the conditional edge that comes after.
    """
    if not state.advice:
        # Defensive — shouldn't happen because validate_output gates this
        return {"routing_decision": RoutingDecision.EXCEPTION}

    confidence = state.advice.confidence

    if confidence >= 0.95:
        decision = RoutingDecision.AUTO_APPLY
    elif confidence >= 0.70:
        decision = RoutingDecision.HITL_REVIEW
    else:
        decision = RoutingDecision.EXCEPTION

    return {"routing_decision": decision}

@trace_node
def auto_apply_node(state: AgentState) -> dict:
    """
    Mock auto-apply: in production this would write to the GL subledger,
    update invoice statuses, post the cash, and emit an audit record.
    For Week 2 we just record what would have happened.
    """
    advice = state.advice
    result = {
        "action": "auto_apply",
        "ledger_entries": [
            {
                "invoice_number": a.invoice_number,
                "amount_applied": str(a.amount_paid),
                "customer_id": advice.payer_customer_id,
                "status": "would_post_to_GL",
            }
            for a in advice.allocations
        ],
        "unallocated": str(advice.unallocated_amount),
        "confidence": advice.confidence,
        "audit_note": "Auto-applied without human review; confidence >= 0.95",
    }
    return {"action_result": result}

@trace_node
def hitl_review_node(state: AgentState) -> dict:
    """
    Mock HITL queue: in production this would enqueue the advice into the
    analyst review tool, send a notification, and set an SLA timer.
    """
    advice = state.advice
    result = {
        "action": "hitl_review",
        "review_queue_entry": {
            "remittance_id": "would_be_a_uuid",
            "payer_customer_id": advice.payer_customer_id,
            "total_amount": str(advice.total_amount),
            "allocations": [
                {"invoice_number": a.invoice_number, "amount": str(a.amount_paid)}
                for a in advice.allocations
            ],
            "confidence": advice.confidence,
            "agent_notes": advice.extraction_notes,
            "queue_priority": "normal",
        },
        "audit_note": "Routed to HITL queue; confidence 0.70-0.94",
    }
    return {"action_result": result}

@trace_node
def exception_node(state: AgentState) -> dict:
    """
    Mock exception escalation: in production this would notify the ops team,
    create a ticket, and route the unmatched payment to an investigation queue.
    """
    advice = state.advice
    result = {
        "action": "exception",
        "exception_record": {
            "reason": "low_confidence_extraction",
            "payer_name_raw": advice.payer_name,
            "payer_customer_id": advice.payer_customer_id,
            "total_amount": str(advice.total_amount),
            "extraction_notes": advice.extraction_notes,
            "confidence": advice.confidence,
            "escalation_level": "ops_team",
        },
        "audit_note": "Escalated as exception; confidence < 0.70",
    }
    return {"action_result": result}

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
    if state.advice is not None:
        return "route_by_confidence"   # ← changed
    if state.validation_retries >= 2:
        return END
    return "call_llm"

def route_to_terminal(state: AgentState) -> str:
    """Dispatch from route_by_confidence_node to the appropriate terminal node."""
    decision = state.routing_decision
    if decision == RoutingDecision.AUTO_APPLY:
        return "auto_apply"
    elif decision == RoutingDecision.HITL_REVIEW:
        return "hitl_review"
    else:
        return "exception"

# ---------- Build the graph ----------

builder = StateGraph(AgentState)

# Existing nodes
builder.add_node("call_llm", call_llm_node)
builder.add_node("execute_tools", execute_tools_node)
builder.add_node("validate_output", validate_output_node)

# New nodes
builder.add_node("route_by_confidence", route_by_confidence_node)
builder.add_node("auto_apply", auto_apply_node)
builder.add_node("hitl_review", hitl_review_node)
builder.add_node("exception", exception_node)

# Existing edges (unchanged)
builder.add_edge(START, "call_llm")

builder.add_conditional_edges(
    "call_llm",
    after_llm,
    {"execute_tools": "execute_tools", "validate_output": "validate_output"},
)
builder.add_edge("execute_tools", "call_llm")

# Updated: validate_output now routes to route_by_confidence on success
builder.add_conditional_edges(
    "validate_output",
    after_validation,
    {
        "route_by_confidence": "route_by_confidence",
        "call_llm": "call_llm",
        END: END,
    },
)

# New: dispatch from router to one of three terminals
builder.add_conditional_edges(
    "route_by_confidence",
    route_to_terminal,
    {
        "auto_apply": "auto_apply",
        "hitl_review": "hitl_review",
        "exception": "exception",
    },
)

# Each terminal goes to END
builder.add_edge("auto_apply", END)
builder.add_edge("hitl_review", END)
builder.add_edge("exception", END)

graph = builder.compile()


# ---------- Public API ----------

def extract_remittance(remittance_text: str) -> dict:
    """
    Run the full agent workflow. Emits a trace to traces/{run_id}.jsonl.
    """
    run_id = start_run()
    initial = AgentState(remittance_text=remittance_text)

    try:
        final_state = graph.invoke(initial, config={"recursion_limit": 25})
    except Exception as e:
        end_run(status="error", error=str(e))
        raise

    if not final_state.get("advice"):
        end_run(status="no_advice", validation_error=final_state.get("validation_error"))
        raise RuntimeError(
            f"Agent did not produce valid output. "
            f"Validation error: {final_state.get('validation_error')}"
        )

    routing = final_state.get("routing_decision")
    end_run(
        status="ok",
        confidence=final_state["advice"].confidence,
        routing_decision=str(routing) if routing else None,
    )

    return {
        "advice": final_state["advice"],
        "routing_decision": routing,
        "action_result": final_state.get("action_result"),
        "run_id": run_id,
    }


# ---------- Demo ----------

if __name__ == "__main__":
    test_cases = [
        (
            "Auto-apply test",
            "Payment $4,300.00 from Acme Corporation via wire ref WIRE-789 for "
            "INV-1001 ($2,500) and INV-1002 ($1,800).",
        ),
        (
            "HITL test (under-allocated payment)",
            "Payment from Hooli $20,500.00 for INV-6001 and INV-6002",
        ),
        (
            "Exception test",
            "Payment $5,000.00 from Random Corp for INV-9999",
        ),
    ]

    for label, sample in test_cases:
        print("=" * 70)
        print(f"  {label}")
        print("=" * 70)
        result = extract_remittance(sample)
        print(f"Routing decision: {result['routing_decision']}")
        print(f"Confidence:       {result['advice'].confidence}")
        print(f"Action result:    {result['action_result']['action']}")
        print()
