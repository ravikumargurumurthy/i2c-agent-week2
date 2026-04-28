# tools.py
import json
import re
from decimal import Decimal
from pathlib import Path
from rapidfuzz import fuzz

# Generic corporate suffixes that don't help disambiguate company names.
# Stripped before fuzzy matching so they don't inflate scores.
GENERIC_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "llc", "ltd", "limited",
    "co", "company", "plc", "lp", "llp", "gmbh", "ag", "sa", "nv",
    "pvt", "private", "pty", "holdings", "group", "international",
}

DATA_DIR = Path(__file__).parent / "data"


def _load(filename: str):
    """Internal helper — load a JSON file from the data directory."""
    return json.loads((DATA_DIR / filename).read_text())


# ---------- Tool 3: Open invoice lookup ----------
def lookup_open_invoices(customer_id: str, invoice_numbers: list[str] | None = None) -> list[dict]:
    """
    Return open invoices for a customer, optionally filtered by specific invoice numbers.

    Args:
        customer_id: e.g. "CUST001"
        invoice_numbers: optional list to filter, e.g. ["INV-1001", "INV-1002"]

    Returns:
        List of invoice dicts: [{invoice_number, customer_id, amount, issue_date, due_date}, ...]
    """
    invoices = _load("open_invoices.json")
    result = [i for i in invoices if i["customer_id"] == customer_id]
    if invoice_numbers:
        wanted = set(invoice_numbers)
        result = [i for i in result if i["invoice_number"] in wanted]
    return result

# ---------- Tool 1: Regex parser ----------
def parse_amounts_and_invoices(text: str) -> dict:
    """
    Deterministic regex extraction of money amounts and invoice-number-like tokens.
    The LLM uses this output as evidence to ground its extraction.

    Args:
        text: raw remittance text

    Returns:
        {
            "amounts_detected": ["2500.00", "50.00"],
            "invoice_numbers_detected": ["INV-1024", "INV-1025"]
        }
    """
    # Money pattern: optional $, optional space, then digits with optional commas + optional .XX
    money_pattern = r"\$?\s?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|\d+\.\d{2})"
    amounts = [m.replace(",", "") for m in re.findall(money_pattern, text)]

    # Invoice number pattern: INV-1234, INV1234, or 5+ digit numbers
    inv_pattern = r"\b(?:INV[-_]?\d{3,}|\d{5,})\b"
    invoices = re.findall(inv_pattern, text, re.IGNORECASE)

    # Dedupe while preserving order
    invoices = list(dict.fromkeys(invoices))

    return {
        "amounts_detected": amounts,
        "invoice_numbers_detected": invoices,
    }

# ---------- Tool 2: Customer lookup with fuzzy matching ----------
def _strip_suffixes(name: str) -> str:
    """Remove generic corporate suffixes from a name for fairer fuzzy matching."""
    tokens = [t for t in name.lower().replace(",", " ").replace(".", " ").split() if t]
    meaningful = [t for t in tokens if t not in GENERIC_SUFFIXES]
    # If stripping leaves nothing, fall back to original (avoid empty match keys)
    return " ".join(meaningful) if meaningful else name.lower()

def lookup_customer(name_query: str) -> dict:
    """
    Resolve a payer name string to a customer_id using fuzzy matching against
    legal name and aliases in the customer master. Strips generic corporate
    suffixes before scoring and detects ambiguity between top candidates.
    """
    customers = _load("customers.json")
    stripped_query = _strip_suffixes(name_query)
    candidates = []

    for c in customers:
        names_to_check = [c["legal_name"]] + c.get("aliases", [])
        stripped_names = [_strip_suffixes(n) for n in names_to_check]
        best_score = max(fuzz.WRatio(stripped_query, n) for n in stripped_names)
        candidates.append({
            "customer_id": c["customer_id"],
            "legal_name": c["legal_name"],
            "score": best_score,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else {"score": 0}

    # ----- NEW: exact-match override -----
    # If the query exactly matches one of the top customer's names or aliases
    # (case-insensitive, whitespace-trimmed), bypass the gap-based ambiguity check.
    # Exact match means the payer specified the customer fully — no ambiguity possible.
    top_customer = next(c for c in customers if c["customer_id"] == top["customer_id"])
    exact_match_names = [top_customer["legal_name"]] + top_customer.get("aliases", [])
    exact_match = name_query.strip().lower() in [n.strip().lower() for n in exact_match_names]
    # -------------------------------------

    confident = top["score"] >= 80
    ambiguous = (top["score"] - runner_up["score"]) <= 15 and not exact_match

    customer_id = top["customer_id"] if (confident and not ambiguous) else None
    legal_name = top["legal_name"] if (confident and not ambiguous) else None

    return {
        "customer_id": customer_id,
        "legal_name": legal_name,
        "match_score": top["score"],
        "ambiguous": ambiguous,
        "top_candidates": candidates[:3],
    }