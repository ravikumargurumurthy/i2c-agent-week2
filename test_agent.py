# test_agent.py
"""
Pytest harness for the remittance extraction agent.

Runs each case in EVAL_SET through extract_remittance() and asserts on
specific fields. Confidence is checked against a band, not exact equality.

Run with:
    pytest -v test_agent.py
    pytest -v test_agent.py -k "ev_006"     # run a single case
    pytest -v test_agent.py --tb=short      # less verbose tracebacks
"""

from decimal import Decimal
import pytest

from agent import extract_remittance
from eval_data import EVAL_SET


def _check_allocations(actual, expected):
    """
    Compare allocations as a set of (invoice_number, amount_paid) tuples.
    Optionally check deduction_amount and deduction_reason if present in expected.
    """
    actual_core = {(a.invoice_number, a.amount_paid) for a in actual}
    expected_core = {(e["invoice_number"], Decimal(e["amount_paid"])) for e in expected}
    assert actual_core == expected_core, (
        f"Allocations mismatch.\n  Actual:   {actual_core}\n  Expected: {expected_core}"
    )

    # Optional per-allocation checks
    actual_by_inv = {a.invoice_number: a for a in actual}
    for e in expected:
        if "deduction_amount" in e:
            actual_alloc = actual_by_inv[e["invoice_number"]]
            assert actual_alloc.deduction_amount == Decimal(e["deduction_amount"]), (
                f"deduction_amount mismatch for {e['invoice_number']}: "
                f"got {actual_alloc.deduction_amount}, expected {e['deduction_amount']}"
            )
        if "deduction_reason" in e:
            actual_alloc = actual_by_inv[e["invoice_number"]]
            actual_reason = (
                actual_alloc.deduction_reason.value
                if actual_alloc.deduction_reason else None
            )
            assert actual_reason == e["deduction_reason"], (
                f"deduction_reason mismatch for {e['invoice_number']}: "
                f"got {actual_reason}, expected {e['deduction_reason']}"
            )

@pytest.mark.parametrize("case", EVAL_SET, ids=[c["id"] for c in EVAL_SET])
def test_extraction(case):
    result = extract_remittance(case["input"])
    advice = result["advice"]                      # ← extract from dict
    expected = case["expected"]

    # 1. Customer resolution
    assert advice.payer_customer_id == expected["payer_customer_id"], (
        f"payer_customer_id mismatch: got {advice.payer_customer_id}, "
        f"expected {expected['payer_customer_id']}"
    )

    # 2. Total amount
    assert advice.total_amount == Decimal(expected["total_amount"]), (
        f"total_amount mismatch: got {advice.total_amount}, "
        f"expected {expected['total_amount']}"
    )

    # 3. Allocations
    _check_allocations(advice.allocations, expected["allocations"])

    # 4. Unallocated amount (if specified)
    if "unallocated_amount" in expected:
        assert advice.unallocated_amount == Decimal(expected["unallocated_amount"]), (
            f"unallocated_amount mismatch: got {advice.unallocated_amount}, "
            f"expected {expected['unallocated_amount']}"
        )

    # 5. Confidence band
    assert expected["min_confidence"] <= advice.confidence <= expected["max_confidence"], (
        f"confidence {advice.confidence} outside band "
        f"[{expected['min_confidence']}, {expected['max_confidence']}]"
    )

    # 6. Routing decision (NEW — only checked if specified in expected)
    if "routing_decision" in expected:
        assert result["routing_decision"] == expected["routing_decision"], (
            f"routing_decision mismatch: got {result['routing_decision']}, "
            f"expected {expected['routing_decision']}"
        )