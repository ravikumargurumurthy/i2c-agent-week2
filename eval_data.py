# eval_data.py
"""
Hand-labeled remittance extraction eval set.

Each case has an `expected` dict with the fields the agent MUST produce correctly.
Fields not listed are not checked. `min_confidence`/`max_confidence` bound the
LLM's confidence score.

Cases that intentionally encode known bugs from Day 3 findings:
  - ev_006_ambiguous_customer  → expects null customer_id (will fail today)
  - ev_010_unknown_payer       → expects null customer_id (will fail today)
"""

EVAL_SET = [
    # ---------- Happy path baseline ----------
    {
        "id": "ev_001_clean_single_invoice",
        "input": "Wire $2,500.00 from Acme Corporation re INV-1001",
        "expected": {
            "payer_customer_id": "CUST001",
            "total_amount": "2500.00",
            "allocations": [
                {"invoice_number": "INV-1001", "amount_paid": "2500.00"},
            ],
            "min_confidence": 0.90,
            "max_confidence": 1.0,
        },
    },

    # ---------- Multi-allocation ----------
    {
        "id": "ev_002_clean_two_invoices",
        "input": "Payment $4,300.00 from Acme Corporation for INV-1001 ($2,500) and INV-1002 ($1,800)",
        "expected": {
            "payer_customer_id": "CUST001",
            "total_amount": "4300.00",
            "allocations": [
                {"invoice_number": "INV-1001", "amount_paid": "2500.00"},
                {"invoice_number": "INV-1002", "amount_paid": "1800.00"},
            ],
            "min_confidence": 0.90,
            "max_confidence": 1.0,
        },
    },

    # ---------- Short pay with explicit reason ----------
    {
        "id": "ev_003_short_pay_with_reason",
        "input": (
            "Payment $2,450.00 from Acme Corporation for INV-1001. "
            "Short pay $50 due to damaged units in shipment."
        ),
        "expected": {
            "payer_customer_id": "CUST001",
            "total_amount": "2450.00",
            "allocations": [
                {
                    "invoice_number": "INV-1001",
                    "amount_paid": "2450.00",
                    "deduction_amount": "50.00",
                    "deduction_reason": "damage",
                },
            ],
            "min_confidence": 0.85,
            "max_confidence": 1.0,
        },
    },

    # ---------- Short pay, no reason given ----------
    {
        "id": "ev_004_short_pay_no_reason",
        "input": "Payment $1,750.00 from Acme Corporation re INV-1002. $50 deduction.",
        "expected": {
            "payer_customer_id": "CUST001",
            "total_amount": "1750.00",
            "allocations": [
                {
                    "invoice_number": "INV-1002",
                    "amount_paid": "1750.00",
                    "deduction_amount": "50.00",
                    "deduction_reason": "unknown",
                },
            ],
            "min_confidence": 0.75,
            "max_confidence": 1.0,
        },
    },

    # ---------- Alias resolution ----------
    {
        "id": "ev_005_alias_match",
        "input": "Wire from Globex $12,500.00 for INV-2001",
        "expected": {
            "payer_customer_id": "CUST003",
            "total_amount": "12500.00",
            "allocations": [
                {"invoice_number": "INV-2001", "amount_paid": "12500.00"},
            ],
            "min_confidence": 0.85,
            "max_confidence": 1.0,
        },
    },

    # ---------- Day 3 finding #1: ambiguous customer ----------
    {
        "id": "ev_006_ambiguous_customer",
        "input": "Payment $950.00 from Acme for INV-1004",
        "expected": {
            # "Acme" alone matches both CUST001 and CUST002 — should NOT confidently resolve.
            # INV-1004 actually belongs to CUST002, so an honest agent would either
            # flag ambiguity (customer_id=null) or correctly resolve via the invoice.
            # Strict reading: ambiguity should win → null.
            "payer_customer_id": None,
            "total_amount": "950.00",
            "allocations": [
                {"invoice_number": "INV-1004", "amount_paid": "950.00"},
            ],
            "min_confidence": 0.0,
            "max_confidence": 0.75,  # forbid auto-apply
        },
    },

    # ---------- Invoice not in open AR ----------
    {
        "id": "ev_007_invoice_not_open",
        "input": "Payment $3,000.00 from Acme Corporation for INV-9999",
        "expected": {
            "payer_customer_id": "CUST001",
            "total_amount": "3000.00",
            "allocations": [
                {"invoice_number": "INV-9999", "amount_paid": "3000.00"},
            ],
            "min_confidence": 0.0,
            "max_confidence": 0.75,  # invoice doesn't exist in AR — must flag
        },
    },

    # ---------- Reconciliation: under-allocated ----------
    {
        "id": "ev_008_under_allocated",
        "input": (
            "Payment $5,000.00 from Acme Corporation. Apply $2,500 to INV-1001 "
            "and $1,800 to INV-1002. Remainder is on-account."
        ),
        "expected": {
            "payer_customer_id": "CUST001",
            "total_amount": "5000.00",
            "allocations": [
                {"invoice_number": "INV-1001", "amount_paid": "2500.00"},
                {"invoice_number": "INV-1002", "amount_paid": "1800.00"},
            ],
            "unallocated_amount": "700.00",  # 5000 - 4300
            "min_confidence": 0.80,
            "max_confidence": 1.0,
        },
    },

    # ---------- Reconciliation: over-pay edge case ----------
    {
        "id": "ev_009_over_payment",
        "input": (
            "Wire $3,500.00 from Acme Corporation re INV-1001. "
            "Invoice was $2,500; customer overpaid by $1,000 in error."
        ),
        "expected": {
            "payer_customer_id": "CUST001",
            "total_amount": "3500.00",
            "allocations": [
                {"invoice_number": "INV-1001", "amount_paid": "2500.00"},
            ],
            "unallocated_amount": "1000.00",
            "min_confidence": 0.75,
            "max_confidence": 1.0,
        },
    },

    # ---------- Day 3 finding #2: unknown payer ----------
    {
        "id": "ev_010_unknown_payer",
        "input": "Payment $5,000.00 from Random Corp for INV-9999",
        "expected": {
            # "Random Corp" doesn't match any real customer — must NOT resolve.
            "payer_customer_id": None,
            "total_amount": "5000.00",
            "allocations": [
                {"invoice_number": "INV-9999", "amount_paid": "5000.00"},
            ],
            "min_confidence": 0.0,
            "max_confidence": 0.60,
        },
    },
]