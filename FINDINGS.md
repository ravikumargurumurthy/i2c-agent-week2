# FINDINGS — Week 2

Engineering decisions, debugging notes, and lessons from refactoring
the I2C cash application agent into a LangGraph state machine.

---

## Day 0 — Setup
- New repo `i2c-agent-week2` for clean separation from Week 1.
- Reused unchanged: `schemas.py`, `tools.py`, `eval_data.py`, `data/`.
  This deliberate split is what makes the eval suite a regression-test
  for the refactor.
- Switched LLM client from raw `openai.AzureOpenAI` to LangChain's
  `AzureChatOpenAI` for better LangGraph integration.
- Sanity-checked end-to-end with `hello_graph.py` (3-node trivial graph).

## Day 1 — Agent rewritten as state machine

### Mental model shift
Week 1's `extract_remittance` was one function with a `for` loop and
embedded if/else. Week 2's is a graph: 3 nodes, 2 conditional edges.
Same logic, different organization.

### Key concepts internalized
- **State** as a Pydantic model with typed fields and reducers.
- **Reducer** (`Annotated[list, add]`) for fields that should append, not replace.
- **Static edges** for deterministic flow; **conditional edges** for
  decisions based on state.
- **Stateless nodes** — counters and run-scoped data must live in state,
  not in node-local variables.

### Demo on happy path
Same input as Week 1 produced equivalent output:
- payer_customer_id: CUST001 ✓
- total_amount: 4300.00 ✓
- 2 allocations with correct amounts ✓
- confidence in the >=0.95 band ✓

## Day 2 — Eval suite integration

### Goal: 10/10 unchanged
The Week 1 eval suite (`eval_data.py` + `test_agent.py`) was reused
verbatim against the new agent. The only coupling between harness and
implementation is the `extract_remittance` function name — which the
new `agent.py` preserves.

### Result: TODO — fill in actual pass rate

### TODO — document any regressions found and how they were resolved

## Day 3 (Routing demo) — Eval design lesson

First attempt at three routing-band demo cases:
- Auto-apply: clean two-invoice payment → routed correctly
- HITL: clean short-pay with deduction → routed to AUTO_APPLY (0.96)
- Exception: unknown payer → routed correctly

The HITL case wasn't actually ambiguous. The rubric is internally consistent:
clean short-pays where customer + invoice + math all check out should score
high regardless of the presence of a deduction. The demo case design was bad,
not the rubric.

Replaced with an under-allocated payment ($20,500 covering invoices totaling
$21,000) — a case where the agent should genuinely flag uncertainty about
where the unallocated $500 belongs.

Lesson: when designing demos for routing bands, work backwards from the
rubric. Pick inputs that *actually* trigger the rubric's branches, not
inputs that *seem* like they should.

This is the same eval-design discipline as Week 1 ev_004: the agent was
right; the test expectation was uncalibrated.
