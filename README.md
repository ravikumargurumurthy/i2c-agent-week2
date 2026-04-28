# I2C Cash Application Agent — Week 2 (LangGraph)

LangGraph rewrite of the [Week 1 hand-rolled agent](https://github.com/<your-username>/i2c-agent-week1). Same eval suite, same domain, new framework. Adds confidence-based routing nodes (auto-apply / HITL / exception) that make the agent operate inside a workflow rather than just return a value.

## Status
🚧 Day 0 — LangGraph setup complete, hello-world graph runs end-to-end.

## Why this exists
Week 1 proved I could build an agent. Week 2 proves I can refactor an agent without breaking it (same evals, target 10/10) AND extend it with the workflow scaffolding production cash app actually needs (routing, observability, audit-friendly state).

## Quick start

```bash
git clone https://github.com/<your-username>/i2c-agent-week2.git
cd i2c-agent-week2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in Azure credentials

python hello_graph.py     # sanity check
```

## Roadmap
- [x] Day 0: LangGraph setup, hello-world graph
- [ ] Day 1: state schema + 3-node graph (call_llm, execute_tools, validate)
- [ ] Day 2: validation-repair as conditional edge
- [ ] Day 3: confidence-based routing (auto / HITL / exception)
- [ ] Day 4: observability + tracing
- [ ] Day 5: README + Week 2 retro
