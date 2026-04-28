# inspect_trace.py
"""Quick CLI to read a trace file in human format."""

import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_trace.py <run_id_or_path>")
        sys.exit(1)

    arg = sys.argv[1]
    path = Path(arg) if Path(arg).exists() else Path("traces") / f"{arg}.jsonl"
    if not path.exists():
        print(f"Trace not found: {path}")
        sys.exit(1)

    print(f"=== Trace: {path} ===\n")
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        ev = rec["event"]
        if ev == "run_start":
            print(f"▶ Run started: {rec['run_id']}")
        elif ev == "node_start":
            print(f"  → {rec['node']:20} input: {rec['input_summary']}")
        elif ev == "node_end":
            print(f"  ✓ {rec['node']:20} {rec['duration_ms']:>7}ms  output: {rec['output_summary']}")
        elif ev == "node_error":
            print(f"  ✗ {rec['node']:20} ERROR: {rec['error']}")
        elif ev == "run_end":
            print(f"\n■ Run ended: {rec['status']}")
            for k, v in rec.items():
                if k not in ("event", "run_id", "timestamp", "status"):
                    print(f"    {k}: {v}")

if __name__ == "__main__":
    main()
