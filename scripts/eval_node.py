#!/usr/bin/env python
"""
Node-level evaluation harness.

Usage:
  python scripts/eval_node.py parse_query "Build me an aggressive Krenko deck"
  python scripts/eval_node.py plan_deck --input fixtures/parsed_dragon.json
  python scripts/eval_node.py worker --slot ramp --context fixtures/dragon_context.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)


def run_parse_query(query: str):
    from mtg.graph.nodes.parse_query import parse_query
    state = {"user_query": query}
    result = parse_query(state)
    parsed = result["parsed"]
    print(json.dumps(parsed.model_dump(), indent=2))
    _save_output("parse_query", parsed.model_dump())
    return result


def run_plan_deck(input_path: str | None, query: str | None):
    if input_path:
        data = json.loads(Path(input_path).read_text())
        from mtg.schemas import ParsedQuery
        parsed = ParsedQuery(**data)
    elif query:
        r = run_parse_query(query)
        parsed = r["parsed"]
    else:
        print("Provide --input <path> or a query string")
        sys.exit(1)

    from mtg.graph.nodes.plan_deck import plan_deck
    state = {"parsed": parsed}
    result = plan_deck(state)
    plan = result["plan"]
    print(json.dumps(plan.model_dump(), indent=2))
    _save_output("plan_deck", plan.model_dump())
    return result


def run_worker(slot_name: str, context_path: str | None):
    if context_path:
        ctx = json.loads(Path(context_path).read_text())
        from mtg.schemas import ParsedQuery
        parsed = ParsedQuery(**ctx)
    else:
        # Use a default dragon context
        from mtg.schemas import ParsedQuery
        parsed = ParsedQuery(
            format="commander",
            commander="The Ur-Dragon",
            colors=["W", "U", "B", "R", "G"],
            theme="dragon tribal",
            style="tribal",
        )

    from mtg.graph.templates import get_template
    from mtg.schemas import Slot

    template = get_template(parsed.style, parsed.format)
    wave_map = template["waves"]
    ratios = template["ratios"]

    # Find which wave this slot is in
    slot_wave = 1
    for wave_num, slots in wave_map.items():
        if slot_name in slots:
            slot_wave = wave_num
            break

    slot = Slot(
        role=slot_name,
        count=ratios.get(slot_name, 10),
        hint=f"Cards for the {slot_name} role",
        wave=slot_wave,
    )

    from mtg.graph.nodes.worker import worker
    state = {"slot": slot, "deck_context": {"parsed": parsed}}
    result = worker(state)
    outputs = [o.model_dump() for o in result["worker_outputs"]]
    print(json.dumps(outputs, indent=2))
    _save_output(f"worker_{slot_name}", outputs)
    return result


def _save_output(node: str, data):
    ts = int(time.time())
    path = OUTPUTS_DIR / f"{node}_run_{ts}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"\nSaved to {path}")


def main():
    parser = argparse.ArgumentParser(description="MTG node eval harness")
    parser.add_argument("node", choices=["parse_query", "plan_deck", "worker"])
    parser.add_argument("query", nargs="?", help="Query string for parse_query/plan_deck")
    parser.add_argument("--input", "-i", help="Path to JSON fixture file")
    parser.add_argument("--slot", "-s", default="ramp", help="Slot name for worker node")
    parser.add_argument("--context", "-c", help="Path to parsed context JSON for worker")
    args = parser.parse_args()

    if args.node == "parse_query":
        if not args.query:
            print("Provide a query string")
            sys.exit(1)
        run_parse_query(args.query)

    elif args.node == "plan_deck":
        run_plan_deck(args.input, args.query)

    elif args.node == "worker":
        run_worker(args.slot, args.context or args.input)


if __name__ == "__main__":
    main()
