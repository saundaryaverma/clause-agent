"""clause-agent ask CONTRACT.txt "QUESTION"   self-correcting cited answer
clause-agent eval                           benchmark on CUAD (no API key needed)"""
import argparse
import json
import sys
from pathlib import Path

from clausewise import data as cuad

from clause_agent.evaluate import benchmark
from clause_agent.graph import ask, build_agent
from clause_agent.llms import ExtractiveLLM, FlakyLLM, openai_llm


def pick_llm(args):
    if getattr(args, "model", None):
        return openai_llm(args.model)
    return FlakyLLM() if getattr(args, "flaky", False) else ExtractiveLLM()


def run_ask(args):
    text = Path(args.contract).read_text(errors="ignore")
    result = ask(build_agent(pick_llm(args), max_attempts=args.max_attempts), text, args.question, args.definition)
    print("Agent steps:")
    for step in result["steps"]:
        print("  -", step)
    print()
    if result["status"] == "flagged":
        print("No verified answer. The draft below failed the citation check and should not be relied on:")
    print(result["draft"])
    print("\nSources:")
    for c in result["citations"]:
        print(f"[{c.id}] characters {c.start}-{c.end}: {' '.join(c.text.split())[:240]}...")
    return 0 if result["status"] != "flagged" else 1


def run_eval(args):
    folder = Path(args.data_path) if args.data_path else cuad.download(args.data_dir)
    contracts = cuad.load(folder / "test.json", args.limit)
    runs = [("Extractive model (no API key)", lambda: build_agent(ExtractiveLLM()))]
    if args.model:
        runs.append((f"{args.model}", lambda: build_agent(openai_llm(args.model))))
    runs += [("Faulty model, no repair loop", lambda: build_agent(FlakyLLM(), max_attempts=1)),
             ("Faulty model, with repair loop", lambda: build_agent(FlakyLLM(), max_attempts=3))]
    lines = ["| Setup | Verified answers | Grounded in lawyer highlight | Flagged | Avg attempts |",
             "| --- | --- | --- | --- | --- |"]
    results = {}
    for name, make in runs:
        r = benchmark(contracts, make())
        results[name] = r
        lines.append(f"| {name} | {r['verified']:.1%} | {r['grounded']:.1%} | {r['flagged']:.1%} | {r['avg_attempts']:.2f} |")
    table = "\n".join(lines)
    print(f"CUAD test: {len(contracts)} contracts, {r['questions']} clauses lawyers highlighted\n\n{table}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(table + "\n")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="clause-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("ask")
    a.add_argument("contract")
    a.add_argument("question")
    a.add_argument("--definition", default="", help="plain-English description of the clause, improves retrieval")
    a.add_argument("--model", help="OpenAI model via LangChain, e.g. gpt-4o-mini")
    a.add_argument("--flaky", action="store_true", help="inject a misquote to watch the repair loop")
    a.add_argument("--max-attempts", type=int, default=3)
    e = sub.add_parser("eval")
    e.add_argument("--data-dir", default="data")
    e.add_argument("--data-path")
    e.add_argument("--limit", type=int)
    e.add_argument("--model")
    e.add_argument("--out", default="results")
    args = parser.parse_args(argv)
    return run_ask(args) if args.command == "ask" else run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
