"""Benchmark the agent on CUAD clauses that lawyers highlighted.

grounded: the agent's cited passage overlaps the text the lawyers highlighted
verified: every quote in the answer passed the citation check"""
import re

from clausewise.chunking import overlaps

from clause_agent.graph import ask

_CITED = re.compile(r"\[(C\d+)\]")


def grounded(result, gold_spans):
    by_id = {c.id: (c.start, c.end) for c in result.get("citations", [])}
    cited = [by_id[c] for c in _CITED.findall(result.get("draft", "")) if c in by_id]
    return any(overlaps(span, gold_spans) for span in cited)


def benchmark(contracts, agent):
    rows = []
    for contract in contracts:
        for q in contract.questions:
            if not q.spans:
                continue
            r = ask(agent, contract.text, q.category, q.description)
            rows.append({"status": r["status"], "attempts": r["attempt"],
                         "grounded": r["status"] == "verified" and grounded(r, q.spans)})
    n = len(rows)
    return {
        "questions": n,
        "verified": sum(r["status"] == "verified" for r in rows) / n,
        "grounded": sum(r["grounded"] for r in rows) / n,
        "flagged": sum(r["status"] == "flagged" for r in rows) / n,
        "avg_attempts": sum(r["attempts"] for r in rows) / n,
    }
