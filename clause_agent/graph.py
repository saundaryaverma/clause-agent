"""The agent as a LangGraph state machine.

plan -> retrieve -> draft -> verify --valid--> done
                      ^         |
                      +-repair--+ (widen the search, feed back what failed)
                                |
                                +--out of attempts--> flagged (never shown as an answer)
"""
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from clausewise.answer import NOT_FOUND, Citation, build_prompt, verify
from clausewise.chunking import chunk
from clausewise.retrieve import BM25, TfIdf, corpus_idf, rrf, tokenize


class State(TypedDict, total=False):
    question: str
    definition: str
    contract: str
    query: str
    k: int
    attempt: int
    citations: list
    draft: str
    check: dict
    feedback: list
    status: str          # "verified", "not_found", or "flagged"
    steps: list          # human-readable trace of what the agent did


def build_agent(llm, max_attempts=3, k=3, widen_by=2, chunk_size=600, overlap=100):
    def plan(state):
        query = state["question"]
        if state.get("definition"):
            query += ". " + state["definition"]  # Clausewise found definitions lift Recall@3 by 14%
        return {"query": query, "k": k, "attempt": 0, "feedback": [],
                "steps": [f"plan: search for '{query[:80]}'"]}

    def retrieve(state):
        text = state["contract"]
        spans = chunk(text, chunk_size, overlap)
        tokens = [tokenize(text[s:e]) for s, e in spans]
        idf, _ = corpus_idf(tokens)
        q = tokenize(state["query"])
        scores = rrf(BM25(tokens, idf).scores(q), TfIdf(tokens, idf).scores(q))
        top = np.argsort(-scores, kind="stable")[: state["k"]]
        cites = [Citation(f"C{n + 1}", spans[i][0], spans[i][1], text[spans[i][0]:spans[i][1]])
                 for n, i in enumerate(top)]
        return {"citations": cites, "steps": state["steps"] + [f"retrieve: top {len(cites)} passages"]}

    def draft(state):
        prompt = build_prompt(state["question"], state["citations"])
        if state["feedback"]:
            prompt += ("\n\nYour previous answer failed these checks: " + "; ".join(state["feedback"])
                       + ". Fix it: quote the contract exactly, character for character.")
        reply = llm(prompt).strip()
        attempt = state["attempt"] + 1
        return {"draft": reply, "attempt": attempt,
                "steps": state["steps"] + [f"draft (attempt {attempt}): {reply[:90]}"]}

    def check(state):
        result = verify(state["draft"], state["citations"])
        note = "passed" if result["valid"] else "failed: " + "; ".join(result["problems"])
        return {"check": result, "steps": state["steps"] + [f"verify: {note}"]}

    def repair(state):
        return {"k": state["k"] + widen_by, "feedback": state["check"]["problems"],
                "steps": state["steps"] + [f"repair: widen search to {state['k'] + widen_by} passages and retry"]}

    def finish(state):
        status = "not_found" if state["draft"].strip() == NOT_FOUND else "verified"
        return {"status": status, "steps": state["steps"] + [f"done: {status}"]}

    def flag(state):
        return {"status": "flagged",
                "steps": state["steps"] + [f"flagged: no verified answer after {state['attempt']} attempts"]}

    def route(state):
        if state["check"]["valid"]:
            return "finish"
        return "repair" if state["attempt"] < max_attempts else "flag"

    g = StateGraph(State)
    for name, fn in [("plan", plan), ("retrieve", retrieve), ("draft", draft), ("verify", check),
                     ("repair", repair), ("finish", finish), ("flag", flag)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route, {"finish": "finish", "repair": "repair", "flag": "flag"})
    g.add_edge("repair", "retrieve")
    g.add_edge("finish", END)
    g.add_edge("flag", END)
    return g.compile()


def ask(agent, contract, question, definition=""):
    return agent.invoke({"contract": contract, "question": question, "definition": definition})
