# Clause Agent

A self-correcting LangGraph agent that answers questions about contracts, cites the exact passage, checks every quote against the source, and repairs its own answer when a check fails.

![CI](https://github.com/saundaryaverma/clause-agent/actions/workflows/ci.yml/badge.svg)

An LLM that misquotes a contract is worse than one that says nothing, because a confident wrong quote looks like evidence. Clause Agent never shows an answer it hasn't verified. It's built on [Clausewise](https://github.com/saundaryaverma/Clausewise), which benchmarks contract retrieval and provides the citation checker, and adds an agent loop around it.

## How the agent works

```
plan -> retrieve -> draft -> verify --passed--> done
           ^                   |
           +----- repair ------+  widen the search, feed back exactly what failed
                               |
                               +-- out of attempts --> flagged (never shown as an answer)
```

- **plan:** builds the search query. If a plain-English definition of the clause is available, it's added, because Clausewise showed that raises Recall@3 by 14%.
- **retrieve:** BM25 and TF-IDF with reciprocal rank fusion over 600-character chunks, keeping each chunk's character offsets for citation.
- **draft:** asks the model to quote exact contract language with a source id after each quote, or to say "Not found in this contract."
- **verify:** checks that every cited id exists, every quote appears in the passage it cites, and no claim is uncited.
- **repair:** on failure, widens the search by two passages and tells the model exactly which check failed.
- **flag:** after three failed attempts, the answer is marked unverified and the command exits with an error.

Each node is a LangGraph node, so with LangSmith tracing turned on (`LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`), every run shows up as a full trace of plan, retrieval, drafts, checks, and repairs.

## Watch it repair a misquote

`--flaky` swaps in a model that paraphrases "shall" as "will" on its first try, a small change that alters legal meaning:

```
$ clause-agent ask contract.txt "Governing Law" --definition "Which state's law governs the contract?" --flaky
Agent steps:
  - plan: search for 'Governing Law. Which state's law governs the contract?'
  - retrieve: top 3 passages
  - draft (attempt 1): "This Note will be governed by and construed in accordance with the laws of the State of N...
  - verify: failed: quote not found in [C1]: "This Note will be governed by and construed in accordance wi"
  - repair: widen search to 5 passages and retry
  - retrieve: top 5 passages
  - draft (attempt 2): "This Note shall be governed by and construed in accordance with the laws of the State of ...
  - verify: passed
  - done: verified
```

That's real output on a contract from the CUAD dataset.

## Results on CUAD

All 1,244 clauses that lawyers highlighted in the 102-contract CUAD test set. "Grounded" means the passage the agent cited overlaps the lawyers' highlight.

| Setup | Verified answers | Grounded in lawyer highlight | Flagged | Avg attempts |
| --- | --- | --- | --- | --- |
| Extractive model (no API key) | 84.6% | 37.2% | 2.4% | 1.05 |
| Faulty model, no repair loop | 7.0% | 3.6% | 80.0% | 1.00 |
| Faulty model, with repair loop | 84.6% | 37.3% | 2.4% | 1.83 |

What this shows:

- **The verifier blocks misquotes, and the repair loop recovers them.** When the model misquotes, the checker stops every bad answer. Without repair, 80% of questions end flagged. With repair, the agent recovers to the same verified rate as a model that never misquoted, in about 1.8 attempts on average.
- **The extractive baseline sets a floor.** It quotes the sentence that best matches the question, with no LLM. It cites the lawyers' passage 37% of the time. A real model (`--model gpt-4o-mini`) should do better, and `clause-agent eval --model gpt-4o-mini` adds it to the same table.
- **Verified isn't the same as correct.** An answer can quote the contract exactly and still quote the wrong clause. That's why the table reports grounding separately, and why grounding is the number to improve next.

## Run it

```bash
git clone https://github.com/saundaryaverma/clause-agent.git
cd clause-agent
pip install -e .
clause-agent eval                                  # downloads CUAD, benchmarks without an API key
clause-agent ask my_contract.txt "Termination for convenience" --flaky

pip install -e ".[openai]"                         # optional: a real model through LangChain
export OPENAI_API_KEY=sk-...
clause-agent ask my_contract.txt "Governing Law" --model gpt-4o-mini
```

## Tests

```bash
pip install -e ".[dev]"
pytest --cov=clause_agent
```

10 tests (94% coverage) cover every path through the graph: verified on the first try, misquote caught and repaired with feedback and a wider search, flagged after repeated failures, and "not found" as a valid answer. They also check that clause definitions reach the search query, that grounding compares cited passages against lawyer highlights, and that the benchmark and CLI work end to end on CUAD-format data. CI runs the tests on Python 3.10 and 3.12 and then the full CUAD benchmark.

## Next

- **Benchmark a real model** in the same table, and compare its grounding with the extractive baseline.
- **A smarter repair step:** when grounding fails, rewrite the search query instead of only widening it.
- **Human review:** send flagged answers to a reviewer using LangGraph's interrupt feature.

## Data and license

CUAD is from The Atticus Project, licensed CC BY 4.0, and is downloaded on first run. Code is MIT licensed.
