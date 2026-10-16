"""Language models the agent can use. Each is a function: prompt -> reply text.

ExtractiveLLM needs no API key, so the agent, tests, and benchmark run anywhere.
FlakyLLM injects a realistic fault (a misquote on the first try) to exercise the repair loop.
openai_llm uses a real model through LangChain."""
import re

from clausewise.answer import NOT_FOUND
from clausewise.retrieve import tokenize

_SOURCE = re.compile(r"\[(C\d+)\] \(characters \d+-\d+\)\n(.*?)(?=\n\n\[C\d+\] \(characters|\Z)", re.S)
_QUESTION = re.compile(r"Question: (.*?)\n")
_SENTENCE = re.compile(r"[^.;:\n]{20,}[.;:]?")


def stems(text):
    """Crude stemming so 'governing' matches 'governed' and 'law' matches 'laws'."""
    return {w.rstrip("s")[:6] for w in tokenize(text)}


def parse_prompt(prompt):
    question = _QUESTION.search(prompt).group(1)
    sources = [(cid, text.strip()) for cid, text in _SOURCE.findall(prompt)]
    return question, sources


class ExtractiveLLM:
    """Quotes the sentence that best matches the question, with its source id. A
    deterministic stand-in for a model, useful as a baseline and for offline runs."""

    def __call__(self, prompt):
        question, sources = parse_prompt(prompt)
        q = stems(question)
        best, best_overlap = None, 0
        for cid, text in sources:
            for sentence in _SENTENCE.findall(text):
                overlap = len(q & stems(sentence))
                if overlap > best_overlap:
                    best, best_overlap = (sentence.strip(), cid), overlap
        if best is None:
            return NOT_FOUND
        sentence, cid = best
        return f'"{sentence}" [{cid}]'


class FlakyLLM:
    """Misquotes on the first attempt (changes 'shall' to 'will', a common paraphrase),
    then answers correctly once it sees the verifier's feedback."""

    def __init__(self):
        self.inner = ExtractiveLLM()
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        reply = self.inner(prompt)
        if "failed these checks" not in prompt and reply != NOT_FOUND:
            return re.sub(r"\bshall\b", "will", reply, count=1) if " shall " in reply else reply.replace('"', '"Notably, ', 1)
        return reply


def openai_llm(model="gpt-4o-mini"):
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(model=model, temperature=0)
    return lambda prompt: chat.invoke(prompt).content
