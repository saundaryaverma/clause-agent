from clausewise.answer import NOT_FOUND

from clause_agent.evaluate import benchmark, grounded
from clause_agent.graph import ask, build_agent
from clause_agent.llms import ExtractiveLLM, FlakyLLM, parse_prompt
from clausewise.data import load
from tests.conftest import CONTRACT

Q, D = "Governing Law", "Which state's law governs the contract?"


class Recorder:
    """Wraps a model and records every prompt it receives."""
    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.inner(prompt)


def test_extractive_model_quotes_a_verifiable_sentence_with_its_source():
    prompt = ('Question: governing law\n\nSources:\n[C1] (characters 0-50)\nNotices go by mail.\n\n'
              '[C2] (characters 50-120)\nThis agreement shall be governed by the laws of New York.')
    question, sources = parse_prompt(prompt)
    assert question == "governing law" and [c for c, _ in sources] == ["C1", "C2"]
    assert ExtractiveLLM()(prompt) == '"This agreement shall be governed by the laws of New York." [C2]'


def test_clean_answer_is_verified_on_the_first_attempt():
    out = ask(build_agent(ExtractiveLLM()), CONTRACT, Q, D)
    assert out["status"] == "verified" and out["attempt"] == 1
    assert "State of New York" in out["draft"]


def test_definition_is_added_to_the_search_query():
    out = ask(build_agent(ExtractiveLLM()), CONTRACT, Q, D)
    assert out["query"] == f"{Q}. {D}"


def test_misquote_is_caught_then_repaired_with_feedback_and_a_wider_search():
    model = Recorder(FlakyLLM())
    out = ask(build_agent(model), CONTRACT, Q, D)
    assert out["status"] == "verified" and out["attempt"] == 2
    assert "failed these checks" not in model.prompts[0]
    assert "failed these checks" in model.prompts[1] and "quote not found" in model.prompts[1]
    assert out["k"] == 5  # search widened from 3 passages to 5


def test_answers_that_never_pass_are_flagged_not_shown():
    always_wrong = lambda prompt: '"governed by the laws of Delaware" [C1]'
    out = ask(build_agent(always_wrong, max_attempts=3), CONTRACT, Q, D)
    assert out["status"] == "flagged" and out["attempt"] == 3
    assert any(s.startswith("flagged") for s in out["steps"])


def test_not_found_is_a_valid_outcome():
    out = ask(build_agent(lambda p: NOT_FOUND), CONTRACT, "Non-Compete", "Is there a restriction on competing?")
    assert out["status"] == "not_found" and out["attempt"] == 1


def test_grounding_checks_the_cited_passage_against_the_highlight():
    out = ask(build_agent(ExtractiveLLM()), CONTRACT, Q, D)
    start = CONTRACT.index("This agreement shall be governed")
    assert grounded(out, [(start, start + 40)])
    assert not grounded(out, [(0, 10)])


def test_benchmark_on_cuad_format(cuad_dir):
    contracts = load(cuad_dir / "test.json")
    clean = benchmark(contracts, build_agent(ExtractiveLLM()))
    assert clean["questions"] == 2 and clean["verified"] == 1.0 and clean["grounded"] == 1.0
    no_repair = benchmark(contracts, build_agent(FlakyLLM(), max_attempts=1))
    assert no_repair["verified"] == 0.0 and no_repair["flagged"] == 1.0
    with_repair = benchmark(contracts, build_agent(FlakyLLM(), max_attempts=3))
    assert with_repair["verified"] == 1.0 and with_repair["avg_attempts"] == 2.0
