import json

import pytest

CONTRACT = (
    "SUPPLY AGREEMENT. This agreement is made between Acme Corp and Beta LLC. "
    + "Filler text about deliveries, invoices, and packaging requirements. " * 10
    + "9. Governing Law. This agreement shall be governed by the laws of the State of New York. "
    + "More filler about notices and amendments to the schedules. " * 10
    + "12. Termination. Either party may terminate this agreement for convenience on ninety days notice."
)


def qa(category, description, answer_text):
    answers = [{"text": answer_text, "answer_start": CONTRACT.index(answer_text)}] if answer_text else []
    return {"question": f'Highlight the parts (if any) of this contract related to "{category}". Details: {description}',
            "answers": answers, "id": category}


@pytest.fixture
def cuad_dir(tmp_path):
    qas = [qa("Governing Law", "Which state's law governs the contract?",
              "This agreement shall be governed by the laws of the State of New York"),
           qa("Non-Compete", "Is there a restriction on competing?", None)]
    data = {"data": [{"title": f"C{i}", "paragraphs": [{"context": CONTRACT, "qas": qas}]} for i in range(2)]}
    (tmp_path / "test.json").write_text(json.dumps(data))
    return tmp_path
