import json

from clause_agent.cli import main
from tests.conftest import CONTRACT


def test_ask_prints_steps_answer_and_sources(tmp_path, capsys):
    path = tmp_path / "contract.txt"
    path.write_text(CONTRACT)
    assert main(["ask", str(path), "Governing Law", "--flaky"]) == 0
    out = capsys.readouterr().out
    assert "repair: widen search" in out and "verify: passed" in out and "[C1] characters" in out


def test_eval_writes_results(cuad_dir, tmp_path, capsys):
    out = tmp_path / "results"
    assert main(["eval", "--data-path", str(cuad_dir), "--out", str(out)]) == 0
    assert "with repair loop" in capsys.readouterr().out
    assert len(json.loads((out / "results.json").read_text())) == 3
