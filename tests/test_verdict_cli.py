"""Front R's options of the `bench` command (design §6.3): `judge j7 --teacher-self-replay`."""
import pytest

from bench import cli
from fixtures.fake import repo

J7 = ["judge", "j7", "--centroids", "c.json", "--adapters", "a.json", "--cheap-alt", "replay-cheap=eval-cheap",
      "--slm", "replay-b4=eval-b4", "--teacher-eval", "eval-t", "--j8", "j8.json", "--j6", "j6.json"]


def test_judge_j7_takes_the_teachers_self_replay(tmp_path, monkeypatch, capsys):
    from bench.judge import j7
    config_path, _ = repo(tmp_path, monkeypatch)
    seen = {}
    monkeypatch.setattr(j7, "run", lambda *args, **kwargs: seen.update(args=args, kwargs=kwargs) or ("result", "fact"))
    assert cli.main(J7 + ["--teacher-self-replay", "replay-self", "--config", str(config_path)]) == 0
    assert seen["kwargs"] == {"teacher_self_replay": "replay-self"}
    assert seen["args"][2] == {"cheap_alt": ("replay-cheap", "eval-cheap"), "slm": ("replay-b4", "eval-b4")}
    assert capsys.readouterr().out.split() == ["result", "fact"]


def test_judge_j7_without_the_self_replay_is_refused(tmp_path, monkeypatch, capsys):
    config_path, _ = repo(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as refused:
        cli.main(J7 + ["--config", str(config_path)])
    assert refused.value.code == 2 and "--teacher-self-replay" in capsys.readouterr().err
