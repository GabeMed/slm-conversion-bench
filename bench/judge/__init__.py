"""Judgments (design §1.2): pure functions over facts. They read what executions wrote and call no
model, GPU or database, so re-running one is free and gives the same answer. They run in
env/analysis and never import CHESS; most need only the standard library, J5 needs numpy and
scikit-learn (pinned in env/analysis/requirements.txt)."""


class JudgeError(ValueError):
    pass
