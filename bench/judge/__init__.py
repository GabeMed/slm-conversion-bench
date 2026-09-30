"""Judgments (design §1.2): pure functions over facts. They read what executions wrote and call no
model, GPU or database, so re-running one is free and gives the same answer. Standard library only:
they run in env/analysis and never import CHESS."""


class JudgeError(ValueError):
    pass
