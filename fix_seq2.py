with open("nvision/belief/abstract_marginal.py", "r") as f:
    content = f.read()

content = content.replace("from collections.abc import Mapping", "from collections.abc import Mapping, Sequence")
content = content.replace("def batch_update(self, observations: 'Sequence[Observation]') -> None:", "def batch_update(self, observations: Sequence[Observation]) -> None:")

with open("nvision/belief/abstract_marginal.py", "w") as f:
    f.write(content)
