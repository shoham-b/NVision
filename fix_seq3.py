with open("nvision/belief/abstract_marginal.py", "r") as f:
    content = f.read()

content = content.replace("from collections.abc import Iterator, Mapping", "from collections.abc import Iterator, Mapping, Sequence")

with open("nvision/belief/abstract_marginal.py", "w") as f:
    f.write(content)
