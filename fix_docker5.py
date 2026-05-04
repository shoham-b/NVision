print("CI failed again in build worker: ModuleNotFoundError: No module named 'numpy'")
print(
    "This means the previous fix 'explicitly add numpy to pyproject.toml' was not properly pushed or did not resolve the issue!"
)
