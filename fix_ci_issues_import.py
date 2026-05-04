with open("nvision/spectra/signal.py") as f:
    content = f.read()

print("Is GenericParamSpec in signal.py?", "GenericParamSpec" in content)
