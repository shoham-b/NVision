"""PyCharm launcher: run all NVision preset combinations (cache enabled by default)."""

from nvision import run_preset
from nvision.sim.cases import RunCaseName


def main() -> None:
    run_preset(case_name=RunCaseName.ALL)


if __name__ == "__main__":
    main()
