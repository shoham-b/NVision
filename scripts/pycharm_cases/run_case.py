"""PyCharm-friendly case launcher.

Open this file and click the green Run button.
Change CASE_NAME to pick which preset to run.
"""

from nvision.cli.cases_cmd import run_preset
from nvision.sim.cases import RunCaseName

# Type-safe case selection with autocomplete in IDEs.
CASE_NAME: RunCaseName = RunCaseName.NVCENTER


def main() -> None:
    run_preset(case_name=CASE_NAME)


if __name__ == "__main__":
    main()
