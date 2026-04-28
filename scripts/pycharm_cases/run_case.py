"""PyCharm-friendly group launcher.

Open this file and click the green Run button.
Change GROUP_NAME to pick which preset group to run.
"""

from nvision.cli.groups_cmd import run_preset

# Change this string to pick a group (e.g., "all", "sweep_only", "bayesian_only").
GROUP_NAME: str = "all"


def main() -> None:
    run_preset(group_name=GROUP_NAME)


if __name__ == "__main__":
    main()
