"""PyCharm launcher: run all NVision preset combinations (cache enabled by default)."""

from nvision.cli.groups_cmd import run_preset


def main() -> None:
    run_preset(group_name="all")


if __name__ == "__main__":
    main()
