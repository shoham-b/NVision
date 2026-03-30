def install_rich_tracebacks() -> None:
    """Configure rich traceback to show relevant stack frames only."""
    import concurrent.futures
    import multiprocessing

    import typer
    from rich.traceback import install

    try:
        import numba

        suppress = [typer, multiprocessing, concurrent.futures, numba]
    except ImportError:
        suppress = [typer, multiprocessing, concurrent.futures]

    install(show_locals=False, suppress=suppress, width=100, word_wrap=True)


# Note: We don't call it here to avoid early import side-effects.
# The CLI entrypoints will call it as needed.
