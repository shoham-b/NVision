from rich.traceback import install
import concurrent.futures
import multiprocessing
import typer

try:
    import numba

    suppress_list = [typer, multiprocessing, concurrent.futures, numba]
except ImportError:
    suppress_list = [typer, multiprocessing, concurrent.futures]

# Configure rich traceback to show relevant stack frames only
install(show_locals=False, suppress=suppress_list, width=100, word_wrap=True)
