from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, List

class ParallelExecutor:
    """Handles parallel execution of molecule computations."""

    def __init__(self, n_workers: int, use_threading: bool = False):
        """Initialize parallel executor.
        Args:
            n_workers: Number of worker processes/threads
            use_threading: Use threads instead of processes
        """
        self.n_workers = n_workers
        self.use_threading = use_threading

    def execute_batch(self, items: List[Any], process_func: Callable, **kwargs) -> List[Any]:
        """Execute function on items in parallel."""
        wrapped_func = partial(process_func, **kwargs)
        Executor = ThreadPoolExecutor if self.use_threading else ProcessPoolExecutor
        with Executor(max_workers=self.n_workers) as executor:
            results = list(executor.map(wrapped_func, items))

        return results
