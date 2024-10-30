from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
from functools import partial

def process_batch(
    items: List[Any],
    process_func: Callable,
    n_workers: int,
    use_threading: bool = False,
    **kwargs
) -> List[Any]:
    """Process a batch of items in parallel."""
    
    wrapped_func = partial(process_func, **kwargs)
    Executor = ThreadPoolExecutor if use_threading else ProcessPoolExecutor
    
    with Executor(max_workers=n_workers) as executor:
        results = list(executor.map(wrapped_func, items))
    
    return results
