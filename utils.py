"""Convenience functions for benchmarking."""

import time

cache_tools_available = False
try:
    from cache_tools import dontneed
except ImportError:
    pass
else:
    cache_tools_available = True


def timeit(f):
    """Decorator for function execution timing."""
    def timed(*arg, **kwargs):
        start = time.time()
        ret = f(*arg, **kwargs)
        end = time.time()
        if hasattr(f, "func_name"):
            fname = f.func_name
        else:
            fname = "<unknown>"
        print("Elapsed time for %s(): %.3f s"
              % (fname, (end - start)))
        return ret
    return timed


