# from functools import wraps
# from . import config

import logging

try:
    from rich.console import Console
    from rich.logging import RichHandler
except ImportError:
    raise ImportError("Rich library is required for logging.")


handler = RichHandler(
    console=Console(width=120),
    show_time=False,
    show_level=False,
    show_path=False
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[handler]
)
logger = logging.getLogger("periodicflow")


# def optimizer(func):
#     """
#     A decorator that checks once for an optimized version of the function.
#     A global flag 'USE_OPTIMIZED' controls whether to use the optimized or regular version,
#     but this check is done only once during the decoration process.
#     """
#     print(config.USE_OPTIMIZED)

#     try:
#         mod = globals().get('optimization', None)
#         optimized_func = getattr(mod, func.__name__, None) if mod else None
#     except AttributeError:
#         print(f"Optimized version of {func.__name__} not found.")
#         optimized_func = None

#     # Determine once whether to use optimized or regular function
#     if config.USE_OPTIMIZED and optimized_func:
#         wrapped_func = optimized_func
#     else:
#         wrapped_func = func

#     @wraps(func)
#     def wrapped_function(*args, **kwargs):
#         return wrapped_func(*args, **kwargs)

#     return wrapped_function
