"""Keep CLI-installed gsMap logging from leaking between tests."""

import logging
from contextlib import contextmanager


@contextmanager
def preserved_gsmap_logging():
    """Restore the ``gsMap`` logger after code that installs CLI or stage logging.

    The CLI disables propagation and adds console handlers to the ``gsMap`` logger. If
    that state leaks, ``caplog`` captures nothing in every later test.
    """
    from gsMap.logging_config import _reset_logging_for_tests

    logger = logging.getLogger("gsMap")
    level, propagate, disabled = logger.level, logger.propagate, logger.disabled
    handlers = logger.handlers[:]
    try:
        yield
    finally:
        _reset_logging_for_tests()
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate
        logger.disabled = disabled
