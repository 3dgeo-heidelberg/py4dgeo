import contextlib
import logging
import sys
import time


def create_default_logger(filename=None):
    # Create the logger instance
    logger = logging.getLogger("py4dgeo")

    # Close and remove existing handlers to avoid duplication and leaks
    for handler in logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)

    # Apply default for logfile name
    if filename is None:
        filename = "py4dgeo.log"

    # We format messages including the date
    _formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # We use stdout for DEBUG and INFO messages
    _stdoutandler = logging.StreamHandler(sys.stdout)
    _stdoutandler.setLevel(logging.DEBUG)
    _stdoutandler.addFilter(lambda r: r.levelno <= logging.INFO)
    _stdoutandler.setFormatter(_formatter)
    logger.addHandler(_stdoutandler)

    # We use stderr for WARNING and ERROR messages
    _stderrhandler = logging.StreamHandler(sys.stderr)
    _stderrhandler.setLevel(logging.WARNING)
    _stderrhandler.setFormatter(_formatter)
    logger.addHandler(_stderrhandler)

    # We additionally use a file that is automatically generated
    _filehandler = logging.FileHandler(filename, mode="a", delay=True)
    _filehandler.setLevel(logging.DEBUG)
    _filehandler.setFormatter(_formatter)
    logger.addHandler(_filehandler)

    logger.setLevel(logging.INFO)

    return logger


# Storage to keep the logger instance alive + initial creation
_logger = create_default_logger()


def set_py4dgeo_logfile(filename):
    """Set the logfile used by py4dgeo

    All log messages produced by py4dgeo are logged into this file
    in addition to be logged to stdout/stderr. By default, that file
    is called 'py4dgeo.log'.

    :param filename:
        The name of the logfile to use
    :type filename: str
    """
    global _logger
    _logger = create_default_logger(filename)


@contextlib.contextmanager
def logger_context(msg, level=logging.INFO):
    # Log a message that we started the task described by message
    logger = logging.getLogger("py4dgeo")
    logger.log(level, f"Starting: {msg}")

    # Measure time
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start

    logger.log(level, f"Finished in {duration:.4f}s: {msg}")
