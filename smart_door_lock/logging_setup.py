import logging
import sys


def setup_logging(level=logging.DEBUG, name=None):
    """
    Configure root logging to stream to the original stdout with timestamps.
    Redirects `sys.stdout` and `sys.stderr` to logger so prints and tracebacks
    appear in terminal with consistent formatting.

    Returns the configured logger.
    """
    logger = logging.getLogger(name) if name else logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate output
    for h in list(logger.handlers):
        logger.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.__stdout__)
    fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    handler.setFormatter(logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)

    class StreamToLogger:
        def __init__(self, log, level=logging.INFO):
            self.log = log
            self.level = level

        def write(self, buf):
            if not buf:
                return
            for line in buf.rstrip().splitlines():
                if line:
                    self.log.log(self.level, line)

        def flush(self):
            for h in self.log.handlers:
                try:
                    h.flush()
                except Exception:
                    pass

    # Redirect stdout/stderr to logger (handler writes to sys.__stdout__ so no loop)
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

    return logger
