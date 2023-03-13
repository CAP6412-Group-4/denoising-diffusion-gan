"""Logger configurations"""

import sys
import logging


class LogFormatter(logging.Formatter):
    """Custom formatter to display colors"""
    
    _grey = "\x1b[38;20m"
    _yellow = "\x1b[33;20m"
    _red = "\x1b[31;20m"
    _bold_red = "\x1b[31;1m"
    _reset = "\x1b[0m"
    _format = "%(asctime)s.%(msecs)-3d %(levelname)-8s %(name)-20s \t %(message)s"
    
    _formatter_map = {
        logging.DEBUG: _grey + _format + _reset,
        logging.INFO: _grey + _format + _reset,
        logging.WARNING: _yellow + _format + _reset,
        logging.ERROR: _red + _format + _reset,
        logging.CRITICAL: _bold_red + _format + _reset
    }
    
    def format(self, record):
        log_fmt = self._formatter_map.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record=record)


def setup_logger(debug: bool = False) -> None:
    """Sets up my own little logger to record all steps in the fuzz test."""

    # Set level
    logging.root.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Logging format
    formatter = LogFormatter()

    # STDOUT for the logger to log to the console
    stream_out_handler = logging.StreamHandler(sys.stdout)
    stream_out_handler.setLevel(logging.DEBUG)
    stream_out_handler.setFormatter(formatter)
    logging.root.addHandler(stream_out_handler)
