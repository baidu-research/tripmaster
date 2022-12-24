"""
TM logger
"""
import copy
import logging


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The background is set with 40 plus the number of the color, and the foreground with 30

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

from .logger import setup_logging

def formatter_message(message, use_color=True):
    """

    Args:
        message:
        use_color:

    Returns:

    """
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    """
    ColoredFormatter
    """

    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        """

        Args:
            record:

        Returns:

        """
        formatted_record = copy.deepcopy(record)
        levelname = formatted_record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            formatted_record.levelname = levelname_color
        return logging.Formatter.format(self, formatted_record)

#
# class TMLogger(logging.Logger):
#     """
#     TMLogger
#     """
#     FORMAT = "%(asctime)s [$BOLD%(name)-20s$RESET][%(levelname)-18s] %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
#     COLOR_FORMAT = formatter_message(FORMAT, True)
#
#     def __init__(self, name):
#         logging.Logger.__init__(self, name, logging.DEBUG)
#
#         color_formatter = ColoredFormatter(self.COLOR_FORMAT)
#
#         handler = self.handler()
#         handler.setFormatter(color_formatter)
#
#         self.addHandler(handler)
#         self.manager.loggerDict[name] = self
#
#     def handler(self):
#         """
#
#         Returns:
#
#         """
#         return logging.StreamHandler()


import os
script_directory = os.path.dirname(os.path.realpath(__file__))

def setup(config_path="", multi_processing=False):

    """

    Returns:

    """

    if not config_path:
        config_path = os.path.join(script_directory, "config.yaml")

    setup_logging(config_path=config_path, log_path="",
                  capture_print=False, strict=False, guess_level=False,
                  full_context=2,
                  use_multiprocessing=multi_processing)


def setLevel(level):
    """
    globally set info level for all debugger
    Args:
        level:

    Returns:

    """
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

from logging import INFO, WARN, WARNING, ERROR, DEBUG, CRITICAL, FATAL, NOTSET


def getLogger(name=None):
    """

    Returns:

    """
    if name is None:
        name = "tripmaster"

    logger = logging.getLogger(name)

    # FORMAT = "%(asctime)s [$BOLD%(name)-20s$RESET][%(levelname)-18s] %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    # COLOR_FORMAT = formatter_message(FORMAT, True)
    #
    # color_formatter = ColoredFormatter(COLOR_FORMAT)
    #
    # for handler in logger.handlers:
    #     handler.setFormatter(color_formatter)

    return logger

