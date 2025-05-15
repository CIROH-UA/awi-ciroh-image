""" Logger setup for TEVA """
import logging
import datetime
import os.path
import sys

def setup_logger(logger_name: str,
                 logger_file: str,
                 output_logging_level: int = logging.INFO,
                 logfile_logging_level: int = logging.INFO) -> logging.Logger:
    """ Sets up a :class:`logging.Logger` that will log to both the console and an output file

    :param logger_name: The name of the logger
    :param logger_file: The filepath of the logger file to be written to
    :param output_logging_level: The logging level of the console
    :param logfile_logging_level: The logging level of the output file
    :return: The newly setup logger
    """
    if not os.path.exists("teva_logs"):
        os.mkdir("teva_logs")

    # set the logger level to the minimum requested level
    logger = logging.getLogger(logger_name)
    logger.setLevel(min(output_logging_level, logfile_logging_level))

    # Create a file handler which logs even debug messages
    fh = logging.FileHandler(f"teva_logs/{logger_file}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    fh.setLevel(logfile_logging_level)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        # Create a console handler with its own log level
        # Pipe all logger commands INFO or lower to STDOUT
        ch_std = logging.StreamHandler(stream=sys.stdout)
        ch_std.addFilter(lambda rec: rec.levelno <= logging.INFO)
        ch_std.setLevel(output_logging_level)

        # Pipe all logger commands greater than INFO to STDERR
        ch_err = logging.StreamHandler(stream=sys.stderr)
        ch_err.addFilter(lambda rec: rec.levelno > logging.INFO)
        ch_err.setLevel(output_logging_level)

        ch_std.setFormatter(formatter)
        ch_err.setFormatter(formatter)

        logger.addHandler(ch_std)
        logger.addHandler(ch_err)

    fh.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    return logger
