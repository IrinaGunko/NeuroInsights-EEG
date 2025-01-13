import logging
from logging import Logger

class LoggerManager:
    _loggers = {}

    @staticmethod
    def get_logger(name: str) -> Logger:
        if name not in LoggerManager._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)  # Default log level

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)

            logger.addHandler(console_handler)

            logger.propagate = False

            LoggerManager._loggers[name] = logger
        return LoggerManager._loggers[name]
