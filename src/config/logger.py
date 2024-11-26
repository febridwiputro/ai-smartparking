import os, sys
import logging
from datetime import datetime
from config.config import config

class Logger:
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0

    def __init__(self, name: str, is_save: bool = False):
        """
        Initialize the logger.
        If is_save=True, log to file. Otherwise, log only to the console.
        """

        IS_PC = False
        self.BASE_DIR = config.BASE_PC_DIR if IS_PC else config.BASE_LAPTOP_DIR
        self.DATASET_DIR = os.path.join(self.BASE_DIR, "dataset", "log", "log")
        os.makedirs(self.DATASET_DIR, exist_ok=True)

        self.logger = logging.getLogger(name)
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%d-%b-%y %H:%M:%S')
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            if is_save:
                log_filename = f'logging-{datetime.now().strftime("%Y%m%d")}.log'
                log_filename = os.path.join(self.DATASET_DIR, log_filename)
                file_handler = logging.FileHandler(log_filename)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def write(self, message, level):
        # Map level to the appropriate logging method
        if level == self.CRITICAL:
            self.logger.critical(message)
        elif level == self.ERROR:
            self.logger.error(message)
        elif level == self.WARNING:
            self.logger.warning(message)
        elif level == self.INFO:
            self.logger.info(message)
        elif level == self.DEBUG:
            self.logger.debug(message)
        elif level == self.NOTSET:
            self.logger.log(logging.NOTSET, message)
        else:
            raise ValueError(f"Unknown level: {level}")

# logger = Logger("main", is_save=True)

# logger = Logger("main", is_save=True)  # Logs only to the console
# logger_console.write("This is a debug message on console only.", Logger.DEBUG)

# logger_file = Logger("main", is_save=True)  # Logs to both file and console
# logger_file.write("This is an info message saved to the file and console.", Logger.INFO)