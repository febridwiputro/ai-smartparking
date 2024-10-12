import logging
from datetime import datetime

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

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%d-%b-%y %H:%M:%S')

        # Create handlers
        console_handler = logging.StreamHandler()  # Always log to the console
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if is_save:
            # Log to file with daily log file name
            log_filename = f'logging-{datetime.now().strftime("%Y%m%d")}.log'
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

# Example usage:
logger = Logger("main", is_save=False)  # Logs only to the console
# logger_console.write("This is a debug message on console only.", Logger.DEBUG)

# logger_file = Logger("main", is_save=True)  # Logs to both file and console
# logger_file.write("This is an info message saved to the file and console.", Logger.INFO)
