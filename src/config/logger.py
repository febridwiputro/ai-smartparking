import logging

class Logger:
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0
    
    def __init__(self, name: str):
        # Create a logger object
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
        
        # Create handlers
        file_handler = logging.FileHandler('logging.log')
        console_handler = logging.StreamHandler()
        
        # Create a formatter and set it for handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%d-%b-%y %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)


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


logger = Logger("main")
