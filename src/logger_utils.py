import os
import logging

def get_logger(name,params):

    # Configure logging
    log_file=params['logging']['file']
    os.makedirs(os.path.dirname(log_file),exist_ok=True)

    logging.basicConfig(level=params['logging']['level'])
    logger = logging.getLogger(name)

    if not logger.handlers:
        # File handler for logging to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level=params['logging']['level'])
        file_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))  
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    return logger