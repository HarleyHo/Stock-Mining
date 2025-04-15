import logging


def setup_logger() -> logging.Logger:
    """
    Initialize a simple console logger with INFO level.
    Returns: Configured logger instance
    """
    logger = logging.getLogger('StockData')
    logger.setLevel(logging.INFO)

    # Create console handler if not already present
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Set simple format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger