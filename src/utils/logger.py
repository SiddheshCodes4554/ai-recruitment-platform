import logging
import sys

__all__ = ["get_logger"]

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a standardized, pre-configured logger instance.
    
    Args:
        name (str): The name of the module invoking the logger.
        
    Returns:
        logging.Logger: The configured Logger instance.
    """
    return logging.getLogger(name)
