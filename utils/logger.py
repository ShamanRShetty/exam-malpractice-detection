"""
Centralized logging system
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import colorlog

class SystemLogger:
    """Singleton logger for the entire system"""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
    def setup_logger(self, name, log_file=None, level=logging.INFO):
        """
        Setup a logger with console and file handlers
        
        Args:
            name: Logger name
            log_file: Path to log file
            level: Logging level
        
        Returns:
            logger: Configured logger instance
        """
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        
        # Console handler with colors
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        console_format = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(level)
            
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        self._loggers[name] = logger
        return logger
    
    def get_logger(self, name):
        """Get existing logger or create new one"""
        if name in self._loggers:
            return self._loggers[name]
        return self.setup_logger(name)


def get_logger(name, log_file=None):
    """
    Convenience function to get a logger
    
    Args:
        name: Logger name
        log_file: Optional log file path
    
    Returns:
        logger: Logger instance
    """
    system_logger = SystemLogger()
    if log_file:
        return system_logger.setup_logger(name, log_file)
    return system_logger.get_logger(name)