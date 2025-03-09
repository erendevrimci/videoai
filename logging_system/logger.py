"""
Centralized logging system for VideoAI project.

Provides standardized logging with different log levels, file rotation,
and customized formatters for both console and file output.
"""
import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any, Union
from enum import Enum
import traceback
import sys
import datetime

class LogLevel(Enum):
    """Enum defining log levels with readable names."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class LoggerConfig:
    """Configuration for Logger."""
    
    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        log_level: LogLevel = LogLevel.INFO,
        console_level: LogLevel = LogLevel.INFO,
        max_file_size_mb: int = 10,
        backup_count: int = 5,
        log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        """
        Initialize logger configuration.
        
        Args:
            log_dir: Directory to store log files. If None, logs directory in project root is used.
            log_level: Minimum level of messages to log to file
            console_level: Minimum level of messages to display in console
            max_file_size_mb: Maximum size of each log file in MB
            backup_count: Number of backup log files to keep
            log_format: Format of log messages
            date_format: Format of date/time in log messages
        """
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).resolve().parent.parent / "logs"
        self.log_level = log_level
        self.console_level = console_level
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        self.log_format = log_format
        self.date_format = date_format

class Logger:
    """
    Centralized logger for VideoAI project.
    
    Handles log configuration, formatting, and output to both console and files.
    Supports different log levels and log rotation.
    """
    
    _loggers: Dict[str, 'Logger'] = {}
    _initialized = False
    _config = LoggerConfig()
    
    @classmethod
    def initialize(cls, config: Optional[LoggerConfig] = None) -> None:
        """
        Initialize the logging system with the given configuration.
        
        Args:
            config: Logger configuration. If None, default configuration is used.
        """
        if config:
            cls._config = config
            
        # Ensure log directory exists
        os.makedirs(cls._config.log_dir, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture all messages at root
        
        # Remove existing handlers if re-initializing
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
        
        # Create formatters
        formatter = logging.Formatter(
            cls._config.log_format,
            cls._config.date_format
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(cls._config.console_level.value)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Create file handler with rotation
        log_file = cls._config.log_dir / "videoai.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=cls._config.max_file_size_mb * 1024 * 1024,
            backupCount=cls._config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(cls._config.log_level.value)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Mark as initialized
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> 'Logger':
        """
        Get a logger instance for the specified name.
        
        Args:
            name: Name of the logger, typically module name
            
        Returns:
            Logger instance
        """
        if not cls._initialized:
            cls.initialize()
            
        if name not in cls._loggers:
            cls._loggers[name] = Logger(name)
            
        return cls._loggers[name]
    
    def __init__(self, name: str):
        """
        Initialize a logger with the given name.
        
        Args:
            name: Name of the logger, typically module name
        """
        self.name = name
        self.logger = logging.getLogger(name)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log a debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log an info message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log a warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, *args, **kwargs) -> None:
        """
        Log an error message.
        
        Args:
            message: Error message
            exc_info: Whether to include exception info in the log. Default is False.
            args, kwargs: Additional arguments for the message
        """
        self.logger.error(message, exc_info=exc_info, *args, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, *args, **kwargs) -> None:
        """
        Log a critical error message.
        
        Args:
            message: Critical error message
            exc_info: Whether to include exception info in the log. Default is True.
            args, kwargs: Additional arguments for the message
        """
        self.logger.critical(message, exc_info=exc_info, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """
        Log an exception message with traceback.
        
        Args:
            message: Exception message
            args, kwargs: Additional arguments for the message
        """
        self.logger.exception(message, *args, **kwargs)
    
    def log_exception(self, e: Exception, message: Optional[str] = None) -> None:
        """
        Log an exception with a custom message.
        
        Args:
            e: The exception to log
            message: Optional custom message to include
        """
        exc_message = f"{message + ': ' if message else ''}{type(e).__name__}: {str(e)}"
        self.logger.error(exc_message, exc_info=True)
        
    def log_and_return(self, e: Exception, message: str, default_value: Any = None) -> Any:
        """
        Log an exception and return a default value.
        
        Args:
            e: Exception to log
            message: Error message
            default_value: Value to return
            
        Returns:
            The default value
        """
        self.log_exception(e, message)
        return default_value
        
    @property
    def log_file_path(self) -> Path:
        """Get the current log file path."""
        return self._config.log_dir / "videoai.log"
    
    @classmethod
    def get_today_log_file(cls) -> Path:
        """Get a log file specific to today's date."""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        return cls._config.log_dir / f"videoai_{today}.log"


# Initialize a global default logger
logger = Logger.get_logger("videoai")