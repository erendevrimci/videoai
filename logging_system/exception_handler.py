"""
Exception handler module for VideoAI project.

Provides global exception handling utilities and custom exception types.
"""
from typing import Optional, Callable, Any, TypeVar, Type
import sys
import traceback
from functools import wraps

from .logger import Logger, logger as global_logger

# Type variable for generic functions
T = TypeVar('T')

class VideoAIException(Exception):
    """Base exception class for all VideoAI exceptions."""
    
    def __init__(self, message: str, *args, cause: Optional[Exception] = None):
        super().__init__(message, *args)
        self.cause = cause
        
    def __str__(self) -> str:
        result = super().__str__()
        if self.cause:
            result += f" Caused by: {type(self.cause).__name__}: {str(self.cause)}"
        return result


class FileOperationError(VideoAIException):
    """Exception raised for file operation errors."""
    pass


class APIError(VideoAIException):
    """Exception raised for API errors."""
    
    def __init__(self, message: str, *args, status_code: Optional[int] = None, 
                 response_text: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(message, *args, cause=cause)
        self.status_code = status_code
        self.response_text = response_text
        
    def __str__(self) -> str:
        result = super().__str__()
        details = []
        if self.status_code is not None:
            details.append(f"Status: {self.status_code}")
        if self.response_text:
            details.append(f"Response: {self.response_text[:200]}...")
        if details:
            result += f" [{', '.join(details)}]"
        return result


class ConfigError(VideoAIException):
    """Exception raised for configuration errors."""
    pass


class ProcessError(VideoAIException):
    """Exception raised for process execution errors."""
    pass


def log_exceptions(
    logger_instance: Optional[Logger] = None,
    exit_on_error: bool = False,
    default_return: Any = None,
    reraise: bool = False,
    exception_types: Optional[tuple[Type[Exception]]] = None
) -> Callable:
    """
    Decorator for logging exceptions raised in a function.
    
    Args:
        logger_instance: Logger instance to use. If None, uses the global logger.
        exit_on_error: Whether to exit the program on error.
        default_return: Value to return if an exception occurs.
        reraise: Whether to re-raise the exception after logging.
        exception_types: Tuple of exception types to catch. If None, catches all exceptions.
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            log = logger_instance or global_logger
            try:
                return func(*args, **kwargs)
            except exception_types or Exception as e:
                # Get function metadata for better error context
                module_name = func.__module__
                func_name = func.__qualname__
                error_context = f"{module_name}.{func_name}"
                
                log.error(f"Exception in {error_context}: {str(e)}", exc_info=True)
                
                if exit_on_error:
                    log.critical(f"Exiting due to error in {error_context}")
                    sys.exit(1)
                    
                if reraise:
                    raise
                    
                return default_return
        return wrapper
    return decorator


def global_exception_handler(exc_type, exc_value, exc_traceback):
    """
    Global exception handler for unhandled exceptions.
    
    Args:
        exc_type: Exception type
        exc_value: Exception value
        exc_traceback: Exception traceback
    """
    # Skip KeyboardInterrupt handling to allow clean Ctrl+C exit
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
        
    global_logger.critical(
        f"Unhandled exception: {exc_type.__name__}: {exc_value}",
        exc_info=(exc_type, exc_value, exc_traceback)
    )


# Install the global exception handler
sys.excepthook = global_exception_handler


def setup_thread_exception_handling():
    """Configure exception handling for all threads."""
    import threading
    
    def thread_exception_handler(args):
        global_logger.error(
            f"Unhandled exception in thread: {args.exc_type.__name__}: {args.exc_value}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
        )
    
    threading.excepthook = thread_exception_handler


# Utility function to safely convert exceptions to custom types
def convert_exception(e: Exception, 
                     new_type: Type[Exception], 
                     message: Optional[str] = None) -> Exception:
    """
    Convert an exception to a custom exception type.
    
    Args:
        e: Original exception
        new_type: New exception type
        message: Optional custom message (if None, uses str(e))
        
    Returns:
        New exception instance
    """
    if message is None:
        message = str(e)
        
    if hasattr(new_type, '__init__') and 'cause' in new_type.__init__.__code__.co_varnames:
        return new_type(message, cause=e)
    else:
        return new_type(message)