# VideoAI Logging System

A comprehensive logging system for the VideoAI project with exception handling capabilities.

## Features

- Centralized logging with configurable log levels
- File rotation for log files with size limits
- Console and file output with customizable formatters
- Module-specific loggers
- Exception handling utilities
- Custom exception types
- Function decorators for exception handling

## Usage

### Basic Logging

```python
from logging_system import Logger, LogLevel

# Get a module-specific logger
logger = Logger.get_logger("my_module")

# Log different levels
logger.debug("Debug information")
logger.info("Information message")
logger.warning("Warning message")
logger.error("Error message", exc_info=True)  # With exception info
logger.critical("Critical error", exc_info=True)

# Log an exception with context
try:
    # Some code that might fail
    result = process_data()
except Exception as e:
    logger.log_exception(e, "Error processing data")
```

### Custom Configuration

```python
from logging_system import Logger, LoggerConfig, LogLevel
from pathlib import Path

# Create custom configuration
config = LoggerConfig(
    log_dir=Path("/custom/log/dir"),
    log_level=LogLevel.DEBUG,
    console_level=LogLevel.INFO,
    max_file_size_mb=20,
    backup_count=10
)

# Initialize logging system with custom config
Logger.initialize(config)

# Get a logger
logger = Logger.get_logger("my_module")
```

### Exception Handling

```python
from logging_system import log_exceptions, APIError, FileOperationError

# Decorate functions for automatic exception handling
@log_exceptions()
def process_file(file_path):
    # This function will have exceptions logged automatically
    with open(file_path) as f:
        data = f.read()
    return data

# With custom return value on error
@log_exceptions(default_return=[])
def get_items():
    # On exception, returns empty list instead of None
    items = fetch_items_from_api()
    return items

# Raising custom exceptions
from logging_system import convert_exception

try:
    response = call_api()
    if response.status_code != 200:
        raise APIError(
            "API call failed", 
            status_code=response.status_code,
            response_text=response.text
        )
except Exception as e:
    # Convert the exception to a custom type
    raise convert_exception(e, FileOperationError, "Could not process the API response")
```

### Global Exception Handling

The logging system automatically installs a global exception handler to catch unhandled exceptions.

```python
from logging_system import setup_thread_exception_handling

# Set up exception handling for all threads
setup_thread_exception_handling()
```

## Integration with FileManager

The logging system is fully integrated with the FileManager class:

```python
from logging_system import Logger
from file_manager import FileManager

logger = Logger.get_logger("my_module")
file_mgr = FileManager(module_name="my_module")

# FileManager will automatically log operations
result = file_mgr.read_text("config.json")

# Log file operations manually
file_mgr.log_file_operation(
    success=True,
    operation="File backup",
    file_path="/path/to/file.txt"
)
```