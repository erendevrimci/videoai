# FileManager Usage Guide

The `FileManager` class provides a centralized approach to file operations in the VideoAI project. This guide explains how to use the FileManager in your modules and how to convert existing file operations.

## Basic Usage

### Initialization

```python
from file_manager import FileManager

# Initialize with default base directory (project root)
file_mgr = FileManager()

# Or initialize with a custom base directory
file_mgr = FileManager("/custom/path")
```

### Path Management

```python
# Get absolute path from relative path
abs_path = file_mgr.get_abs_path("relative/path")

# Get channel-specific output directory
channel_dir = file_mgr.get_channel_output_path(channel_number)

# Get path for specific file types
script_path = file_mgr.get_script_path(channel_number)
audio_path = file_mgr.get_audio_output_path(channel_number, "voice_file")
video_path = file_mgr.get_video_output_path(channel_number, "final_video")
caption_path = file_mgr.get_caption_path(channel_number)
```

### Reading and Writing Files

```python
# Text operations
content = file_mgr.read_text(file_path)
success = file_mgr.write_text(file_path, "New content")

# Binary operations
binary_data = file_mgr.read_binary(file_path)
success = file_mgr.write_binary(file_path, binary_data)

# JSON operations
data = file_mgr.read_json(json_path)
success = file_mgr.write_json(json_path, data_dict)
```

### Directory Operations

```python
# Check if file or directory exists
if file_mgr.file_exists(file_path):
    # Do something
    
if file_mgr.dir_exists(dir_path):
    # Do something

# Create directory if it doesn't exist
dir_path = file_mgr.ensure_dir_exists(dir_path)

# List files in a directory
files = file_mgr.list_files(dir_path, "*.mp4")
```

### File Utilities

```python
# Copy a file
file_mgr.copy_file(source_path, dest_path)

# Remove a file
file_mgr.remove_file(file_path)

# With temporary file
with file_mgr.temp_file(suffix=".mp4") as temp_path:
    # Do something with the temporary file
    # File is automatically removed when done
```

### Media Operations

```python
# Run ffmpeg command
success = file_mgr.run_ffmpeg([
    "-i", str(input_file),
    "-c:v", "libx264",
    str(output_file)
])
```

## Converting Existing Code

### Before:

```python
# Path handling
import os
from pathlib import Path
base_dir = Path(__file__).resolve().parent
path = os.path.join(base_dir, "some_dir", "file.txt")

# Directory creation
os.makedirs(dir_path, exist_ok=True)

# File reading/writing
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)
    
# JSON operations
import json
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    
# Error handling scattered throughout
try:
    with open(path, "r") as f:
        data = f.read()
except FileNotFoundError:
    print(f"File not found: {path}")
    return None
```

### After:

```python
from file_manager import FileManager

# Initialize once
file_mgr = FileManager()

# Path handling
path = file_mgr.get_abs_path("some_dir/file.txt")

# Directory creation (built into most methods)
file_mgr.ensure_dir_exists(dir_path)

# File reading/writing with built-in error handling
content = file_mgr.read_text(file_path)
success = file_mgr.write_text(file_path, new_content)

# JSON operations with built-in error handling
data = file_mgr.read_json(json_path)
success = file_mgr.write_json(json_path, data)

# Simplified error handling with consistent returns
if content is None:
    print("Failed to read file")
    return None
```

## Best Practices

1. **Initialize Once**: Create a single FileManager instance at the module level or pass it between functions rather than creating multiple instances.

2. **Use Path Objects**: The FileManager accepts and returns Path objects, which offer better functionality than string paths.

3. **Use Higher-Level Methods**: Use the specialized methods like `get_script_path()` instead of constructing paths manually.

4. **Check Return Values**: All operations return appropriate values to indicate success/failure. Always check these values:
   - Read methods return the content or `None` if failed
   - Write methods return `True` or `False` to indicate success

5. **For Complex Operations**: Use the `safe_operation()` method to wrap functions that might fail:

   ```python
   result = file_mgr.safe_operation(some_risky_function, default_value, arg1, arg2)
   ```

6. **For Temporary Files**: Use the context manager:

   ```python
   with file_mgr.temp_file() as temp_path:
       # Work with temp_path
       # File is automatically cleaned up afterwards
   ```

7. **For FFmpeg Operations**: Use the `run_ffmpeg()` method instead of calling subprocess directly.

## Integration Example

For a complete example of integrating FileManager into an existing module, see the [file_manager_example.py](../examples/file_manager_example.py) file that demonstrates refactoring of the voice_over.py module.