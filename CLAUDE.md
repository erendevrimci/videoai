# VideoAI Project Guidelines

## Commands
- Run main pipeline: `python main.py --channel [channel_number]`
- Individual steps:
  - Generate script: `python write_script.py --channel [channel_number]`
  - Generate voice-over: `python voice_over.py --channel [channel_number]`
  - Generate captions: `python captions.py --channel [channel_number]`
  - Edit video: `python video_edit.py --channel [channel_number]`
  - Generate title/description: `python write_title_desc.py --channel [channel_number]`
  - Upload to YouTube: `python upload_video.py --channel [channel_number]`
- Test: `pytest test_filename.py` or `pytest test_filename.py::test_function`
- Logging: Set DEBUG level: `LOG_LEVEL=debug python main.py`

## Code Style
- Use Python type hints with Pydantic models for structured data
- Import order: standard lib, third-party, local modules
- Function docstrings using """triple quotes""" with Args/Returns sections
- Error handling: Use try/except with specific exception types
- Environment variables loaded with python-dotenv
- Use f-strings for string formatting
- Naming: snake_case for variables/functions, PascalCase for classes
- Wrap API calls in try/except blocks with appropriate error handling

## Project Structure
- Configuration centralized in `config.py` using Pydantic models
- Channel-specific configurations in the `config.py` file
- Output files stored in channel-specific directories under `outputs/channel_X/`
- Environment variables loaded from `.env` file using python-dotenv
- File operations centralized in `file_manager.py` using FileManager class

## Error Handling
- All external API calls wrapped in try/except blocks
- Function results return True/False or result/None to indicate success
- Traceback printing for detailed error information in terminal
- All functions properly typed with meaningful return types

## File Management
- Use the FileManager class for all file operations:
  ```python
  from file_manager import FileManager
  
  # Initialize the file manager
  file_mgr = FileManager()
  
  # Get channel-specific paths
  script_path = file_mgr.get_script_path(channel_number)
  video_path = file_mgr.get_video_output_path(channel_number, "output_video")
  
  # Read/write operations with built-in error handling
  script_content = file_mgr.read_text(script_path)
  success = file_mgr.write_text(script_path, new_script)
  
  # JSON operations
  data = file_mgr.read_json(json_path)
  file_mgr.write_json(json_path, data)
  
  # Temporary files with context manager
  with file_mgr.temp_file(suffix=".mp4") as temp_path:
      # Do something with temporary file
      pass  # File automatically cleaned up
  
  # Media operations
  file_mgr.run_ffmpeg(["-i", str(input_file), str(output_file)])
  ```
  
- All file paths should use Path objects, not string paths
- Always use channel-specific paths with the appropriate FileManager methods
- Error handling is built into all FileManager methods