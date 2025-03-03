# VideoAI Project Guidelines

## Commands
- Run main pipeline: `python main.py`
- Individual steps:
  - Generate script: `python write_script.py`
  - Generate voice-over: `python voice_over.py [channel_number]`
  - Generate captions: `python captions.py`
  - Edit video: `python video_edit.py`
  - Generate title/description: `python write_title_desc.py`
  - Upload to YouTube: `python upload_video.py`
- Test: `pytest test_filename.py` or `pytest test_filename.py::test_function`

## Code Style
- Use Python type hints with Pydantic models for structured data
- Import order: standard lib, third-party, local modules
- Function docstrings using """triple quotes""" with Args/Returns sections
- Error handling: Use try/except with specific exception types
- Environment variables loaded with python-dotenv
- Use f-strings for string formatting
- Naming: snake_case for variables/functions, PascalCase for classes
- Wrap API calls in try/except blocks with appropriate error handling