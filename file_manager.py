"""
File Manager module for VideoAI project.

Provides centralized file operations including path management, directory creation,
and standardized file read/write operations with integrated logging.
"""
import os
import json
import shutil
import traceback
import subprocess
from pathlib import Path
from typing import Union, Optional, Dict, List, Any, TypeVar, Callable, BinaryIO
from contextlib import contextmanager
import tempfile

# Import the logging system
try:
    from logging_system import Logger, LogLevel
    has_logging = True
except ImportError:
    has_logging = False
    
# Type definitions for better type hinting
PathLike = Union[str, Path]
T = TypeVar('T')

class FileManager:
    """
    Centralized file operations manager for the VideoAI project.
    
    Handles path management, directory creation, and file operations
    with standardized error handling, path normalization, and logging.
    """
    
    def __init__(self, base_dir: Optional[PathLike] = None, module_name: str = "file_manager"):
        """
        Initialize the FileManager with the project's base directory.
        
        Args:
            base_dir: The base directory for the project. If None, uses the directory
                     where this file is located.
            module_name: Name of the module for logging (default: 'file_manager')
        """
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent
        else:
            self.base_dir = self.normalize_path(base_dir)
            
        # Initialize logger if available
        self.logger = None
        if has_logging:
            self.logger = Logger.get_logger(module_name)
    
    def normalize_path(self, path: PathLike) -> Path:
        """
        Convert any path-like object to a Path object and resolve it.
        
        Args:
            path: A string path or Path object
            
        Returns:
            Normalized Path object
        """
        if isinstance(path, str):
            path = Path(path)
        return path.resolve()
    
    def get_abs_path(self, rel_path: PathLike) -> Path:
        """
        Convert a path relative to the project base to an absolute path.
        
        Args:
            rel_path: A path relative to the project base directory
            
        Returns:
            An absolute Path object
        """
        path = self.normalize_path(rel_path)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()
    
    def get_channel_output_path(self, channel_number: int, create: bool = True) -> Path:
        """
        Get the output directory for a specific channel.
        
        Args:
            channel_number: The channel number
            create: Whether to create the directory if it doesn't exist
            
        Returns:
            Path to the channel's output directory
        """
        channel_dir = self.get_abs_path(f"outputs/channel_{channel_number}")
        if create and not channel_dir.exists():
            channel_dir.mkdir(parents=True, exist_ok=True)
        return channel_dir
    
    def ensure_dir_exists(self, path: PathLike) -> Path:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Path to the directory
            
        Returns:
            Path object to the directory
        """
        dir_path = self.normalize_path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def file_exists(self, path: PathLike) -> bool:
        """
        Check if a file exists.
        
        Args:
            path: Path to the file
            
        Returns:
            True if the file exists, False otherwise
        """
        file_path = self.normalize_path(path)
        return file_path.is_file()
    
    def dir_exists(self, path: PathLike) -> bool:
        """
        Check if a directory exists.
        
        Args:
            path: Path to the directory
            
        Returns:
            True if the directory exists, False otherwise
        """
        dir_path = self.normalize_path(path)
        return dir_path.is_dir()
    
    def read_text(self, path: PathLike, encoding: str = 'utf-8') -> Optional[str]:
        """
        Read text from a file with error handling.
        
        Args:
            path: Path to the file
            encoding: Text encoding (default: utf-8)
            
        Returns:
            File contents as string, or None if operation failed
        """
        file_path = self.normalize_path(path)
        try:
            return file_path.read_text(encoding=encoding)
        except Exception as e:
            error_msg = f"Error reading file {file_path}: {e}"
            if self.logger:
                self.logger.error(error_msg, exc_info=True)
            else:
                print(error_msg)
                traceback.print_exc()
            return None
    
    def write_text(self, path: PathLike, content: str, encoding: str = 'utf-8') -> bool:
        """
        Write text to a file with error handling.
        
        Args:
            path: Path to the file
            content: Text content to write
            encoding: Text encoding (default: utf-8)
            
        Returns:
            True if successful, False otherwise
        """
        file_path = self.normalize_path(path)
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding=encoding)
            if self.logger:
                self.logger.debug(f"Successfully wrote to file: {file_path}")
            return True
        except Exception as e:
            error_msg = f"Error writing to file {file_path}: {e}"
            if self.logger:
                self.logger.error(error_msg, exc_info=True)
            else:
                print(error_msg)
                traceback.print_exc()
            return False
    
    def read_json(self, path: PathLike, encoding: str = 'utf-8') -> Optional[Dict]:
        """
        Read and parse JSON from a file.
        
        Args:
            path: Path to the JSON file
            encoding: Text encoding (default: utf-8)
            
        Returns:
            Parsed JSON as dictionary, or None if operation failed
        """
        content = self.read_text(path, encoding)
        if content is None:
            return None
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from {path}: {e}")
            traceback.print_exc()
            return None
    
    def write_json(self, path: PathLike, data: Dict, encoding: str = 'utf-8', 
                   indent: int = 2) -> bool:
        """
        Write data as JSON to a file.
        
        Args:
            path: Path to the file
            data: Data to write as JSON
            encoding: Text encoding (default: utf-8)
            indent: JSON indentation level (default: 2)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            content = json.dumps(data, indent=indent)
            return self.write_text(path, content, encoding)
        except Exception as e:
            print(f"Error converting data to JSON for {path}: {e}")
            traceback.print_exc()
            return False
    
    def read_binary(self, path: PathLike) -> Optional[bytes]:
        """
        Read binary data from a file.
        
        Args:
            path: Path to the file
            
        Returns:
            Binary data, or None if operation failed
        """
        file_path = self.normalize_path(path)
        try:
            return file_path.read_bytes()
        except Exception as e:
            print(f"Error reading binary file {file_path}: {e}")
            traceback.print_exc()
            return None
    
    def write_binary(self, path: PathLike, data: bytes) -> bool:
        """
        Write binary data to a file.
        
        Args:
            path: Path to the file
            data: Binary data to write
            
        Returns:
            True if successful, False otherwise
        """
        file_path = self.normalize_path(path)
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(data)
            return True
        except Exception as e:
            print(f"Error writing binary data to {file_path}: {e}")
            traceback.print_exc()
            return False
    
    def copy_file(self, src: PathLike, dst: PathLike) -> bool:
        """
        Copy a file from source to destination.
        
        Args:
            src: Source file path
            dst: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        src_path = self.normalize_path(src)
        dst_path = self.normalize_path(dst)
        
        try:
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            return True
        except Exception as e:
            print(f"Error copying {src_path} to {dst_path}: {e}")
            traceback.print_exc()
            return False
    
    def remove_file(self, path: PathLike, missing_ok: bool = True) -> bool:
        """
        Remove a file with error handling.
        
        Args:
            path: Path to the file
            missing_ok: Don't raise error if file doesn't exist
            
        Returns:
            True if successful or file doesn't exist, False otherwise
        """
        file_path = self.normalize_path(path)
        try:
            if not file_path.exists() and missing_ok:
                return True
            file_path.unlink()
            return True
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")
            traceback.print_exc()
            return False
    
    def list_files(self, dir_path: PathLike, pattern: str = "*") -> List[Path]:
        """
        List files in a directory matching a pattern.
        
        Args:
            dir_path: Directory to search
            pattern: Glob pattern to match (default: "*")
            
        Returns:
            List of matching file paths
        """
        path = self.normalize_path(dir_path)
        return list(path.glob(pattern))
    
    @contextmanager
    def temp_file(self, suffix: str = "", prefix: str = "tmp_", dir: Optional[PathLike] = None):
        """
        Context manager for temporary file handling.
        
        Args:
            suffix: File suffix (extension)
            prefix: File prefix
            dir: Directory for the temporary file
            
        Yields:
            Path object to the temporary file
        """
        # Use the current working directory as a fallback instead of system temp directory
        if dir is None:
            dir = Path.cwd() / "temp_files"
        
        # Normalize and ensure directory exists
        dir = self.normalize_path(dir)
        self.ensure_dir_exists(dir)
        
        # Create a unique filename directly in the specified directory
        import uuid
        temp_name = f"{prefix}{uuid.uuid4().hex}{suffix}"
        temp_path = dir / temp_name
        
        # Create empty file to reserve the name
        temp_path.touch()
        
        try:
            print(f"Created temporary file: {temp_path}")
            yield temp_path
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                    print(f"Removed temporary file: {temp_path}")
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {temp_path}: {e}")
                
    @contextmanager
    def temp_dir(self, suffix: str = "", prefix: str = "tmp_dir_", dir: Optional[PathLike] = None):
        """
        Context manager for temporary directory handling.
        
        Args:
            suffix: Directory suffix
            prefix: Directory prefix
            dir: Parent directory for the temporary directory
            
        Yields:
            Path object to the temporary directory
        """
        # Use the current working directory as a fallback instead of system temp directory
        if dir is None:
            dir = Path.cwd() / "temp_dirs"
        
        # Normalize and ensure directory exists
        dir = self.normalize_path(dir)
        self.ensure_dir_exists(dir)
        
        # Create a unique directory name directly in the specified directory
        import uuid
        temp_name = f"{prefix}{uuid.uuid4().hex}{suffix}"
        temp_dir_path = dir / temp_name
        
        # Create the directory
        temp_dir_path.mkdir(exist_ok=True)
        
        try:
            print(f"Created temporary directory: {temp_dir_path}")
            yield temp_dir_path
        finally:
            # Remove the temporary directory and all its contents
            if temp_dir_path.exists():
                try:
                    shutil.rmtree(temp_dir_path)
                    print(f"Removed temporary directory: {temp_dir_path}")
                except Exception as e:
                    print(f"Error removing temporary directory {temp_dir_path}: {e}")
                    traceback.print_exc()

    def find_video_file(self, path: PathLike, extensions: List[str] = None) -> Optional[Path]:
        """
        Find a video file that might be stored with or without an extension.
        
        Args:
            path: Path to the file (with or without extension)
            extensions: List of extensions to try (default: common video extensions)
            
        Returns:
            Path object to the found file, or None if no file is found
        """
        if extensions is None:
            extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
            
        try:
            norm_path = self.normalize_path(path)
            
            # Debug
            print(f"Searching for video file at {norm_path}")
            
            # First try the exact path
            if norm_path.is_file():
                print(f"Found exact file match: {norm_path}")
                return norm_path
                
            # Try with each extension
            for ext in extensions:
                # If path already has this extension, skip
                if norm_path.suffix.lower() == ext.lower():
                    continue
                    
                # Try with extension
                test_path = Path(f"{norm_path}{ext}")
                if test_path.is_file():
                    print(f"Found file with extension: {test_path}")
                    return test_path
                    
            # Try the clips directory directly
            clips_dir = self.get_abs_path("clips")
            if not norm_path.is_absolute():
                file_name = norm_path.name
                direct_path = clips_dir / file_name
                
                if direct_path.is_file():
                    print(f"Found file in clips directory: {direct_path}")
                    return direct_path
                    
                # Try with extensions in clips directory
                for ext in extensions:
                    if not file_name.lower().endswith(ext.lower()):
                        test_path = clips_dir / f"{file_name}{ext}"
                        if test_path.is_file():
                            print(f"Found file in clips directory with extension: {test_path}")
                            return test_path
                            
            # Try just using the filename (no directory)
            file_name = norm_path.name
            
            # Look for actual clips with similar names
            similar_files = self.list_files(clips_dir, f"*{file_name}*")
            if similar_files:
                print(f"Found similar file: {similar_files[0]}")
                return similar_files[0]
                
            print(f"No matching video file found for {norm_path}")
            return None
            
        except Exception as e:
            print(f"Error in find_video_file: {e}")
            traceback.print_exc()
            return None
        
    def run_ffmpeg(self, args: List[str], check: bool = True) -> bool:
        """
        Run an ffmpeg command with error handling.
        
        Args:
            args: List of arguments to pass to ffmpeg
            check: Whether to raise an exception on non-zero exit status
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ["ffmpeg", "-y"] + args
            subprocess.run(cmd, check=check, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            print(f"Error output: {e.stderr}")
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"Error running FFmpeg: {e}")
            traceback.print_exc()
            return False

    def get_video_output_path(self, channel_number: int, video_name: str) -> Path:
        """
        Get path for video output with proper directory structure.
        
        Args:
            channel_number: Channel number
            video_name: Name of the video file
            
        Returns:
            Path to the video file
        """
        if not video_name.endswith('.mp4'):
            video_name = f"{video_name}.mp4"
            
        return self.get_channel_output_path(channel_number) / video_name
    
    def get_audio_output_path(self, channel_number: int, audio_name: str) -> Path:
        """
        Get path for audio output with proper directory structure.
        
        Args:
            channel_number: Channel number
            audio_name: Name of the audio file
            
        Returns:
            Path to the audio file
        """
        if not audio_name.lower().endswith(('.mp3', '.wav')):
            audio_name = f"{audio_name}.mp3"
        
        # Create the audio directory if it doesn't exist
        audio_dir = self.get_channel_output_path(channel_number) / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
            
        return audio_dir / audio_name

    def get_script_path(self, channel_number: int, script_name: str = "script") -> Path:
        """
        Get path for script file with proper directory structure.
        
        Args:
            channel_number: Channel number
            script_name: Name of the script file
            
        Returns:
            Path to the script file
        """
        if not script_name.endswith('.txt'):
            script_name = f"{script_name}.txt"
            
        return self.get_channel_output_path(channel_number) / script_name
    
    def get_caption_path(self, channel_number: int, caption_name: str = "captions") -> Path:
        """
        Get path for caption file with proper directory structure.
        
        Args:
            channel_number: Channel number
            caption_name: Name of the caption file
            
        Returns:
            Path to the caption file
        """
        if not caption_name.endswith('.srt'):
            caption_name = f"{caption_name}.srt"
            
        return self.get_channel_output_path(channel_number) / caption_name
    
    def safe_operation(self, operation: Callable[..., T], 
                      default: T, operation_name: str = "Operation", 
                      *args, **kwargs) -> T:
        """
        Execute an operation with error handling.
        
        Args:
            operation: Function to execute
            default: Default value to return on failure
            operation_name: Name of the operation for logging (default: "Operation")
            args, kwargs: Arguments to pass to the function
            
        Returns:
            Operation result or default value on failure
        """
        try:
            result = operation(*args, **kwargs)
            if self.logger:
                self.logger.debug(f"{operation_name} completed successfully")
            return result
        except Exception as e:
            error_msg = f"{operation_name} error: {e}"
            if self.logger:
                self.logger.error(error_msg, exc_info=True)
            else:
                print(error_msg)
                traceback.print_exc()
            return default
            
    def log_file_operation(self, success: bool, operation: str, file_path: PathLike,
                           error: Optional[Exception] = None) -> None:
        """
        Log a file operation outcome.
        
        Args:
            success: Whether the operation was successful
            operation: Description of the operation
            file_path: Path to the file involved
            error: Exception if an error occurred
        """
        if not self.logger:
            return
            
        file_path_str = str(self.normalize_path(file_path))
        if success:
            self.logger.debug(f"{operation} successful: {file_path_str}")
        else:
            self.logger.error(
                f"{operation} failed: {file_path_str}", 
                exc_info=error is not None
            )
            if error:
                self.logger.debug(f"Error details: {type(error).__name__}: {str(error)}")