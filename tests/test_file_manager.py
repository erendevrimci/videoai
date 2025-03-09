"""
Unit tests for the FileManager class.
"""
import os
import json
import shutil
import tempfile
from pathlib import Path
from unittest import mock
import pytest

from file_manager import FileManager

class TestFileManager:
    """Test suite for FileManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def file_manager(self, temp_dir):
        """Create a FileManager instance with the temporary directory as base."""
        return FileManager(temp_dir)
    
    def test_init_default(self):
        """Test initialization with default base_dir."""
        with mock.patch('pathlib.Path.resolve', return_value=Path('/fake/path')):
            fm = FileManager()
            assert fm.base_dir == Path('/fake/path')
    
    def test_init_custom(self, temp_dir):
        """Test initialization with custom base_dir."""
        fm = FileManager(temp_dir)
        assert fm.base_dir == temp_dir
    
    def test_normalize_path_string(self, file_manager):
        """Test normalize_path with string input."""
        path = file_manager.normalize_path("test_path")
        assert isinstance(path, Path)
        assert path.is_absolute()
    
    def test_normalize_path_pathlib(self, file_manager, temp_dir):
        """Test normalize_path with Path input."""
        path = file_manager.normalize_path(temp_dir / "test_path")
        assert isinstance(path, Path)
        assert path.is_absolute()
    
    def test_get_abs_path_relative(self, file_manager, temp_dir):
        """Test get_abs_path with relative path."""
        rel_path = "subdir/file.txt"
        abs_path = file_manager.get_abs_path(rel_path)
        assert abs_path == (temp_dir / rel_path).resolve()
    
    def test_get_abs_path_absolute(self, file_manager):
        """Test get_abs_path with absolute path."""
        abs_path = Path("/absolute/path")
        result = file_manager.get_abs_path(abs_path)
        assert result == abs_path
    
    def test_get_channel_output_path(self, file_manager, temp_dir):
        """Test get_channel_output_path creates correct path."""
        channel_number = 2
        expected_path = temp_dir / "outputs" / f"channel_{channel_number}"
        path = file_manager.get_channel_output_path(channel_number)
        assert path == expected_path
        assert path.exists()
        assert path.is_dir()
    
    def test_ensure_dir_exists(self, file_manager, temp_dir):
        """Test ensure_dir_exists creates directory."""
        test_dir = temp_dir / "test_dir" / "nested"
        result = file_manager.ensure_dir_exists(test_dir)
        assert result == test_dir
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_file_exists(self, file_manager, temp_dir):
        """Test file_exists returns correct boolean."""
        # Test non-existent file
        non_existent = temp_dir / "non_existent.txt"
        assert not file_manager.file_exists(non_existent)
        
        # Test existing file
        existing = temp_dir / "existing.txt"
        existing.touch()
        assert file_manager.file_exists(existing)
        
        # Test directory
        directory = temp_dir / "directory"
        directory.mkdir()
        assert not file_manager.file_exists(directory)
    
    def test_dir_exists(self, file_manager, temp_dir):
        """Test dir_exists returns correct boolean."""
        # Test non-existent directory
        non_existent = temp_dir / "non_existent_dir"
        assert not file_manager.dir_exists(non_existent)
        
        # Test existing directory
        existing = temp_dir / "existing_dir"
        existing.mkdir()
        assert file_manager.dir_exists(existing)
        
        # Test file
        file_path = temp_dir / "file.txt"
        file_path.touch()
        assert not file_manager.dir_exists(file_path)
    
    def test_read_write_text(self, file_manager, temp_dir):
        """Test read_text and write_text methods."""
        file_path = temp_dir / "test.txt"
        content = "Hello, world!"
        
        # Test write_text
        result = file_manager.write_text(file_path, content)
        assert result is True
        assert file_path.exists()
        
        # Test read_text
        read_content = file_manager.read_text(file_path)
        assert read_content == content
        
        # Test read_text with non-existent file
        non_existent = temp_dir / "non_existent.txt"
        assert file_manager.read_text(non_existent) is None
    
    def test_read_write_json(self, file_manager, temp_dir):
        """Test read_json and write_json methods."""
        file_path = temp_dir / "test.json"
        data = {"key": "value", "numbers": [1, 2, 3]}
        
        # Test write_json
        result = file_manager.write_json(file_path, data)
        assert result is True
        assert file_path.exists()
        
        # Test read_json
        read_data = file_manager.read_json(file_path)
        assert read_data == data
        
        # Test read_json with non-existent file
        non_existent = temp_dir / "non_existent.json"
        assert file_manager.read_json(non_existent) is None
        
        # Test read_json with invalid JSON
        invalid_json = temp_dir / "invalid.json"
        invalid_json.write_text("{invalid: json}")
        assert file_manager.read_json(invalid_json) is None
    
    def test_read_write_binary(self, file_manager, temp_dir):
        """Test read_binary and write_binary methods."""
        file_path = temp_dir / "test.bin"
        data = b"\x00\x01\x02\x03"
        
        # Test write_binary
        result = file_manager.write_binary(file_path, data)
        assert result is True
        assert file_path.exists()
        
        # Test read_binary
        read_data = file_manager.read_binary(file_path)
        assert read_data == data
        
        # Test read_binary with non-existent file
        non_existent = temp_dir / "non_existent.bin"
        assert file_manager.read_binary(non_existent) is None
    
    def test_copy_file(self, file_manager, temp_dir):
        """Test copy_file method."""
        source = temp_dir / "source.txt"
        dest = temp_dir / "dest.txt"
        content = "Test content"
        
        # Create source file
        source.write_text(content)
        
        # Test copy_file
        result = file_manager.copy_file(source, dest)
        assert result is True
        assert dest.exists()
        assert dest.read_text() == content
        
        # Test copy to nested directory
        nested_dest = temp_dir / "nested" / "dest.txt"
        result = file_manager.copy_file(source, nested_dest)
        assert result is True
        assert nested_dest.exists()
        assert nested_dest.read_text() == content
        
        # Test copy non-existent file
        non_existent = temp_dir / "non_existent.txt"
        result = file_manager.copy_file(non_existent, dest)
        assert result is False
    
    def test_remove_file(self, file_manager, temp_dir):
        """Test remove_file method."""
        file_path = temp_dir / "to_remove.txt"
        file_path.touch()
        
        # Test remove_file
        assert file_path.exists()
        result = file_manager.remove_file(file_path)
        assert result is True
        assert not file_path.exists()
        
        # Test remove non-existent file with missing_ok=True
        result = file_manager.remove_file(file_path, missing_ok=True)
        assert result is True
    
    def test_list_files(self, file_manager, temp_dir):
        """Test list_files method."""
        # Create test files
        (temp_dir / "file1.txt").touch()
        (temp_dir / "file2.txt").touch()
        (temp_dir / "file3.jpg").touch()
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file4.txt").touch()
        
        # Test with default pattern
        files = file_manager.list_files(temp_dir)
        assert len(files) == 4  # 3 files + 1 directory
        
        # Test with specific pattern
        txt_files = file_manager.list_files(temp_dir, "*.txt")
        assert len(txt_files) == 2
        assert all(f.suffix == ".txt" for f in txt_files)
        
        # Test nested path
        subdir_files = file_manager.list_files(temp_dir / "subdir")
        assert len(subdir_files) == 1
    
    def test_temp_file(self, file_manager):
        """Test temp_file context manager."""
        # Test basic usage
        with file_manager.temp_file() as temp_path:
            assert temp_path.exists()
            temp_path.write_text("test")
            assert temp_path.read_text() == "test"
        # File should be removed after context exit
        assert not temp_path.exists()
        
        # Test with suffix and prefix
        with file_manager.temp_file(suffix=".txt", prefix="test_") as temp_path:
            assert temp_path.suffix == ".txt"
            assert temp_path.name.startswith("test_")
    
    def test_run_ffmpeg(self, file_manager):
        """Test run_ffmpeg method with mocked subprocess."""
        with mock.patch('subprocess.run') as mock_run:
            # Successful run
            mock_run.return_value.returncode = 0
            result = file_manager.run_ffmpeg(["-version"])
            assert result is True
            mock_run.assert_called_with(
                ["ffmpeg", "-y", "-version"], 
                check=True, 
                capture_output=True, 
                text=True
            )
            
            # Failed run
            mock_run.side_effect = Exception("Command failed")
            result = file_manager.run_ffmpeg(["-invalid"])
            assert result is False
    
    def test_get_video_output_path(self, file_manager, temp_dir):
        """Test get_video_output_path method."""
        channel_number = 3
        video_name = "test_video"
        
        # Test without extension
        path = file_manager.get_video_output_path(channel_number, video_name)
        expected = temp_dir / "outputs" / f"channel_{channel_number}" / f"{video_name}.mp4"
        assert path == expected
        
        # Test with extension
        path = file_manager.get_video_output_path(channel_number, f"{video_name}.mp4")
        assert path == expected
    
    def test_get_audio_output_path(self, file_manager, temp_dir):
        """Test get_audio_output_path method."""
        channel_number = 3
        audio_name = "test_audio"
        
        # Test without extension
        path = file_manager.get_audio_output_path(channel_number, audio_name)
        expected = temp_dir / "outputs" / f"channel_{channel_number}" / "audio" / f"{audio_name}.mp3"
        assert path == expected
        
        # Test with extension
        path = file_manager.get_audio_output_path(channel_number, f"{audio_name}.mp3")
        assert path == expected
        
        # Test with different extension
        path = file_manager.get_audio_output_path(channel_number, f"{audio_name}.wav")
        expected = temp_dir / "outputs" / f"channel_{channel_number}" / "audio" / f"{audio_name}.wav"
        assert path == expected
    
    def test_get_script_path(self, file_manager, temp_dir):
        """Test get_script_path method."""
        channel_number = 3
        
        # Test with default name
        path = file_manager.get_script_path(channel_number)
        expected = temp_dir / "outputs" / f"channel_{channel_number}" / "script.txt"
        assert path == expected
        
        # Test with custom name
        script_name = "custom_script"
        path = file_manager.get_script_path(channel_number, script_name)
        expected = temp_dir / "outputs" / f"channel_{channel_number}" / f"{script_name}.txt"
        assert path == expected
        
        # Test with extension
        path = file_manager.get_script_path(channel_number, f"{script_name}.txt")
        assert path == expected
    
    def test_get_caption_path(self, file_manager, temp_dir):
        """Test get_caption_path method."""
        channel_number = 3
        
        # Test with default name
        path = file_manager.get_caption_path(channel_number)
        expected = temp_dir / "outputs" / f"channel_{channel_number}" / "captions.srt"
        assert path == expected
        
        # Test with custom name
        caption_name = "custom_captions"
        path = file_manager.get_caption_path(channel_number, caption_name)
        expected = temp_dir / "outputs" / f"channel_{channel_number}" / f"{caption_name}.srt"
        assert path == expected
        
        # Test with extension
        path = file_manager.get_caption_path(channel_number, f"{caption_name}.srt")
        assert path == expected
    
    def test_safe_operation(self, file_manager):
        """Test safe_operation method."""
        # Test with successful operation
        def successful_op(a, b):
            return a + b
        
        result = file_manager.safe_operation(successful_op, "default", 1, 2)
        assert result == 3
        
        # Test with failing operation
        def failing_op():
            raise ValueError("Test error")
        
        result = file_manager.safe_operation(failing_op, "default")
        assert result == "default"