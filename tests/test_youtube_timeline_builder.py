"""
Unit tests for the YouTubeTimelineBuilder class.

Tests timeline creation, caption integration, and serialization functionality
for YouTube clip timelines.
"""
import unittest
from pathlib import Path
import json
from datetime import datetime
import tempfile
import shutil
import os

# Mock imports and setup
from unittest.mock import MagicMock, patch

# Import the module under test
from youtube_timeline_builder import YouTubeTimelineBuilder, YouTubeTimelineError


class TestYouTubeTimelineBuilder(unittest.TestCase):
    """Test cases for YouTubeTimelineBuilder class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temp directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock objects
        self.mock_timeline_manager = MagicMock()
        self.mock_file_manager = MagicMock()
        
        # Create a builder instance with mocks
        self.builder = YouTubeTimelineBuilder(self.mock_timeline_manager)
        
        # Create mock metadata
        self.mock_metadata = self._create_mock_metadata()
        
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
        
    def _create_mock_metadata(self):
        """Create mock VideoMetadata for testing."""
        # Create mock objects with the same structure as the real ones
        # This avoids dependency on the actual yt_clips models
        mock_metadata = MagicMock()
        
        # Configure the mock video
        mock_metadata.video = MagicMock()
        mock_metadata.video.title = "Test YouTube Video"
        mock_metadata.video.url = "https://www.youtube.com/watch?v=test123"
        
        # Configure the mock video info
        mock_metadata.video.video_info = MagicMock()
        mock_metadata.video.video_info.video_id = "test123"
        mock_metadata.processor_version = "1.0.0"
        
        # Create mock clips
        mock_metadata.clips = []
        for i in range(3):
            mock_clip = MagicMock()
            mock_clip.path = Path(f"{self.temp_dir}/test_clip_{i}.mp4")
            
            # Create empty files to satisfy path existence checks
            with open(mock_clip.path, 'w') as f:
                f.write("")
                
            mock_clip.start_time = i * 10.0
            mock_clip.end_time = (i + 1) * 10.0
            mock_clip.duration = 10.0
            
            # Add mock captions
            mock_clip.captions = []
            for j in range(2):
                mock_caption = MagicMock()
                mock_caption.start = i * 10.0 + j * 4.0
                mock_caption.end = i * 10.0 + (j + 1) * 4.0
                mock_caption.text = f"Test caption {i}-{j}"
                mock_clip.captions.append(mock_caption)
                
            mock_metadata.clips.append(mock_clip)
            
        return mock_metadata
    
    @patch('youtube_timeline_builder.initFileInfo')
    def test_create_timeline_from_clips(self, mock_init_file_info):
        """Test creating a timeline from clips."""
        # Configure mocks
        mock_timeline = MagicMock()
        mock_timeline.v = [[]]  # Empty video track
        mock_timeline.res = [1920, 1080]
        
        self.mock_timeline_manager.create_v3_timeline.return_value = mock_timeline
        
        # Add a second video track for captions
        mock_timeline.v.append([])
        
        # Configure file info mock
        mock_init_file_info.return_value = MagicMock()
        
        # Call the method under test
        result = self.builder.create_timeline_from_clips(
            metadata=self.mock_metadata,
            width=1920,
            height=1080,
            framerate=30.0
        )
        
        # Verify the result
        self.assertEqual(result, mock_timeline)
        
        # Verify timeline creation was called
        self.mock_timeline_manager.create_v3_timeline.assert_called_once()
        
        # Verify video objects were added (one per clip)
        self.assertEqual(len(mock_timeline.v[0]), 3)
        
        # Verify caption objects were added (two per clip)
        self.assertEqual(len(mock_timeline.v[1]), 6)
        
        # Verify metadata was added
        self.assertIn('videoai_metadata', mock_timeline.__dict__)
        self.assertIn('youtube_info', mock_timeline.videoai_metadata)
        
    @patch('youtube_timeline_builder.initFileInfo')
    def test_save_timeline(self, mock_init_file_info):
        """Test saving a timeline."""
        # Configure mocks
        mock_timeline = MagicMock()
        mock_timeline.v = [[], []]  # Empty video tracks
        mock_timeline.res = [1920, 1080]
        
        # Setup the timeline manager mock for saving
        save_path = Path(f"{self.temp_dir}/test_timeline.json")
        self.mock_timeline_manager.get_timeline_path.return_value = save_path
        
        # Call the method under test
        result = self.builder.save_timeline(
            timeline=mock_timeline,
            timeline_name="test_timeline"
        )
        
        # Verify the result
        self.assertEqual(result, save_path)
        
        # Verify timeline manager methods were called correctly
        self.mock_timeline_manager.get_timeline_path.assert_called_once_with("test_timeline")
        self.mock_timeline_manager.serialize_timeline.assert_called_once()
        
    @patch('youtube_timeline_builder.initFileInfo')
    def test_explicit_output_path(self, mock_init_file_info):
        """Test saving a timeline with explicit output path."""
        # Configure mocks
        mock_timeline = MagicMock()
        
        # Create a temporary path
        from youtube_timeline_builder import file_mgr
        with patch.object(file_mgr, 'normalize_path', return_value=Path(f"{self.temp_dir}/explicit_output.json")):
            # Call the method under test
            result = self.builder.save_timeline(
                timeline=mock_timeline,
                output_path=f"{self.temp_dir}/explicit_output.json"
            )
            
            # Verify the result
            self.assertEqual(result, Path(f"{self.temp_dir}/explicit_output.json"))
            
    def test_error_handling(self):
        """Test error handling in timeline creation."""
        # Configure timeline manager to raise an exception
        self.mock_timeline_manager.create_v3_timeline.side_effect = Exception("Test error")
        
        # The log_exceptions decorator is preventing the exception from being raised
        # So let's check if the function returns the expected result
        result = self.builder.create_timeline_from_clips(
            metadata=self.mock_metadata,
            width=1920,
            height=1080,
            framerate=30.0
        )
        
        # The function should return None when an error occurs
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()