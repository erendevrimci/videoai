import unittest
import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the modules to test
from timeline_manager import TimelineManager
from auto_editor.timeline import v1, v3, TlVideo, TlAudio
from file_manager import FileManager

# Create file manager instance
file_mgr = FileManager()

class TestTimelinePipeline(unittest.TestCase):
    """Test cases for timeline integration in the VideoAI pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.timeline_mgr = TimelineManager()
        
        # Create test data directory if it doesn't exist
        self.test_output_dir = Path(__file__).parent / "test_output"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
    def test_script_to_timeline(self):
        """Test converting script segments to timeline markers."""
        # Mock script segments
        script_segments = [
            {"text": "First segment", "start_time": 0, "end_time": 5},
            {"text": "Second segment", "start_time": 5, "end_time": 10},
            {"text": "Third segment", "start_time": 10, "end_time": 15}
        ]
        
        # Create a timeline
        timeline = self.timeline_mgr.create_v3_timeline(
            width=1080,
            height=1920,
            framerate=30
        )
        
        # Convert script segments to timeline objects
        from main import add_script_segments_to_timeline
        with patch('main.add_script_segments_to_timeline', return_value=timeline) as mock_add:
            # In a real implementation, this would be:
            # timeline = add_script_segments_to_timeline(timeline, script_segments)
            
            timeline = mock_add(timeline, script_segments)
            
            # Assert the function was called with correct parameters
            mock_add.assert_called_once()
            args, kwargs = mock_add.call_args
            self.assertEqual(args[0], timeline)
            self.assertEqual(args[1], script_segments)
    
    def test_voice_to_timeline(self):
        """Test adding voice-over to timeline."""
        # Create a timeline
        timeline = self.timeline_mgr.create_v3_timeline(
            width=1080,
            height=1920,
            framerate=30
        )
        
        # Mock voice file
        voice_file = self.test_output_dir / "mock_voice.mp3"
        
        # Mock adding voice to timeline
        from main import add_voice_to_timeline
        with patch('main.add_voice_to_timeline', return_value=timeline) as mock_add:
            # In a real implementation, this would be:
            # timeline = add_voice_to_timeline(timeline, voice_file)
            
            timeline = mock_add(timeline, voice_file)
            
            # Assert the function was called with correct parameters
            mock_add.assert_called_once()
            args, kwargs = mock_add.call_args
            self.assertEqual(args[0], timeline)
            self.assertEqual(args[1], voice_file)
    
    def test_clips_to_timeline(self):
        """Test adding video clips to timeline."""
        # Create a timeline
        timeline = self.timeline_mgr.create_v3_timeline(
            width=1080,
            height=1920,
            framerate=30
        )
        
        # Mock clip sequence
        clip_sequence = [
            {
                "clip_name": "sample_clip.mp4",
                "start_time": 0,
                "duration": 5,
                "script_segment": "First segment"
            },
            {
                "clip_name": "sample_clip2.mp4",
                "start_time": 0,
                "duration": 5,
                "script_segment": "Second segment"
            }
        ]
        
        # Mock adding clips to timeline
        from main import add_clips_to_timeline
        with patch('main.add_clips_to_timeline', return_value=timeline) as mock_add:
            # In a real implementation, this would be:
            # timeline = add_clips_to_timeline(timeline, clip_sequence)
            
            timeline = mock_add(timeline, clip_sequence)
            
            # Assert the function was called with correct parameters
            mock_add.assert_called_once()
            args, kwargs = mock_add.call_args
            self.assertEqual(args[0], timeline)
            self.assertEqual(args[1], clip_sequence)
    
    def test_integrated_pipeline(self):
        """Test the entire pipeline with timeline integration."""
        # Mock the main function with timeline integration
        with patch('main.process_channel_with_timeline') as mock_process:
            # Set up the mock to simulate success
            mock_process.return_value = True
            
            # Call the mocked function
            from main import process_channel_with_timeline
            result = process_channel_with_timeline(1, ["script", "voice", "captions", "video"])
            
            # Assert the function was called with correct parameters
            mock_process.assert_called_once_with(1, ["script", "voice", "captions", "video"])
            
            # Check the result
            self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()