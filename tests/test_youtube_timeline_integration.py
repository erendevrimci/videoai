"""
Integration tests for YouTube timeline generation.

Tests the end-to-end workflow of downloading YouTube videos, 
processing clips, and generating timelines.
"""
import unittest
import tempfile
import shutil
import os
from pathlib import Path
import json

# Import modules to test
from yt_clips import process_video, ProcessingOptions, WhisperModelSize
from youtube_timeline_builder import YouTubeTimelineBuilder
from timeline_manager import TimelineManager


class TestYouTubeTimelineIntegration(unittest.TestCase):
    """Integration tests for YouTube timeline generation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temp directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.download_dir = Path(self.temp_dir) / "downloads"
        self.output_dir.mkdir()
        self.download_dir.mkdir()
        
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    @unittest.skip("Integration test requiring network access - run manually")
    def test_end_to_end_timeline_generation(self):
        """Test the entire timeline generation process with a real video."""
        # Create processing options
        options = ProcessingOptions(
            max_results=1,
            output_directory=self.output_dir,
            download_directory=self.download_dir,
            scene_threshold=30.0,
            whisper_model=WhisperModelSize.TINY,  # Use tiny model for faster testing
            min_clip_duration=1.0,
            skip_captions=False,
            generate_timeline=True,
            timeline_name="test_timeline",
            timeline_width=1280,
            timeline_height=720,
            timeline_framerate=30.0
        )
        
        # Process a short video (use a known short video for testing)
        result = process_video("NASA Solar System short", options)
        
        # Verify we got the expected return value structure
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)  # metadata_path, metadata, timeline_path
        
        json_path, metadata, timeline_path = result
        
        # Verify the timeline file exists
        self.assertTrue(timeline_path.exists())
        
        # Load the timeline and verify structure
        timeline_mgr = TimelineManager()
        timeline = timeline_mgr.deserialize_timeline(timeline_path)
        
        # Verify basic timeline properties
        self.assertIsNotNone(timeline)
        self.assertEqual(timeline.res[0], 1280)
        self.assertEqual(timeline.res[1], 720)
        
        # Verify tracks exist with content
        self.assertTrue(len(timeline.v) >= 2)  # At least video and caption tracks
        self.assertTrue(len(timeline.v[0]) > 0)  # Video track has clips
        
        # Verify metadata
        self.assertIn('videoai_metadata', timeline.__dict__)
        self.assertIn('youtube_info', timeline.videoai_metadata)
        self.assertEqual(timeline.videoai_metadata['youtube_info']['video_id'], metadata.video.video_info.video_id)
        
    def test_builder_with_existing_metadata(self):
        """Test using the YouTubeTimelineBuilder with pre-existing metadata."""
        # Create a mock metadata file in the temp directory
        metadata_path = Path(self.temp_dir) / "test_metadata.json"
        
        # Define a minimal metadata structure
        metadata_dict = {
            "video": {
                "id": "test123",
                "title": "Test Video",
                "url": "https://www.youtube.com/watch?v=test123",
                "duration": "10:00",
                "views": "1000",
                "thumbnail": "https://example.com/thumbnail.jpg",
                "video_info": {
                    "video_id": "test123",
                    "title": "Test Video",
                    "length": 600,
                    "author": "Test Author",
                    "description": "Test Description",
                    "upload_date": "20240101"
                }
            },
            "video_info": {
                "video_id": "test123",
                "title": "Test Video",
                "length": 600,
                "author": "Test Author",
                "description": "Test Description",
                "upload_date": "20240101"
            },
            "download_date": "2024-01-01T12:00:00",
            "clips": [
                {
                    "path": str(Path(self.temp_dir) / "clip1.mp4"),
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "duration": 10.0,
                    "captions": [
                        {
                            "start": 0.0,
                            "end": 5.0,
                            "text": "Test caption 1"
                        },
                        {
                            "start": 5.0,
                            "end": 10.0,
                            "text": "Test caption 2"
                        }
                    ]
                }
            ],
            "processor_version": "1.0.0"
        }
        
        # Create a dummy clip file so path validation passes
        with open(Path(self.temp_dir) / "clip1.mp4", 'w') as f:
            f.write("dummy file")
        
        # Write the metadata to a file
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f)
        
        # Skip the actual test if we're just checking for syntax
        if not os.path.exists(Path(self.temp_dir) / "clip1.mp4"):
            self.skipTest("Skipping test since we're just checking for syntax")
            
        # Create a timeline builder with mock components
        timeline_manager = unittest.mock.MagicMock()
        builder = YouTubeTimelineBuilder(timeline_manager=timeline_manager)
        
        # Configure mocks
        mock_timeline = unittest.mock.MagicMock()
        mock_timeline.v = [[], []]  # Two empty tracks
        mock_timeline.res = [1920, 1080]
        timeline_manager.create_v3_timeline.return_value = mock_timeline
        
        # Create a timeline with mock metadata
        mock_metadata = unittest.mock.MagicMock()
        mock_metadata.video = unittest.mock.MagicMock()
        mock_metadata.video.title = "Test Video"
        mock_metadata.video.url = "https://www.youtube.com/watch?v=test123"
        mock_metadata.video.video_info = unittest.mock.MagicMock()
        mock_metadata.video.video_info.video_id = "test123"
        mock_metadata.processor_version = "1.0.0"
        
        # Create a mock clip
        mock_clip = unittest.mock.MagicMock()
        mock_clip.path = Path(self.temp_dir) / "clip1.mp4"
        mock_clip.start_time = 0.0
        mock_clip.end_time = 10.0
        mock_clip.duration = 10.0
        
        # Add mock captions
        mock_clip.captions = []
        mock_caption1 = unittest.mock.MagicMock()
        mock_caption1.start = 0.0
        mock_caption1.end = 5.0
        mock_caption1.text = "Test caption 1"
        mock_caption2 = unittest.mock.MagicMock()
        mock_caption2.start = 5.0
        mock_caption2.end = 10.0
        mock_caption2.text = "Test caption 2"
        mock_clip.captions = [mock_caption1, mock_caption2]
        
        # Add the clip to the metadata
        mock_metadata.clips = [mock_clip]
        
        # Create the timeline with patch for file info
        with unittest.mock.patch('youtube_timeline_builder.initFileInfo') as mock_init_file_info:
            # Configure the mock to return a mock file info
            mock_file_info = unittest.mock.MagicMock()
            mock_init_file_info.return_value = mock_file_info
            
            # Create the timeline
            timeline = builder.create_timeline_from_clips(mock_metadata)
            
            # Verify the timeline was created
            self.assertIsNotNone(timeline)
            
            # Verify timeline manager was called
            timeline_manager.create_v3_timeline.assert_called_once()


if __name__ == '__main__':
    unittest.main()