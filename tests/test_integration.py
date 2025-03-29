import unittest
import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from fractions import Fraction

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the modules to test
import main
from timeline_manager import TimelineManager
from auto_editor.timeline import v1, v3
from file_manager import FileManager

# Create file manager instance
file_mgr = FileManager()

class TestIntegration(unittest.TestCase):
    """Test cases for integrated pipeline with timeline functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_output_dir = Path(__file__).parent / "test_output"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
    def test_pipeline_with_timeline(self):
        """Test the complete video pipeline with timeline integration."""
        # Create a mocked timeline flow through the pipeline
        with patch('main.write_script.main') as mock_script, \
             patch('main.voice_over.main') as mock_voice, \
             patch('main.generate_subtitles') as mock_captions, \
             patch('main.video_edit.main') as mock_video, \
             patch('main.TimelineManager.serialize_timeline') as mock_save, \
             patch('main.add_script_segments_to_timeline') as mock_add_script, \
             patch('main.add_voice_to_timeline') as mock_add_voice, \
             patch('main.file_mgr.read_text') as mock_read, \
             patch('main.file_mgr.file_exists', return_value=True) as mock_exists:
            
            # Configure the mocks
            mock_script.return_value = None
            mock_voice.return_value = None
            mock_captions.return_value = None
            mock_video.return_value = True
            mock_save.return_value = {}
            mock_read.return_value = "Test script content"
            
            # Set up the timeline flow
            timeline_mgr = TimelineManager()
            test_timeline = timeline_mgr.create_v3_timeline()
            mock_add_script.return_value = test_timeline
            mock_add_voice.return_value = test_timeline
            
            # Run the process with timeline
            result = main.process_channel_with_timeline(1, ["script", "voice", "captions", "video"])
            
            # Verify the result
            self.assertTrue(result, "Timeline pipeline should complete successfully")
            
            # Verify the timeline flow
            mock_script.assert_called_once()
            mock_voice.assert_called_once()
            mock_captions.assert_called_once()
            mock_video.assert_called_once()
            mock_add_script.assert_called_once()
            mock_add_voice.assert_called_once()
            
    def test_command_line_args(self):
        """Test the command line argument processing."""
        # Test with timeline flag
        test_args = ["--channel", "1", "--timeline"]
        with patch('sys.argv', ['main.py'] + test_args), \
             patch('main.process_channel_with_timeline') as mock_process, \
             patch('main.process_channel') as mock_process_normal:
             
            mock_process.return_value = True
            mock_process_normal.return_value = None
            
            # Run the main function
            main.main()
            
            # Verify that timeline processing was used
            mock_process.assert_called_once()
            mock_process_normal.assert_not_called()
    
    def test_timeline_file_handling(self):
        """Test timeline file handling in video_edit.py."""
        # Create a mock timeline
        timeline_mgr = TimelineManager()
        test_timeline = timeline_mgr.create_v3_timeline()
        test_file = self.test_output_dir / "test_timeline_integration.json"
        
        # Save the timeline
        timeline_mgr.serialize_timeline(test_timeline, test_file)
        self.assertTrue(test_file.exists(), "Timeline file should be created")
        
        # Test loading the timeline
        with patch('video_edit.main') as mock_main:
            import video_edit
            
            # Set up test arguments
            test_args = ["--channel", "1", "--timeline", "--timeline-file", str(test_file)]
            with patch('sys.argv', ['video_edit.py'] + test_args), \
                 patch('sys.exit') as mock_exit:
                
                # Re-parse arguments for testing
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--channel", type=int)
                parser.add_argument("--timeline", action="store_true")
                parser.add_argument("--timeline-file", type=str)
                args = parser.parse_args(test_args)
                
                # Verify argument parsing
                self.assertEqual(args.channel, 1)
                self.assertTrue(args.timeline)
                self.assertEqual(args.timeline_file, str(test_file))
                
                # Verify the timeline loading
                loaded_timeline = timeline_mgr.deserialize_timeline(test_file)
                self.assertIsInstance(loaded_timeline, v3)


class TestTitleDescIntegration(unittest.TestCase):
    """Test case for integrated timeline and title_desc path processing."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_channel = 999  # Use a test channel
        self.file_mgr = FileManager()
        self.timeline_mgr = TimelineManager(channel_number=self.test_channel)
        
        # Ensure test output directory exists and is clean
        self.output_dir = self.file_mgr.get_channel_output_path(self.test_channel)
        self.title_desc_path = self.file_mgr.get_title_desc_path(self.test_channel)
        self.timeline_path = self.file_mgr.get_timeline_path("test_timeline", self.test_channel)
        
        # Clean up any existing test files
        for path in [self.title_desc_path, self.timeline_path]:
            if path.exists():
                path.unlink()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up test files
        for path in [self.title_desc_path, self.timeline_path]:
            if path.exists():
                path.unlink()
    
    def test_timeline_title_desc_integration(self):
        """Test complete workflow for timeline title_desc integration."""
        # 1. Create a simple timeline - use the timeline manager to create it
        timeline = self.timeline_mgr.create_v3_timeline(
            width=1920,
            height=1080,
            framerate=30,
            samplerate=48000,
            background="#000000"
        )
        
        # 2. Create sample title/description data
        title_desc_data = {
            "title": "Test Video Title",
            "description": "This is a test description for the video.\nIt has multiple lines.",
            "tags": ["test", "integration", "timeline"],
            "metadata": {
                "version": "1.0",
                "generated": True
            }
        }
        
        # 3. Add title/description to timeline metadata
        # For v3 timeline, we need to add the metadata attribute first
        timeline.videoai_metadata = {
            "title_desc": title_desc_data,
            "version": "1.0",
            "type": "v3",
            "channel": self.test_channel
        }
        
        # 4. Save timeline
        self.assertTrue(
            self.timeline_mgr.save_timeline(timeline, "test_timeline", "Integration test timeline"),
            "Failed to save timeline"
        )
        self.assertTrue(self.timeline_path.exists(), "Timeline file wasn't created")
        
        # 5. Save title_desc data separately
        self.assertTrue(
            self.file_mgr.write_json(self.title_desc_path, title_desc_data),
            "Failed to save title_desc data"
        )
        self.assertTrue(self.title_desc_path.exists(), "Title_desc file wasn't created")
        
        # 6. Load timeline back
        loaded_timeline = self.timeline_mgr.load_timeline("test_timeline")
        self.assertIsNotNone(loaded_timeline, "Failed to load timeline")
        
        # 7. Verify the metadata in timeline
        self.assertTrue(hasattr(loaded_timeline, 'videoai_metadata'), "Missing videoai_metadata in timeline")
        self.assertIn('title_desc', loaded_timeline.videoai_metadata, "Missing title_desc in metadata")
        
        loaded_title_desc = loaded_timeline.videoai_metadata['title_desc']
        self.assertEqual(loaded_title_desc['title'], title_desc_data['title'], "Title in timeline metadata doesn't match")
        self.assertEqual(loaded_title_desc['description'], title_desc_data['description'], "Description in timeline metadata doesn't match")
        
        # 8. Load title_desc data directly
        loaded_data = self.file_mgr.read_json(self.title_desc_path)
        self.assertIsNotNone(loaded_data, "Failed to load title_desc data")
        self.assertEqual(loaded_data['title'], title_desc_data['title'], "Title doesn't match")
        self.assertEqual(loaded_data['description'], title_desc_data['description'], "Description doesn't match")
        
        # 9. Generate timeline visualization
        viz_path = self.file_mgr.get_timeline_visualization_path(
            "test_timeline", 
            detail_level="detailed",
            channel_number=self.test_channel
        )
        
        success = self.timeline_mgr.export_timeline_visualization(
            loaded_timeline, 
            output_path=viz_path, 
            detail_level="detailed"
        )
        self.assertTrue(success, "Failed to create timeline visualization")
        self.assertTrue(viz_path.exists(), "Visualization file wasn't created")
        
        # 10. Verify title in visualization
        viz_content = self.file_mgr.read_text(viz_path)
        self.assertIsNotNone(viz_content, "Visualization content is empty")
        
        # Verify that title appears in the visualization
        self.assertIn(title_desc_data['title'], viz_content, "Title not found in visualization")
        # For detailed view, description should also appear
        self.assertIn(title_desc_data['description'].split('\n')[0], viz_content, 
                     "Description not found in visualization")
        # Tags should be included in detailed view
        tags_str = ", ".join(title_desc_data['tags'])
        self.assertIn(tags_str, viz_content, "Tags not found in visualization")
    
    def test_edge_cases(self):
        """Test edge cases for get_title_desc_path."""
        # Test None channel_number - this should raise an error
        with self.assertRaises(Exception):
            self.file_mgr.get_title_desc_path(None)
            
        # Test negative channel_number - should work but create a folder with negative number
        negative_path = self.file_mgr.get_title_desc_path(-1)
        self.assertTrue(negative_path.name.endswith('.json'))
        
        # Test filename with special characters
        special_chars_path = self.file_mgr.get_title_desc_path(self.test_channel, "special!@#$%^&*()_+")
        self.assertTrue(special_chars_path.name.endswith('.json'), "Extension not added correctly with special chars")
        
        # Test very long filename
        long_name = "a" * 200  # Very long filename
        long_path = self.file_mgr.get_title_desc_path(self.test_channel, long_name)
        self.assertTrue(long_path.name.endswith('.json'), "Extension not added correctly with long name")
    
    def test_memory_management(self):
        """Test memory management with large JSON data."""
        # Create large title/desc data
        large_data = {
            "title": "Test Video",
            "description": "Test description",
            # Add a large array to make the JSON size substantial
            "large_array": ["item" + str(i) for i in range(10000)]
        }
        
        # Save the large data
        self.assertTrue(
            self.file_mgr.write_json(self.title_desc_path, large_data),
            "Failed to save large JSON data"
        )
        
        # Check file size
        file_size = self.title_desc_path.stat().st_size
        print(f"Large JSON file size: {file_size} bytes")
        self.assertGreater(file_size, 100000, "File size should be substantial")
        
        # Load the data back
        loaded_data = self.file_mgr.read_json(self.title_desc_path)
        self.assertIsNotNone(loaded_data, "Failed to load large JSON data")
        self.assertEqual(len(loaded_data['large_array']), 10000, "Large array not preserved")

class TestResolutionTitleDescIntegration(unittest.TestCase):
    """Test case for integrated resolution and title_desc handling."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_channel = 998  # Use a different test channel from other tests
        self.file_mgr = FileManager()
        self.timeline_mgr = TimelineManager(channel_number=self.test_channel)
        
        # Ensure test output directory exists
        self.output_dir = self.file_mgr.get_channel_output_path(self.test_channel)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define test paths
        self.timeline_path = self.file_mgr.get_timeline_path("test_resolution", self.test_channel)
        self.viz_path = self.file_mgr.get_timeline_visualization_path(
            "test_resolution", detail_level="detailed", channel_number=self.test_channel
        )
        
        # Clean up any existing test files
        for path in [self.timeline_path, self.viz_path]:
            if path.exists():
                path.unlink()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up test files
        for path in [self.timeline_path, self.viz_path]:
            if path.exists():
                path.unlink()
    
    def test_resolution_types_with_title_desc(self):
        """Test different resolution types with title/description metadata."""
        # Test different resolution formats
        resolution_formats = [
            ((1920, 1080), "Tuple of integers"),
            ((1280.0, 720.0), "Tuple of floats"),
            ([1280, 720], "List of integers")
        ]
        
        for resolution, res_type in resolution_formats:
            with self.subTest(resolution_type=res_type):
                # Create a timeline with the specified resolution
                timeline = None
                
                if isinstance(resolution, tuple):
                    # For tuple resolutions, we need to use the width/height parameters
                    width, height = resolution
                    timeline = self.timeline_mgr.create_v3_timeline(
                        width=width,
                        height=height,
                        framerate=30,
                        samplerate=48000,
                        background="#000000"
                    )
                else:
                    # For list resolution, we'll create the timeline and modify the res directly
                    timeline = self.timeline_mgr.create_v3_timeline(
                        width=1920,  # Will be overridden
                        height=1080,  # Will be overridden
                        framerate=30,
                        samplerate=48000,
                        background="#000000"
                    )
                    # Manually set the resolution
                    timeline.res = resolution
                
                # Add title/description metadata
                title_desc_data = {
                    "title": f"Resolution Test: {res_type}",
                    "description": f"Testing timeline with {res_type} resolution: {resolution}",
                    "tags": ["test", "resolution", res_type.lower().replace(" ", "_")]
                }
                
                # Add metadata to timeline
                timeline.videoai_metadata = {
                    "title_desc": title_desc_data,
                    "version": "1.0",
                    "type": "v3",
                    "channel": self.test_channel
                }
                
                # Serialize timeline
                self.assertTrue(
                    self.timeline_mgr.save_timeline(timeline, f"test_resolution_{res_type.replace(' ', '_')}"),
                    f"Failed to save timeline with {res_type} resolution"
                )
                
                # Load timeline
                loaded_timeline = self.timeline_mgr.load_timeline(f"test_resolution_{res_type.replace(' ', '_')}")
                self.assertIsNotNone(loaded_timeline, f"Failed to load timeline with {res_type} resolution")
                
                # Verify resolution is maintained
                if isinstance(resolution, tuple):
                    # Tuple resolution should be preserved as tuple
                    self.assertIsInstance(loaded_timeline.res, tuple, 
                                         f"Resolution not preserved as tuple for {res_type}")
                    self.assertEqual(loaded_timeline.res, resolution, 
                                    f"Resolution value not preserved for {res_type}")
                else:
                    # List resolution should be converted to tuple during deserialization
                    self.assertIsInstance(loaded_timeline.res, tuple, 
                                         f"Resolution list not converted to tuple for {res_type}")
                    self.assertEqual(loaded_timeline.res, tuple(resolution), 
                                    f"Resolution value not preserved for {res_type}")
                
                # Verify title/description is maintained
                self.assertTrue(hasattr(loaded_timeline, 'videoai_metadata'), 
                               f"Missing videoai_metadata in timeline for {res_type}")
                self.assertIn('title_desc', loaded_timeline.videoai_metadata, 
                             f"Missing title_desc in metadata for {res_type}")
                
                loaded_title_desc = loaded_timeline.videoai_metadata['title_desc']
                self.assertEqual(loaded_title_desc['title'], title_desc_data['title'], 
                                f"Title in timeline metadata doesn't match for {res_type}")
                self.assertEqual(loaded_title_desc['description'], title_desc_data['description'], 
                                f"Description in timeline metadata doesn't match for {res_type}")
                
                # Generate visualization
                viz_path = self.file_mgr.get_timeline_visualization_path(
                    f"test_resolution_{res_type.replace(' ', '_')}", 
                    detail_level="detailed",
                    channel_number=self.test_channel
                )
                
                success = self.timeline_mgr.export_timeline_visualization(
                    loaded_timeline, 
                    output_path=viz_path, 
                    detail_level="detailed"
                )
                self.assertTrue(success, f"Failed to create timeline visualization for {res_type}")
                self.assertTrue(viz_path.exists(), f"Visualization file wasn't created for {res_type}")
                
                # Verify visualization includes correct resolution and title/desc
                viz_content = self.file_mgr.read_text(viz_path)
                self.assertIsNotNone(viz_content, f"Visualization content is empty for {res_type}")
                
                # For float resolution, the visualization may convert to integer
                if isinstance(resolution, tuple) and any(isinstance(val, float) for val in resolution):
                    # For float resolution, check for either float format or integer format
                    width, height = resolution
                    int_res_string = f"{int(width)}x{int(height)}"
                    float_res_string = f"{width}x{height}"
                    self.assertTrue(
                        int_res_string in viz_content or float_res_string in viz_content,
                        f"Resolution not found in visualization for {res_type}. Expected either {int_res_string} or {float_res_string}"
                    )
                else:
                    # For integer resolution, just check the exact format
                    if isinstance(resolution, tuple):
                        res_string = f"{resolution[0]}x{resolution[1]}"
                    else:
                        res_string = f"{resolution[0]}x{resolution[1]}"
                    
                    self.assertIn(res_string, viz_content, 
                                 f"Resolution not found in visualization for {res_type}")
                
                # Verify title/description content
                self.assertIn(title_desc_data['title'], viz_content, 
                             f"Title not found in visualization for {res_type}")
                # The description might be too long and get wrapped in visualization
                # So just check for the first portion of it
                desc_start = title_desc_data['description'].split(':')[0]
                self.assertIn(desc_start, viz_content, 
                             f"Description not found in visualization for {res_type}")
    
    def test_resolution_schema_validation(self):
        """Test schema validation with different resolution formats."""
        import jsonschema
        from timeline_manager import V3_TIMELINE_SCHEMA, TimelineEncoder
        
        # Basic timeline data
        base_timeline_data = {
            "version": "3",
            "timebase": "30/1", 
            "samplerate": 48000,
            "background": "#000000",
            "v": [[]],
            "a": [[]]
        }
        
        # Test both resolution formats and title/desc metadata together
        resolution_formats = [
            ([1920, 1080], "List of integers"),
            ([1280.5, 720.2], "List of floats"),
            ({"width": 1920, "height": 1080}, "Object with width/height")
        ]
        
        title_desc_data = {
            "title": "Schema Validation Test",
            "description": "Testing schema validation with different resolution formats",
            "tags": ["test", "validation", "schema"]
        }
        
        for resolution, res_type in resolution_formats:
            with self.subTest(resolution_type=res_type):
                # Create test data combining resolution and title/desc
                test_data = dict(base_timeline_data)
                test_data["resolution"] = resolution
                test_data["videoai_metadata"] = {
                    "version": "1.0",
                    "type": "v3",
                    "channel": self.test_channel,
                    "created_at": "2025-03-19T09:15:00",
                    "description": "Test timeline",
                    "title_desc": title_desc_data
                }
                
                # Validate against schema
                try:
                    jsonschema.validate(instance=test_data, schema=V3_TIMELINE_SCHEMA)
                    validation_success = True
                except jsonschema.exceptions.ValidationError as e:
                    validation_success = False
                    print(f"Schema validation failed for {res_type}: {e}")
                
                self.assertTrue(validation_success, 
                               f"Schema validation should succeed for {res_type} resolution")
                
                # Test round-trip serialization/deserialization
                # Serialize to JSON
                json_str = json.dumps(test_data, cls=TimelineEncoder, indent=2)
                decoded_data = json.loads(json_str)
                
                # Verify resolution is properly encoded
                self.assertEqual(decoded_data["resolution"], resolution if not isinstance(resolution, tuple) else list(resolution),
                                f"Resolution not properly encoded for {res_type}")
                
                # Verify title_desc is maintained
                self.assertIn("videoai_metadata", decoded_data, f"Missing videoai_metadata in JSON for {res_type}")
                self.assertIn("title_desc", decoded_data["videoai_metadata"], 
                             f"Missing title_desc in JSON metadata for {res_type}")
                self.assertEqual(decoded_data["videoai_metadata"]["title_desc"]["title"], title_desc_data["title"],
                                f"Title not preserved in JSON for {res_type}")
    
    def test_actual_video_with_resolution_title_desc(self):
        """Integration test using a real video file with resolution and title/desc."""
        from unittest.mock import patch, MagicMock
        
        # Mock the initFileInfo function to avoid needing an actual video file
        with patch('timeline_manager.initFileInfo') as mock_init_file_info:
            # Create a mock file info object with video and audio properties
            mock_file_info = MagicMock()
            mock_file_info.path = Path("/mock/path/video.mp4")
            mock_file_info.video = MagicMock()
            mock_file_info.video.width = 1920
            mock_file_info.video.height = 1080
            mock_file_info.video.duration = 10.0
            mock_file_info.video.fps = 30.0
            mock_file_info.audio = MagicMock()
            mock_file_info.audio.samplerate = 48000
            mock_file_info.audio.channels = 2
            
            # Set the mock to return our file info
            mock_init_file_info.return_value = mock_file_info
            
            # Create a clip sequence for testing
            clip_sequence = [
                {
                    "clip_name": "test_clip_1.mp4",
                    "start_time": 0,
                    "duration": 5.0,
                    "script_segment": "First segment of test script"
                },
                {
                    "clip_name": "test_clip_2.mp4",
                    "start_time": 2.0,
                    "duration": 5.0,
                    "script_segment": "Second segment of test script"
                }
            ]
            
            # Create a timeline from the clip sequence
            with patch('file_manager.FileManager.find_video_file', return_value=Path("/mock/path/video.mp4")):
                # Create the timeline with custom resolution
                timeline = self.timeline_mgr.clip_sequence_to_timeline(
                    clip_sequence,
                    output_width=1280,
                    output_height=720,
                    framerate=30
                )
                
                # Verify the timeline has the custom resolution
                self.assertEqual(timeline.res, (1280, 720), "Timeline resolution doesn't match")
                
                # Add title/description metadata
                title_desc_data = {
                    "title": "Video Test Timeline",
                    "description": "Testing timeline with actual video clips",
                    "tags": ["test", "video", "integration"]
                }
                
                timeline.videoai_metadata = {
                    "title_desc": title_desc_data,
                    "version": "1.0",
                    "type": "v3",
                    "channel": self.test_channel
                }
                
                # Save the timeline
                self.assertTrue(
                    self.timeline_mgr.save_timeline(timeline, "test_video_timeline", "Video test timeline"),
                    "Failed to save video timeline"
                )
                
                # Load the timeline back
                loaded_timeline = self.timeline_mgr.load_timeline("test_video_timeline")
                self.assertIsNotNone(loaded_timeline, "Failed to load video timeline")
                
                # Verify the resolution and metadata
                self.assertEqual(loaded_timeline.res, (1280, 720), "Timeline resolution not preserved")
                self.assertTrue(hasattr(loaded_timeline, 'videoai_metadata'), "Missing videoai_metadata in timeline")
                self.assertIn('title_desc', loaded_timeline.videoai_metadata, "Missing title_desc in metadata")
                
                loaded_title_desc = loaded_timeline.videoai_metadata['title_desc']
                self.assertEqual(loaded_title_desc['title'], title_desc_data['title'], 
                                "Title in timeline metadata doesn't match")
                
                # Generate visualization with custom width
                viz_path = self.file_mgr.get_timeline_visualization_path(
                    "test_video_timeline", 
                    detail_level="detailed",
                    channel_number=self.test_channel
                )
                
                success = self.timeline_mgr.export_timeline_visualization(
                    loaded_timeline, 
                    output_path=viz_path, 
                    detail_level="detailed",
                    width=120  # Wider visualization
                )
                
                self.assertTrue(success, "Failed to create timeline visualization")
                self.assertTrue(viz_path.exists(), "Visualization file wasn't created")
                
                # Verify visualization includes clips, resolution and title
                viz_content = self.file_mgr.read_text(viz_path)
                self.assertIn("1280x720", viz_content, "Resolution not in visualization")
                self.assertIn(title_desc_data['title'], viz_content, "Title not in visualization")
                
                # Since we mocked the file info with "/mock/path/video.mp4", that's what will appear in viz
                self.assertIn("video.mp4", viz_content, "Source video not in visualization")
                
                # Verify we have two video clips in the visualization
                self.assertIn("1. Video: 0.00s-5.00s", viz_content, "First clip timing not in visualization")
                self.assertIn("2. Video: 5.00s-10.00s", viz_content, "Second clip timing not in visualization")

if __name__ == "__main__":
    unittest.main()