import unittest
import os
import sys
import json
import tempfile
from pathlib import Path
from fractions import Fraction
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the modules to test
from timeline_manager import TimelineManager, TimelineEncoder, V3_TIMELINE_SCHEMA
from auto_editor.timeline import v1, v3, TlVideo, TlAudio
from file_manager import FileManager
import jsonschema

# Create file manager instance
file_mgr = FileManager()

# Create a mock TimelineConfig class for testing
class MockTimelineConfig:
    """Mock timeline configuration for testing without config dependencies."""
    
    def __init__(self):
        # Default configuration values
        self.default_width = 1920
        self.default_height = 1080
        self.default_framerate = 30
        self.default_samplerate = 48000
        self.default_background = "#000000"
        
        # Visualization settings
        self.visualization = MagicMock()
        self.visualization.max_width = 80
        self.visualization.default_detail_level = "normal"
        self.visualization.time_markers = 5
        
        # Serialization settings
        self.serialization = MagicMock()
        self.serialization.format = "json"
        self.serialization.compression = False
        self.serialization.auto_backup = True
        self.serialization.validate_schema = True
        self.serialization.include_metadata = True

# Mock the get_timeline_config function for tests
def mock_get_timeline_config(channel_number=None):
    return MockTimelineConfig()

class TestTimelineManager(unittest.TestCase):
    """Test cases for the TimelineManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.timeline_mgr = TimelineManager()
        
        # Create test data directory if it doesn't exist
        self.test_output_dir = Path(__file__).parent / "test_output"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
    def test_serialize_deserialize_v3(self):
        """Test serialization and deserialization of v3 timeline."""
        # Create a simple v3 timeline
        timeline = self.timeline_mgr.create_v3_timeline(
            width=1080,
            height=1920,
            framerate=30
        )
        
        # Define test file path
        test_file = self.test_output_dir / "test_timeline_v3.json"
        
        # Test serialization
        result = self.timeline_mgr.serialize_timeline(timeline, test_file)
        self.assertIsInstance(result, dict)
        self.assertTrue(test_file.exists())
        
        # Test deserialization
        loaded_timeline = self.timeline_mgr.deserialize_timeline(test_file)
        self.assertIsInstance(loaded_timeline, v3)
        
        # Verify timeline properties
        self.assertEqual(loaded_timeline.res, (1080, 1920))
        
    def test_visualization(self):
        """Test timeline visualization."""
        # Create a simple v3 timeline
        timeline = self.timeline_mgr.create_v3_timeline(
            width=1080,
            height=1920,
            framerate=30
        )
        
        # Test visualization
        viz = self.timeline_mgr.visualize_timeline(timeline)
        self.assertIsInstance(viz, str)
        self.assertIn("Timeline v3", viz)
        
        # Test visualization with different detail levels
        minimal_viz = self.timeline_mgr.visualize_timeline(timeline, detail_level="minimal")
        self.assertLess(len(minimal_viz), len(viz))
        
        detailed_viz = self.timeline_mgr.visualize_timeline(timeline, detail_level="detailed")
        self.assertGreater(len(detailed_viz), len(viz))
        
    def test_clip_sequence_to_timeline(self):
        """Test converting clip sequence to timeline."""
        # Create a sample clip sequence
        clip_sequence = [
            {
                "clip_name": "sample_clip.mp4",
                "start_time": 0,
                "duration": 10,
                "script_segment": "Test segment"
            }
        ]
        
        # Mock the find_video_file function to avoid file existence checks
        original_find_video_file = file_mgr.find_video_file
        try:
            # Create a mock that returns a dummy path
            def mock_find_video_file(path):
                return Path(path)
                
            file_mgr.find_video_file = mock_find_video_file
            
            # Test the conversion with mocked file finder
            # This will fail if actual video access is attempted
            timeline = self.timeline_mgr.clip_sequence_to_timeline(
                clip_sequence,
                output_width=1080,
                output_height=1920,
                framerate=30,
                clips_dir=self.test_output_dir
            )
            
            # Basic validation of the resulting timeline
            self.assertIsInstance(timeline, v3)
            self.assertEqual(timeline.res, (1080, 1920))
            
        finally:
            # Restore the original function
            file_mgr.find_video_file = original_find_video_file
            
    def test_export_visualization(self):
        """Test exporting timeline visualization to a file."""
        # Create a simple v3 timeline
        timeline = self.timeline_mgr.create_v3_timeline(
            width=1080,
            height=1920,
            framerate=30
        )
        
        # Define test file path
        test_viz_file = self.test_output_dir / "test_timeline_viz.txt"
        
        # Test export
        result = self.timeline_mgr.export_timeline_visualization(
            timeline,
            output_path=test_viz_file,
            detail_level="normal"
        )
        
        self.assertTrue(result)
        self.assertTrue(test_viz_file.exists())
        
        # Check content of visualization file
        with open(test_viz_file, 'r') as f:
            content = f.read()
            self.assertIn("Timeline v3", content)

    def test_encoder_resolution_tuple(self):
        """Test TimelineEncoder handling of tuple-based resolution values."""
        # Create a sample structure with tuple
        resolution_tuple = (1280, 720)
        test_data = {"resolution": resolution_tuple}
        
        # Encode the data
        encoded = json.dumps(test_data, cls=TimelineEncoder)
        decoded = json.loads(encoded)
        
        # Verify the tuple was converted to a list
        self.assertIsInstance(decoded["resolution"], list)
        self.assertEqual(decoded["resolution"], [1280, 720])
        
    def test_schema_validates_resolution_formats(self):
        """Test that all supported resolution formats validate against the schema."""
        # Test with integer array
        test_data = {
            "version": "3",
            "timebase": "24/1", 
            "samplerate": 48000,
            "resolution": [1280, 720],
            "background": "#000000",
            "v": [[]],
            "a": [[]],
            "videoai_metadata": {
                "version": "1.0",
                "type": "v3",
                "channel": 1,
                "created_at": "2025-03-19T09:15:00",
                "description": "Test timeline"
            }
        }
        
        # Verify schema validation succeeds with integer array
        try:
            jsonschema.validate(instance=test_data, schema=V3_TIMELINE_SCHEMA)
            schema_valid = True
        except jsonschema.exceptions.ValidationError as e:
            schema_valid = False
            print(f"Schema validation failed for integer array: {e}")
            
        self.assertTrue(schema_valid, "Schema validation should succeed for integer array resolution")
        
        # Test with float array
        test_data["resolution"] = [1280.0, 720.0]
        
        # Verify schema validation succeeds with float array
        try:
            jsonschema.validate(instance=test_data, schema=V3_TIMELINE_SCHEMA)
            schema_valid = True
        except jsonschema.exceptions.ValidationError as e:
            schema_valid = False
            print(f"Schema validation failed for float array: {e}")
            
        self.assertTrue(schema_valid, "Schema validation should succeed for float array resolution")
        
        # Test with object (for backward compatibility)
        test_data["resolution"] = {"width": 1280, "height": 720}
        
        # Verify schema validation succeeds with object
        try:
            jsonschema.validate(instance=test_data, schema=V3_TIMELINE_SCHEMA)
            schema_valid = True
        except jsonschema.exceptions.ValidationError as e:
            schema_valid = False
            print(f"Schema validation failed for object resolution: {e}")
            
        self.assertTrue(schema_valid, "Schema validation should succeed for object resolution")
        
    def test_v3_timeline_direct_serialization_with_resolution(self):
        """Test direct serialization/deserialization of v3 timeline with resolution."""
        # Create a minimal v3 timeline
        timeline = v3(
            src=None,
            tb=Fraction(24, 1),
            sr=48000,
            res=(1280, 720), 
            background="#000000",
            v=[[]],
            a=[[]],
            v1=None
        )
        
        # Convert to dictionary and serialize with TimelineEncoder
        timeline_dict = timeline.as_dict()
        timeline_dict['videoai_metadata'] = {
            'channel': 1,
            'version': '1.0',
            'type': 'v3',
            'created_at': '2025-03-19T09:30:00',
            'description': 'Test timeline'
        }
        
        # Serialize to JSON
        json_str = json.dumps(timeline_dict, cls=TimelineEncoder, indent=2)
        decoded_data = json.loads(json_str)
        
        # Verify resolution is correctly serialized
        self.assertIsInstance(decoded_data["resolution"], list)
        self.assertEqual(decoded_data["resolution"], [1280, 720])
        
        # Test schema validation on the serialized data
        try:
            jsonschema.validate(instance=decoded_data, schema=V3_TIMELINE_SCHEMA)
            schema_valid = True
        except jsonschema.exceptions.ValidationError as e:
            schema_valid = False
            print(f"Schema validation failed: {e}")
            
        self.assertTrue(schema_valid, "Schema validation should succeed for serialized timeline")
        
        # Mock v3 constructor for direct testing of resolution handling
        orig_v3_init = v3.__init__
        try:
            # Store the received resolution value for testing
            received_res = None
            
            def mock_v3_init(self, src=None, tb=None, sr=None, res=None, background=None, v=None, a=None, v1=None):
                nonlocal received_res
                received_res = res
                orig_v3_init(self, src, tb, sr, res, background, v, a, v1)
                
            v3.__init__ = mock_v3_init
            
            # Create timeline manager instance
            timeline_mgr = TimelineManager()
            
            # Use our resolution handling logic directly
            if isinstance(decoded_data["resolution"], list) and len(decoded_data["resolution"]) == 2:
                resolution_tuple = tuple(int(x) if isinstance(x, (int, float)) else x 
                                        for x in decoded_data["resolution"])
            else:
                resolution_tuple = (1920, 1080)
                
            # Verify the conversion logic produces correct tuple
            self.assertEqual(resolution_tuple, (1280, 720))
            self.assertIsInstance(resolution_tuple, tuple)
            
            # Create v3 timeline with the resolution tuple
            new_timeline = v3(
                src=None,
                tb=Fraction(24, 1),
                sr=48000,
                res=resolution_tuple,
                background="#000000",
                v=[[]],
                a=[[]],
                v1=None
            )
            
            # Verify the resolution was correctly set
            self.assertEqual(received_res, (1280, 720))
            self.assertIsInstance(received_res, tuple)
            
        finally:
            # Restore original v3 init
            v3.__init__ = orig_v3_init

if __name__ == "__main__":
    unittest.main()