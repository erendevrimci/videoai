import unittest
import os
import sys
import json
import tempfile
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the modules to test
from config import (
    TimelineConfig, TimelineRenderingConfig, TimelineSerializationConfig,
    TimelineVisualizationConfig, get_timeline_config, config
)

class TestTimelineConfig(unittest.TestCase):
    """Test cases for the TimelineConfig class and related configurations."""
    
    def test_default_timeline_config(self):
        """Test that default timeline configuration loads correctly."""
        # Get default configuration
        timeline_config = config.timeline
        
        # Check basic properties
        self.assertIsInstance(timeline_config, TimelineConfig)
        self.assertEqual(timeline_config.storage_directory, "timelines")
        self.assertEqual(timeline_config.default_framerate, 30)
        self.assertEqual(timeline_config.default_width, 1080)
        self.assertEqual(timeline_config.default_height, 1920)
        
        # Check nested configurations
        self.assertIsInstance(timeline_config.visualization, TimelineVisualizationConfig)
        self.assertIsInstance(timeline_config.serialization, TimelineSerializationConfig)
        self.assertIsInstance(timeline_config.rendering, TimelineRenderingConfig)
        
        # Check default rendering settings
        self.assertFalse(timeline_config.rendering.enabled)
        self.assertTrue(timeline_config.rendering.compatibility_mode)
        self.assertFalse(timeline_config.rendering.force_fallback)
        
    def test_channel_specific_overrides(self):
        """Test that channel-specific overrides apply correctly."""
        # Get channel 1 configuration (has overrides in config.py)
        channel1_config = get_timeline_config(1)
        
        # Check that default values are overridden for this channel
        self.assertEqual(channel1_config.default_framerate, 24)  # From channel config
        self.assertEqual(channel1_config.default_width, 120)  # From channel config
        self.assertEqual(channel1_config.default_height, 720)  # From channel config
        
        # Default rendering settings should still apply
        self.assertFalse(channel1_config.rendering.enabled)
        self.assertTrue(channel1_config.rendering.compatibility_mode)
        
    def test_feature_flag_overrides(self):
        """Test that feature flags can be overridden dynamically."""
        # Get default timeline config
        timeline_config = get_timeline_config()
        
        # Check initial state
        self.assertFalse(timeline_config.rendering.enabled)
        
        # Override feature flag
        timeline_config.rendering.enabled = True
        self.assertTrue(timeline_config.rendering.enabled)
        
        # Disable compatibility for testing edge cases
        timeline_config.rendering.compatibility_mode = False
        self.assertFalse(timeline_config.rendering.compatibility_mode)
        
        # Force fallback mode
        timeline_config.rendering.force_fallback = True
        self.assertTrue(timeline_config.rendering.force_fallback)
        
        # These changes should not affect the global config
        self.assertFalse(config.timeline.rendering.enabled)
        
    def test_config_deepcopy(self):
        """Test that get_timeline_config creates proper deep copies."""
        # Get two config objects
        config1 = get_timeline_config()
        config2 = get_timeline_config()
        
        # Modify one config
        config1.rendering.enabled = True
        config1.visualization.default_detail_level = "detailed"
        
        # Check that the other config is unaffected
        self.assertFalse(config2.rendering.enabled)
        self.assertEqual(config2.visualization.default_detail_level, "normal")

if __name__ == "__main__":
    unittest.main()