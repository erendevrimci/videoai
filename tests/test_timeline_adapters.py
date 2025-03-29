"""
Unit tests for timeline adapters module.

Tests the functionality of the timeline_adapters module, which converts between
VideoAI timeline structures and the formats required by auto-editor's export modules.
"""

import unittest
import json
import tempfile
from pathlib import Path
from fractions import Fraction

# Import timeline adapters
from timeline_adapters import (
    TimelineAdapter, 
    JSONAdapter, 
    ProfessionalFormatAdapter,
    FCP7Adapter,
    FCP11Adapter,
    ShotcutAdapter,
    AdapterFactory
)

# Import auto-editor components
from auto_editor.timeline import v1, v3, TlVideo, TlAudio, TlImage, TlRect
from auto_editor.utils.log import Log

# Import VideoAI components
from timeline_manager import TimelineManager, TlText


class TestTimelineAdapter(unittest.TestCase):
    """Test the base TimelineAdapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = TimelineAdapter()
        
        # Create a dummy timeline with minimal structure
        self.v1_timeline = self._create_v1_timeline()
        self.v3_timeline = self._create_v3_timeline()
    
    def _create_v1_timeline(self):
        """Create a dummy v1 timeline for testing."""
        # Create a minimal v1 timeline with required attributes
        timeline = type('DummyV1Timeline', (), {})()
        timeline.source = type('DummySource', (), {})()
        timeline.source.path = Path("/tmp/test.mp4")
        timeline.chunks = [(0, 100, 1.0)]
        timeline.videoai_metadata = {
            "version": "1.0",
            "type": "v1",
            "description": "Test timeline"
        }
        return timeline
    
    def _create_v3_timeline(self):
        """Create a dummy v3 timeline for testing."""
        # Create a minimal v3 timeline with required attributes
        timeline = type('DummyV3Timeline', (), {})()
        timeline.src = type('DummySource', (), {})()
        timeline.src.path = Path("/tmp/test.mp4")
        timeline.tb = Fraction(30, 1)
        timeline.sr = 48000
        timeline.res = (1920, 1080)
        timeline.background = "#000000"
        
        # Create video tracks
        timeline.v = []
        video_track = []
        video_clip = type('DummyVideoClip', (), {})()
        video_clip.src = timeline.src
        video_clip.start = 0
        video_clip.dur = 100
        video_track.append(video_clip)
        timeline.v.append(video_track)
        
        # Create audio tracks
        timeline.a = []
        audio_track = []
        audio_clip = type('DummyAudioClip', (), {})()
        audio_clip.src = timeline.src
        audio_clip.start = 0
        audio_clip.dur = 100
        audio_track.append(audio_clip)
        timeline.a.append(audio_track)
        
        # Add metadata
        timeline.videoai_metadata = {
            "version": "1.0",
            "type": "v3",
            "description": "Test timeline"
        }
        
        return timeline
    
    def test_normalize_paths_v1(self):
        """Test normalizing paths in a v1 timeline."""
        normalized = self.adapter._normalize_paths(self.v1_timeline)
        self.assertTrue(hasattr(normalized.source, 'path'))
        self.assertEqual(str(normalized.source.path), "/tmp/test.mp4")
        
    def test_normalize_paths_v3(self):
        """Test normalizing paths in a v3 timeline."""
        normalized = self.adapter._normalize_paths(self.v3_timeline)
        self.assertTrue(hasattr(normalized.src, 'path'))
        self.assertEqual(str(normalized.src.path), "/tmp/test.mp4")
    
    def test_preserve_metadata_v1(self):
        """Test preserving metadata from a v1 timeline."""
        metadata = self.adapter.preserve_metadata(self.v1_timeline, "json")
        self.assertEqual(metadata["videoai_export"]["format"], "json")
        self.assertEqual(metadata["videoai_export"]["original_metadata"]["version"], "1.0")
        self.assertEqual(metadata["videoai_export"]["original_metadata"]["description"], "Test timeline")
    
    def test_preserve_metadata_v3(self):
        """Test preserving metadata from a v3 timeline."""
        metadata = self.adapter.preserve_metadata(self.v3_timeline, "fcp7")
        self.assertEqual(metadata["videoai_export"]["format"], "fcp7")
        self.assertEqual(metadata["videoai_export"]["original_metadata"]["version"], "1.0")
        self.assertEqual(metadata["videoai_export"]["original_metadata"]["description"], "Test timeline")
    
    def test_find_media_files_v1(self):
        """Test finding media files in a v1 timeline."""
        media_files = self.adapter.find_media_files(self.v1_timeline)
        self.assertEqual(len(media_files), 1)
        self.assertEqual(str(list(media_files)[0]), "/tmp/test.mp4")
    
    def test_find_media_files_v3(self):
        """Test finding media files in a v3 timeline."""
        media_files = self.adapter.find_media_files(self.v3_timeline)
        self.assertEqual(len(media_files), 1)
        self.assertEqual(str(list(media_files)[0]), "/tmp/test.mp4")


class TestJSONAdapter(unittest.TestCase):
    """Test the JSONAdapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = JSONAdapter()
        
        # Create a dummy timeline with minimal structure
        self.v1_timeline = self._create_v1_timeline()
        self.v3_timeline = self._create_v3_timeline()
    
    def _create_v1_timeline(self):
        """Create a dummy v1 timeline for testing."""
        # Create a minimal v1 timeline with required attributes
        timeline = type('DummyV1Timeline', (), {})()
        timeline.source = type('DummySource', (), {})()
        timeline.source.path = Path("/tmp/test.mp4")
        timeline.chunks = [(0, 100, 1.0)]
        timeline.videoai_metadata = {
            "version": "1.0",
            "type": "v1",
            "description": "Test timeline"
        }
        return timeline
    
    def _create_v3_timeline(self):
        """Create a dummy v3 timeline for testing."""
        # Create a minimal v3 timeline with required attributes
        timeline = type('DummyV3Timeline', (), {})()
        timeline.src = type('DummySource', (), {})()
        timeline.src.path = Path("/tmp/test.mp4")
        timeline.tb = Fraction(30, 1)
        timeline.sr = 48000
        timeline.res = (1920, 1080)
        timeline.background = "#000000"
        
        # Create video tracks
        timeline.v = []
        video_track = []
        video_clip = type('DummyVideoClip', (), {})()
        video_clip.src = timeline.src
        video_clip.start = 0
        video_clip.dur = 100
        video_track.append(video_clip)
        timeline.v.append(video_track)
        
        # Create audio tracks
        timeline.a = []
        audio_track = []
        audio_clip = type('DummyAudioClip', (), {})()
        audio_clip.src = timeline.src
        audio_clip.start = 0
        audio_clip.dur = 100
        audio_track.append(audio_clip)
        timeline.a.append(audio_track)
        
        # Add metadata
        timeline.videoai_metadata = {
            "version": "1.0",
            "type": "v3",
            "description": "Test timeline"
        }
        
        return timeline
    
    def test_adapt_timeline_v1(self):
        """Test adapting a v1 timeline for JSON export."""
        adapted, metadata = self.adapter.adapt_timeline(self.v1_timeline)
        self.assertEqual(adapted, self.v1_timeline)  # JSON adapter doesn't modify the timeline
        self.assertEqual(metadata["videoai_export"]["format"], "json")
        self.assertEqual(metadata["videoai_export"]["format_details"]["supports_custom_elements"], True)
        self.assertEqual(metadata["videoai_export"]["timeline_type"], "v1")
    
    def test_adapt_timeline_v3(self):
        """Test adapting a v3 timeline for JSON export."""
        adapted, metadata = self.adapter.adapt_timeline(self.v3_timeline)
        self.assertEqual(adapted, self.v3_timeline)  # JSON adapter doesn't modify the timeline
        self.assertEqual(metadata["videoai_export"]["format"], "json")
        self.assertEqual(metadata["videoai_export"]["format_details"]["supports_custom_elements"], True)
        self.assertEqual(metadata["videoai_export"]["timeline_type"], "v3")
    
    def test_inject_metadata(self):
        """Test injecting metadata into a timeline dictionary."""
        timeline_dict = {"version": "3", "timebase": "30/1", "v": [], "a": []}
        metadata = {"test": "metadata"}
        
        result = self.adapter.inject_metadata(timeline_dict, metadata)
        self.assertEqual(result["videoai_metadata"], metadata)


class TestProfessionalFormatAdapter(unittest.TestCase):
    """Test the ProfessionalFormatAdapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = ProfessionalFormatAdapter()
        
        # Create a dummy v3 timeline with text elements
        self.v3_timeline = self._create_v3_timeline_with_text()
    
    def _create_v3_timeline_with_text(self):
        """Create a dummy v3 timeline with text elements for testing."""
        # Create a minimal v3 timeline with required attributes
        timeline = type('DummyV3Timeline', (), {})()
        timeline.src = type('DummySource', (), {})()
        timeline.src.path = Path("/tmp/test.mp4")
        timeline.tb = Fraction(30, 1)
        timeline.sr = 48000
        timeline.res = (1920, 1080)
        timeline.background = "#000000"
        
        # Create video tracks
        timeline.v = []
        
        # Track with video elements
        video_track = []
        video_clip = type('DummyVideoClip', (), {})()
        video_clip.src = timeline.src
        video_clip.start = 0
        video_clip.dur = 100
        video_track.append(video_clip)
        
        # Add a text element
        text_clip = TlText(
            start=50,
            dur=30,
            text="Test Text",
            x=100,
            y=100
        )
        video_track.append(text_clip)
        
        timeline.v.append(video_track)
        
        # Create audio tracks
        timeline.a = []
        audio_track = []
        audio_clip = type('DummyAudioClip', (), {})()
        audio_clip.src = timeline.src
        audio_clip.start = 0
        audio_clip.dur = 100
        audio_track.append(audio_clip)
        timeline.a.append(audio_track)
        
        return timeline
    
    def test_convert_custom_elements(self):
        """Test converting custom elements in a v3 timeline."""
        converted = self.adapter._convert_custom_elements(self.v3_timeline)
        
        # Check that text elements were converted to rectangles
        self.assertEqual(len(converted.v[0]), 2)  # Should still have 2 elements
        self.assertIsInstance(converted.v[0][0], type(self.v3_timeline.v[0][0]))  # First element should be unchanged
        self.assertIsInstance(converted.v[0][1], TlRect)  # Second element should now be a rectangle
        
        # Verify the rectangle properties
        rect = converted.v[0][1]
        self.assertEqual(rect.start, 50)
        self.assertEqual(rect.dur, 30)
        self.assertEqual(rect.x, 100)
        self.assertEqual(rect.y, 100)


class TestFCP7Adapter(unittest.TestCase):
    """Test the FCP7Adapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = FCP7Adapter()
        
        # Create a dummy v3 timeline
        self.v3_timeline = self._create_v3_timeline()
    
    def _create_v3_timeline(self):
        """Create a dummy v3 timeline for testing."""
        # Create a minimal v3 timeline with required attributes
        timeline = type('DummyV3Timeline', (), {})()
        timeline.src = type('DummySource', (), {})()
        timeline.src.path = Path("/tmp/test.mp4")
        timeline.tb = Fraction(30, 1)
        timeline.sr = 48000
        timeline.res = (1920, 1080)
        timeline.background = "#000000"
        
        # Create video tracks
        timeline.v = []
        video_track = []
        video_clip = type('DummyVideoClip', (), {})()
        video_clip.src = timeline.src
        video_clip.start = 0
        video_clip.dur = 100
        video_track.append(video_clip)
        timeline.v.append(video_track)
        
        # Create audio tracks
        timeline.a = []
        audio_track = []
        audio_clip = type('DummyAudioClip', (), {})()
        audio_clip.src = timeline.src
        audio_clip.start = 0
        audio_clip.dur = 100
        audio_track.append(audio_clip)
        timeline.a.append(audio_track)
        
        return timeline
    
    def test_adapt_timeline(self):
        """Test adapting a v3 timeline for FCP7 export."""
        adapted, metadata = self.adapter.adapt_timeline(self.v3_timeline)
        self.assertEqual(metadata["videoai_export"]["format"], "fcp7")
        self.assertEqual(metadata["videoai_export"]["format_details"]["premiere_compatible"], True)
        self.assertEqual(metadata["videoai_export"]["format_details"]["supports_speed_effects"], True)


class TestFCP11Adapter(unittest.TestCase):
    """Test the FCP11Adapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = FCP11Adapter()
        
        # Create a dummy v3 timeline
        self.v3_timeline = self._create_v3_timeline()
    
    def _create_v3_timeline(self):
        """Create a dummy v3 timeline for testing."""
        # Create a minimal v3 timeline with required attributes
        timeline = type('DummyV3Timeline', (), {})()
        timeline.src = type('DummySource', (), {})()
        timeline.src.path = Path("/tmp/test.mp4")
        timeline.tb = Fraction(30, 1)
        timeline.sr = 48000
        timeline.res = (1920, 1080)
        timeline.background = "#000000"
        
        # Create video tracks
        timeline.v = []
        video_track = []
        video_clip = type('DummyVideoClip', (), {})()
        video_clip.src = timeline.src
        video_clip.start = 0
        video_clip.dur = 100
        video_track.append(video_clip)
        timeline.v.append(video_track)
        
        # Create audio tracks
        timeline.a = []
        audio_track = []
        audio_clip = type('DummyAudioClip', (), {})()
        audio_clip.src = timeline.src
        audio_clip.start = 0
        audio_clip.dur = 100
        audio_track.append(audio_clip)
        timeline.a.append(audio_track)
        
        return timeline
    
    def test_adapt_timeline(self):
        """Test adapting a v3 timeline for FCP11 export."""
        adapted, metadata = self.adapter.adapt_timeline(self.v3_timeline)
        self.assertEqual(metadata["videoai_export"]["format"], "fcp11")
        self.assertEqual(metadata["videoai_export"]["format_details"]["fcpx_version"], 5)
        self.assertEqual(metadata["videoai_export"]["format_details"]["supports_roles"], True)
        self.assertEqual(metadata["videoai_export"]["format_details"]["supports_markers"], True)


class TestShotcutAdapter(unittest.TestCase):
    """Test the ShotcutAdapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = ShotcutAdapter()
        
        # Create a dummy v3 timeline
        self.v3_timeline = self._create_v3_timeline()
    
    def _create_v3_timeline(self):
        """Create a dummy v3 timeline for testing."""
        # Create a minimal v3 timeline with required attributes
        timeline = type('DummyV3Timeline', (), {})()
        timeline.src = type('DummySource', (), {})()
        timeline.src.path = Path("/tmp/test.mp4")
        timeline.tb = Fraction(30, 1)
        timeline.sr = 48000
        timeline.res = (1920, 1080)
        timeline.background = "#000000"
        
        # Create video tracks
        timeline.v = []
        video_track = []
        video_clip = type('DummyVideoClip', (), {})()
        video_clip.src = timeline.src
        video_clip.start = 0
        video_clip.dur = 100
        video_track.append(video_clip)
        timeline.v.append(video_track)
        
        # Create audio tracks
        timeline.a = []
        audio_track = []
        audio_clip = type('DummyAudioClip', (), {})()
        audio_clip.src = timeline.src
        audio_clip.start = 0
        audio_clip.dur = 100
        audio_track.append(audio_clip)
        timeline.a.append(audio_track)
        
        return timeline
    
    def test_adapt_timeline(self):
        """Test adapting a v3 timeline for Shotcut export."""
        adapted, metadata = self.adapter.adapt_timeline(self.v3_timeline)
        self.assertEqual(metadata["videoai_export"]["format"], "shotcut")
        self.assertEqual(metadata["videoai_export"]["format_details"]["mlt_version"], "7.0")
        self.assertEqual(metadata["videoai_export"]["format_details"]["supports_filters"], True)


class TestAdapterFactory(unittest.TestCase):
    """Test the AdapterFactory class."""
    
    def test_create_json_adapter(self):
        """Test creating a JSON adapter."""
        adapter = AdapterFactory.create_adapter("json")
        self.assertIsInstance(adapter, JSONAdapter)
    
    def test_create_fcp7_adapter(self):
        """Test creating an FCP7 adapter."""
        adapter = AdapterFactory.create_adapter("fcp7")
        self.assertIsInstance(adapter, FCP7Adapter)
    
    def test_create_fcp11_adapter(self):
        """Test creating an FCP11 adapter."""
        adapter = AdapterFactory.create_adapter("fcp11")
        self.assertIsInstance(adapter, FCP11Adapter)
    
    def test_create_shotcut_adapter(self):
        """Test creating a Shotcut adapter."""
        adapter = AdapterFactory.create_adapter("shotcut")
        self.assertIsInstance(adapter, ShotcutAdapter)
    
    def test_create_invalid_adapter(self):
        """Test creating an adapter for an invalid format."""
        with self.assertRaises(ValueError):
            AdapterFactory.create_adapter("invalid_format")


if __name__ == '__main__':
    unittest.main()