"""
Test module for the ExportManager
"""
import os
import sys
import unittest
from pathlib import Path
from fractions import Fraction

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from export_manager import ExportManager, FORMAT_JSON, FORMAT_FCP7, FORMAT_FCP11, FORMAT_SHOTCUT
from timeline_manager import TimelineManager, TlText
from auto_editor.timeline import v1, v3, TlVideo, TlAudio
from auto_editor.ffwrapper import initFileInfo

class TestExportManager(unittest.TestCase):
    """Tests for the ExportManager class"""
    
    def setUp(self):
        """Set up for each test"""
        self.export_manager = ExportManager()
        self.timeline_manager = TimelineManager()
        self.test_output_dir = Path("tests/test_output")
        
        # Create test output directory if it doesn't exist
        if not self.test_output_dir.exists():
            os.makedirs(self.test_output_dir)
    
    def test_get_supported_formats(self):
        """Test getting supported formats"""
        formats = self.export_manager.get_supported_formats()
        self.assertIn(FORMAT_JSON, formats)
        self.assertIn(FORMAT_FCP7, formats)
        self.assertIn(FORMAT_FCP11, formats)
        self.assertIn(FORMAT_SHOTCUT, formats)
    
    def test_get_format_info(self):
        """Test getting format information"""
        json_info = self.export_manager.get_format_info(FORMAT_JSON)
        self.assertEqual(json_info["name"], "JSON")
        self.assertTrue(json_info["supports_v1"])
        self.assertTrue(json_info["supports_v3"])
        
        fcp7_info = self.export_manager.get_format_info(FORMAT_FCP7)
        self.assertEqual(fcp7_info["name"], "Final Cut Pro 7 XML")
        self.assertFalse(fcp7_info["supports_v1"])
        self.assertTrue(fcp7_info["supports_v3"])
    
    def test_export_to_json(self):
        """Test exporting a timeline to JSON format"""
        # Create a simple v3 timeline
        timeline = self._create_test_v3_timeline()
        
        # Export to JSON
        output_path = self.test_output_dir / "test_export.json"
        result_path = self.export_manager.export_to_json(timeline, output_path)
        
        # Verify the export was successful
        self.assertIsNotNone(result_path)
        self.assertTrue(output_path.exists())
        
        # Verify we can load the exported timeline
        loaded_timeline = self.timeline_manager.deserialize_timeline(output_path)
        self.assertIsNotNone(loaded_timeline)
        self.assertIsInstance(loaded_timeline, v3)
    
    def test_get_timeline_info(self):
        """Test getting timeline information"""
        # Create a v3 timeline
        v3_timeline = self._create_test_v3_timeline()
        
        # Get timeline info
        v3_info = self.export_manager.get_timeline_info(v3_timeline)
        self.assertEqual(v3_info["status"], "ok")
        self.assertEqual(v3_info["type"], "v3")
        self.assertIn(FORMAT_JSON, v3_info["compatible_formats"])
        self.assertIn(FORMAT_FCP7, v3_info["compatible_formats"])
        
        # Create a v1 timeline
        v1_timeline = self._create_test_v1_timeline()
        
        # Get timeline info
        v1_info = self.export_manager.get_timeline_info(v1_timeline)
        self.assertEqual(v1_info["status"], "ok")
        self.assertEqual(v1_info["type"], "v1")
        self.assertIn(FORMAT_JSON, v1_info["compatible_formats"])
        self.assertNotIn(FORMAT_FCP7, v1_info["compatible_formats"])
    
    def _create_test_v3_timeline(self):
        """Create a test v3 timeline for testing"""
        # Create a simple v3 timeline
        framerate = Fraction(30, 1)
        width, height = 1080, 1920
        samplerate = 48000
        
        # Create a v3 timeline
        timeline = self.timeline_manager.create_v3_timeline(
            width=width,
            height=height,
            framerate=framerate,
            samplerate=samplerate
        )
        
        # Add a dummy TlText object
        text_obj = TlText(
            start=0,
            dur=90,  # 3 seconds at 30fps
            text="Test Text",
            x=width // 2,
            y=height // 2,
            font="Arial",
            font_size=36,
            color="#FFFFFF"
        )
        
        # Add to the timeline
        timeline.v[0].append(text_obj)
        
        return timeline
    
    def _create_test_v1_timeline(self):
        """Create a test v1 timeline for testing"""
        # We need a real video file for this test to work properly
        # For unit testing purposes, we'll create a mock source
        
        from unittest.mock import MagicMock
        
        # Create a mock FileInfo
        mock_source = MagicMock()
        mock_source.path = Path("/fake/path/video.mp4")
        mock_source.video.duration = 10
        mock_source.video.fps = 30
        
        # Create mock chunks
        mock_chunks = MagicMock()
        mock_chunks.chunks = [(0, 90), (150, 240)]  # Two segments
        
        # Create v1 timeline
        timeline = v1(source=mock_source, chunks=mock_chunks)
        
        return timeline

if __name__ == "__main__":
    unittest.main()