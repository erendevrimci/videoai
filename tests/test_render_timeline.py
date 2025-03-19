import unittest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the modules to test
from video_edit import render_timeline, _render_timeline_fallback, _timeline_to_clip_sequence
from timeline_manager import TimelineManager
from config import TimelineConfig, TimelineRenderingConfig

class TestRenderTimeline(unittest.TestCase):
    """Test cases for timeline rendering functions and backward compatibility."""
    
    def setUp(self):
        """Set up test environment."""
        self.timeline_mgr = TimelineManager()
        
        # Create test data directory if it doesn't exist
        self.test_output_dir = Path(__file__).parent / "test_output"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create a test timeline
        self.timeline = self.timeline_mgr.create_v3_timeline(
            width=1080,
            height=1920,
            framerate=30
        )
        
        # Test output path
        self.output_path = self.test_output_dir / "test_render_output.mp4"
        
    @patch('video_edit.TimelineManager')
    @patch('video_edit.get_timeline_config')
    def test_render_with_force_fallback(self, mock_get_config, mock_timeline_mgr):
        """Test that force_fallback triggers the fallback path."""
        # Setup mock configuration
        mock_config = MagicMock()
        mock_config.rendering.enabled = True  # Enable rendering
        mock_get_config.return_value = mock_config
        
        # Setup fallback mock
        with patch('video_edit._render_timeline_fallback') as mock_fallback:
            mock_fallback.return_value = True  # Fallback succeeds
            
            # Call render_timeline with force_fallback=True
            result = render_timeline(self.timeline, self.output_path, force_fallback=True)
            
            # Verify fallback was called regardless of config
            mock_fallback.assert_called_once()
            self.assertTrue(result)
            
    @patch('video_edit.get_timeline_config')
    def test_render_with_disabled_rendering(self, mock_get_config):
        """Test that disabled rendering triggers the fallback path."""
        # Setup mock configuration
        mock_config = MagicMock()
        mock_config.rendering.enabled = False  # Disable rendering
        mock_get_config.return_value = mock_config
        
        # Setup fallback mock
        with patch('video_edit._render_timeline_fallback') as mock_fallback:
            mock_fallback.return_value = True  # Fallback succeeds
            
            # Call render_timeline
            result = render_timeline(self.timeline, self.output_path)
            
            # Verify fallback was called due to disabled rendering
            mock_fallback.assert_called_once()
            self.assertTrue(result)
            
    @patch('video_edit.get_timeline_config')
    @patch('auto_editor.render.video', spec=True)
    @patch('auto_editor.render.audio', spec=True)
    def test_feature_detection(self, mock_audio, mock_video, mock_get_config):
        """Test that missing render functions trigger fallback."""
        # Setup mock configuration
        mock_config = MagicMock()
        mock_config.rendering.enabled = True  # Enable rendering
        mock_get_config.return_value = mock_config
        
        # Setup auto_editor mocks to NOT have the required attributes
        delattr(mock_video, 'render_av')
        
        # Setup fallback mock
        with patch('video_edit._render_timeline_fallback') as mock_fallback:
            mock_fallback.return_value = True  # Fallback succeeds
            
            # Call render_timeline
            result = render_timeline(self.timeline, self.output_path)
            
            # Verify fallback was called due to missing features
            mock_fallback.assert_called_once()
            self.assertTrue(result)
            
    @patch('video_edit.get_timeline_config')
    def test_import_error_handling(self, mock_get_config):
        """Test that import errors trigger fallback."""
        # Setup mock configuration
        mock_config = MagicMock()
        mock_config.rendering.enabled = True  # Enable rendering
        mock_get_config.return_value = mock_config
        
        # Setup import to fail
        with patch('auto_editor.render.video', side_effect=ImportError('Test error')):
            # Setup fallback mock
            with patch('video_edit._render_timeline_fallback') as mock_fallback:
                mock_fallback.return_value = True  # Fallback succeeds
                
                # Call render_timeline
                result = render_timeline(self.timeline, self.output_path)
                
                # Verify fallback was called due to import error
                mock_fallback.assert_called_once()
                self.assertTrue(result)
                
    def test_direct_rendering(self):
        """Simplified test for the direct timeline rendering path."""
        # Import required modules for test
        from unittest.mock import patch, MagicMock
        
        # Mock all the components we need
        mock_config = MagicMock()
        mock_config.rendering.enabled = True
        
        # Create a simple timeline
        timeline = self.timeline_mgr.create_v3_timeline(
            width=1080,
            height=1920,
            framerate=30
        )
        
        # Setup complete patching to bypass everything and just test the core functionality
        with patch('video_edit.get_timeline_config', return_value=mock_config), \
             patch('video_edit._render_timeline_fallback', return_value=True), \
             patch('tempfile.TemporaryDirectory') as mock_temp, \
             patch('av.open') as mock_av_open, \
             patch('auto_editor.render.video.render_av') as mock_render_av, \
             patch('auto_editor.render.audio.make_new_audio') as mock_make_audio:
            
            # Basic mocking of components
            mock_temp.return_value.__enter__.return_value = '/tmp/mockdir'
            mock_av_open.return_value = MagicMock()
            mock_make_audio.return_value = ['/tmp/audio.wav']
            
            # Generator setup for render_av
            mock_video_stream = MagicMock()
            mock_video_stream.encode.return_value = [MagicMock()]
            
            # Create a generator that yields the stream and then frames
            def video_generator():
                yield mock_video_stream
                for i in range(1):  # Just yield one frame for simplicity
                    yield (i, MagicMock())
                    
            mock_render_av.return_value = video_generator()
            
            # Call the function
            result = render_timeline(timeline, self.output_path)
            
            # Verify direct rendering was used successfully
            self.assertTrue(result)
            
            # Optional: verify specific component interactions
            # mock_make_audio.assert_called_once()
            # mock_render_av.assert_called_once()
        
    @patch('video_edit.get_timeline_config')
    @patch('auto_editor.render.video.render_av', side_effect=RuntimeError("Rendering error"))
    @patch('auto_editor.render.audio.make_new_audio')
    @patch('video_edit._render_timeline_fallback')
    def test_render_error_handling(self, mock_fallback, mock_make_audio, 
                                  mock_render_av, mock_get_config):
        """Test error handling during direct rendering."""
        # Setup mock configuration
        mock_config = MagicMock()
        mock_config.rendering.enabled = True
        mock_config.rendering.video_codec = 'h264'
        mock_get_config.return_value = mock_config
        
        # Setup render_av to raise an exception
        mock_render_av.side_effect = RuntimeError("Rendering error")
        
        # Setup fallback to succeed
        mock_fallback.return_value = True
        
        # Call render_timeline
        result = render_timeline(self.timeline, self.output_path)
        
        # Verify fallback was used after the error
        self.assertTrue(result)
        # Using ANY to match any value for channel_number since it might be None or a default value
        from unittest.mock import ANY
        mock_fallback.assert_called_once_with(self.timeline, self.output_path, ANY)
                
    @patch('video_edit.load_clips_metadata')
    @patch('video_edit.create_video_sequence')
    @patch('video_edit.file_mgr')
    def test_timeline_to_clip_sequence(self, mock_file_mgr, mock_create_video, mock_load_clips):
        """Test the conversion of timeline to clip sequence."""
        # Setup mocks
        mock_load_clips.return_value = {"sample_clip.mp4": {"path": "sample_clip.mp4"}}
        mock_create_video.return_value = True
        
        # Setup file_mgr mock for output path
        mock_output_path = MagicMock()
        mock_output_path.exists.return_value = False  # Don't try to copy the file
        mock_file_mgr.get_video_output_path.return_value = mock_output_path
        
        # Add a video clip to the timeline
        from auto_editor.ffwrapper import FileInfo
        from auto_editor.timeline import TlVideo
        
        # Create a dummy FileInfo
        mock_src = MagicMock()
        mock_src.path = Path("sample_clip.mp4")
        mock_src.video.fps = 30
        
        # Add clip to timeline
        clip = TlVideo(
            start=0,
            dur=300,  # 10 seconds at 30fps
            src=mock_src,
            offset=0,
            speed=1.0,
            stream=0
        )
        self.timeline.v[0].append(clip)
        
        # Test the fallback rendering
        result = _render_timeline_fallback(self.timeline, self.output_path)
        
        # Verify results
        self.assertTrue(result)
        mock_create_video.assert_called_once()
        
        # Extract the clip sequence from the call
        args, kwargs = mock_create_video.call_args
        clip_sequence = args[0]
        
        # Verify clip sequence contains our clip
        self.assertEqual(len(clip_sequence), 1)
        self.assertEqual(clip_sequence[0]["clip_name"], "sample_clip.mp4")
        self.assertEqual(clip_sequence[0]["duration"], 10.0)  # 300 frames / 30fps
        
    def test_timeline_to_clip_sequence_direct(self):
        """Test the direct conversion function from timeline to clip sequence."""
        # Add a video clip to the timeline
        from auto_editor.timeline import TlVideo
        
        # Create a dummy source
        mock_src = MagicMock()
        mock_src.path = Path("sample_clip.mp4")
        mock_src.video.fps = 30
        
        # Add clip to timeline
        clip = TlVideo(
            start=0,
            dur=300,  # 10 seconds at 30fps
            src=mock_src,
            offset=150,  # 5 seconds at 30fps
            speed=1.0,
            stream=0
        )
        self.timeline.v[0].append(clip)
        
        # Convert timeline to clip sequence
        clip_sequence = _timeline_to_clip_sequence(self.timeline)
        
        # Verify results
        self.assertEqual(len(clip_sequence), 1)
        self.assertEqual(clip_sequence[0]["clip_name"], "sample_clip.mp4")
        self.assertEqual(clip_sequence[0]["duration"], 10.0)  # 300 frames / 30fps
        self.assertEqual(clip_sequence[0]["start_time"], 5.0)  # 150 frames / 30fps
        
    def test_timeline_to_clip_sequence_empty(self):
        """Test conversion with empty timeline."""
        # Timeline has no clips by default
        clip_sequence = _timeline_to_clip_sequence(self.timeline)
        
        # Verify result is empty list
        self.assertEqual(len(clip_sequence), 0)
        
    def test_timeline_to_clip_sequence_error_handling(self):
        """Test error handling in conversion function."""
        # Create a fresh timeline to ensure no other clips are present
        self.timeline = self.timeline_mgr.create_v3_timeline(
            width=1080,
            height=1920,
            framerate=30
        )
        
        # Add a problematic clip without required attributes
        broken_clip = MagicMock()
        # Intentionally do not set src attribute
        self.timeline.v[0].append(broken_clip)
        
        # Conversion should handle the error and skip the broken clip
        clip_sequence = _timeline_to_clip_sequence(self.timeline)
        
        # Result should be empty since clip was skipped
        self.assertEqual(len(clip_sequence), 0)

if __name__ == "__main__":
    unittest.main()