"""
Performance-enhanced implementation of render_timeline for VideoAI.

This file contains a modified version of the render_timeline function
with added performance monitoring capabilities.
"""

import traceback
import tempfile
import time
from typing import Optional
from pathlib import Path

from auto_editor.timeline import v3
from logging_system.performance_monitor import RenderingPerformanceTracker
from logging_system.logger import Logger

# Initialize logger
logger = Logger.get_logger("perf_render")

def render_timeline_with_monitoring(timeline: v3, output_path: Path, channel_number: Optional[int] = None, 
                  force_fallback: bool = False) -> bool:
    """
    Render a timeline to a video file using auto_editor's rendering capabilities.
    Enhanced with performance monitoring.
    
    Args:
        timeline (v3): The timeline object to render
        output_path (Path): Path where to save the output video
        channel_number (Optional[int]): Channel number to use, or None for default
        force_fallback (bool): Whether to force using the fallback rendering method
        
    Returns:
        bool: Whether rendering was successful
    """
    # Import functions directly to make mocking easier in tests
    from video_edit import get_timeline_config
    from video_edit import _render_timeline_fallback
    
    # Create performance tracker
    tracker = RenderingPerformanceTracker(name=f"rendering_channel_{channel_number}")
    
    # Get timeline configuration
    timeline_config = get_timeline_config(channel_number)
    
    # Check if we need to use the fallback path
    if force_fallback or not timeline_config.rendering.enabled:
        logger.info("Using fallback rendering path (direct rendering disabled or forced)")
        
        # Start tracking fallback rendering
        tracker.start_fallback_rendering_tracking(timeline)
        start_time = time.time()
        
        # Perform fallback rendering
        result = _render_timeline_fallback(timeline, output_path, channel_number)
        
        # Complete tracking and record duration
        fallback_duration = time.time() - start_time
        tracker.record_io_operation(
            operation="fallback_render", 
            size_mb=_estimate_output_size_mb(output_path), 
            duration=fallback_duration
        )
        tracker.complete_fallback_rendering()
        
        # Generate and save performance report
        tracker.save_reports()
        return result
    
    # Try direct rendering path
    try:
        # Import render modules from auto-editor
        try:
            import av
            from auto_editor.render import video, audio
            
            # Check if the required functions exist
            if not hasattr(video, 'render_av'):
                logger.info("Auto-editor does not have render_av function. Using fallback.")
                raise ImportError("Missing render_av function")
        except ImportError as e:
            logger.warning(f"Could not import auto-editor render modules: {e}")
            
            # Start tracking fallback rendering
            tracker.start_fallback_rendering_tracking(timeline)
            start_time = time.time()
            
            # Perform fallback rendering
            result = _render_timeline_fallback(timeline, output_path, channel_number)
            
            # Complete tracking and record duration
            fallback_duration = time.time() - start_time
            tracker.record_io_operation(
                operation="fallback_render_after_import_error", 
                size_mb=_estimate_output_size_mb(output_path), 
                duration=fallback_duration
            )
            tracker.complete_fallback_rendering()
            
            # Generate and save performance report
            tracker.save_reports()
            return result
        
        # Start tracking direct rendering
        tracker.start_direct_rendering_tracking(timeline)
        rendering_start_time = time.time()
        
        # Setup for rendering: create temp dir for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate audio first
            logger.info(f"Generating audio...")
            audio_result = []
            audio_start_time = time.time()
            try:
                # Generate audio using auto-editor
                audio_result = audio.make_new_audio(timeline, temp_dir)
                audio_duration = time.time() - audio_start_time
                
                # Record audio processing metrics
                tracker.record_audio_processing(
                    duration=audio_duration,
                    sample_count=len(audio_result)
                )
            except Exception as e:
                logger.error(f"Error generating audio: {e}")
                traceback.print_exc()
                
                # Record the failed audio processing
                audio_duration = time.time() - audio_start_time
                tracker.record_audio_processing(
                    duration=audio_duration,
                    sample_count=0  # 0 samples due to failure
                )
                
                # Switch to fallback rendering and track it
                tracker.complete_direct_rendering()  # Complete the failed direct rendering tracking
                tracker.start_fallback_rendering_tracking(timeline)
                fallback_start_time = time.time()
                
                # Fall back to simpler rendering
                result = _render_timeline_fallback(timeline, output_path, channel_number)
                
                # Record fallback metrics
                fallback_duration = time.time() - fallback_start_time
                tracker.record_io_operation(
                    operation="fallback_render_after_audio_error", 
                    size_mb=_estimate_output_size_mb(output_path), 
                    duration=fallback_duration
                )
                tracker.complete_fallback_rendering()
                
                # Generate and save performance report
                performance_report = tracker.generate_performance_report()
                tracker.save_reports()
                
                _log_performance_summary(performance_report, "Audio generation failed, used fallback")
                return result
            
            # Get codec and other configuration
            codec = timeline_config.rendering.video_codec
            
            # Open output file for writing
            logger.info(f"Rendering video...")
            video_start_time = time.time()
            total_frames = 0
            
            try:
                with av.open(str(output_path), mode='w') as output_container:
                    # Record container creation
                    container_start_time = time.time()
                    
                    # Configure output video stream
                    stream = output_container.add_stream(codec, rate=timeline.framerate)
                    stream.width = timeline.width
                    stream.height = timeline.height
                    stream.pix_fmt = timeline_config.rendering.pixel_format
                    
                    # Record container setup time
                    container_setup_duration = time.time() - container_start_time
                    tracker.monitor.add_metric(
                        name="container_setup",
                        value=container_setup_duration,
                        metric_type="duration",
                        unit="seconds"
                    )
                    
                    # Use auto-editor's render_av function to generate the video
                    render_generator = video.render_av(timeline, stream, temp_dir)
                    
                    # First yield is the stream object
                    render_stream = next(render_generator)
                    
                    # Setup progress bar
                    from utils.bar import Bar
                    bar = Bar("Rendering", max=100)
                    frame_count = 0
                    expected_frames = 0
                    
                    # Render frame by frame
                    for i, (frame_num, frame) in enumerate(render_generator):
                        frame_start_time = time.time()
                        
                        packets = render_stream.encode(frame)
                        for packet in packets:
                            output_container.mux(packet)
                            
                        frame_count += 1
                        total_frames += 1
                        
                        # Record frame processing time
                        frame_duration = time.time() - frame_start_time
                        tracker.record_frame_processed(
                            frame_index=frame_num,
                            processing_time=frame_duration
                        )
                        
                        # Update progress bar
                        if i == 0:
                            # We don't know total frames exactly, so estimate based on framerate and duration
                            expected_frames = int(timeline.framerate * (timeline.duration / timeline.sampling_rate))
                            bar.max = expected_frames
                        
                        if frame_count % 5 == 0:  # Update every 5 frames
                            progress = min(100, int(100 * frame_count / expected_frames)) if expected_frames > 0 else 0
                            bar.goto(frame_count)
                    
                    # Flush remaining frames
                    flush_start_time = time.time()
                    for packet in render_stream.encode(None):
                        output_container.mux(packet)
                    
                    # Record flush time
                    flush_duration = time.time() - flush_start_time
                    tracker.monitor.add_metric(
                        name="flush_frames",
                        value=flush_duration,
                        metric_type="duration",
                        unit="seconds"
                    )
                    
                    # Close progress bar
                    bar.finish()
                    
                    # Add audio to the video if available
                    if audio_result and len(audio_result) > 0:
                        # Note: We already wrote video to output_path, so we need to
                        # create a temporary file to hold the combined video+audio
                        audio_file = audio_result[0]
                        
                        logger.info("Adding audio to video...")
                        # This will be a no-op for now since we already have the output container
                        # In the future, we may implement proper muxing here if needed
                
                # Record overall video processing time
                video_duration = time.time() - video_start_time
                tracker.monitor.add_metric(
                    name="video_processing",
                    value=video_duration,
                    metric_type="duration",
                    unit="seconds",
                    context={
                        "total_frames": total_frames,
                        "fps": total_frames / video_duration if video_duration > 0 else 0
                    }
                )
                
                # Complete the direct rendering tracking
                total_render_duration = time.time() - rendering_start_time
                tracker.record_io_operation(
                    operation="direct_render_output", 
                    size_mb=_estimate_output_size_mb(output_path), 
                    duration=total_render_duration
                )
                tracker.complete_direct_rendering()
                
                # Generate and save performance report
                performance_report = tracker.generate_performance_report()
                tracker.save_reports()
                
                _log_performance_summary(performance_report, "Direct rendering successful")
                return True
                
            except Exception as e:
                logger.error(f"Error in direct rendering: {e}")
                traceback.print_exc()
                
                # Record the failed video processing
                video_duration = time.time() - video_start_time
                tracker.monitor.add_metric(
                    name="failed_video_processing",
                    value=video_duration,
                    metric_type="duration",
                    unit="seconds",
                    context={
                        "error": str(e),
                        "frames_before_error": total_frames
                    }
                )
                
                # Complete the failed direct rendering tracking
                tracker.complete_direct_rendering()
                
                # Fall back to the timeline fallback renderer
                logger.info("Trying fallback rendering method...")
                
                # Start tracking fallback rendering
                tracker.start_fallback_rendering_tracking(timeline)
                fallback_start_time = time.time()
                
                # Perform fallback rendering
                result = _render_timeline_fallback(timeline, output_path, channel_number)
                
                # Complete tracking and record duration
                fallback_duration = time.time() - fallback_start_time
                tracker.record_io_operation(
                    operation="fallback_render_after_video_error", 
                    size_mb=_estimate_output_size_mb(output_path), 
                    duration=fallback_duration
                )
                tracker.complete_fallback_rendering()
                
                # Generate and save performance report
                performance_report = tracker.generate_performance_report()
                tracker.save_reports()
                
                _log_performance_summary(performance_report, "Video processing failed, used fallback")
                return result
    
    except Exception as e:
        logger.error(f"Unexpected error in render_timeline: {e}")
        traceback.print_exc()
        
        # If we have an active tracking session, complete it
        try:
            tracker.complete_direct_rendering()
            tracker.save_reports()
        except:
            pass
            
        return False


def _estimate_output_size_mb(file_path: Path) -> float:
    """
    Estimate the size of the output file in MB.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Estimated size in MB
    """
    try:
        if file_path.exists():
            # Get actual file size
            return file_path.stat().st_size / (1024 * 1024)
        return 0.0
    except Exception:
        return 0.0


def _log_performance_summary(performance_report: dict, status_message: str) -> None:
    """
    Log a summary of the performance report.
    
    Args:
        performance_report: Performance report dictionary
        status_message: Status message to include
    """
    try:
        logger.info("\n--- RENDERING PERFORMANCE SUMMARY ---")
        logger.info(f"Status: {status_message}")
        
        if "rendering_comparison" in performance_report and performance_report["rendering_comparison"] != "Only one approach measured":
            comparison = performance_report["rendering_comparison"]
            if "summary" in comparison:
                summary = comparison["summary"]
                logger.info(f"Direct rendering: {summary.get('direct_duration', 0):.2f} seconds")
                logger.info(f"Fallback rendering: {summary.get('fallback_duration', 0):.2f} seconds")
                logger.info(f"Speedup: {summary.get('speedup_factor', 0):.2f}x ({summary.get('speedup_percent', 0):.1f}%)")
        
        if "recommendations" in performance_report and performance_report["recommendations"]:
            logger.info("\nRecommendations:")
            for recommendation in performance_report["recommendations"]:
                logger.info(f"- {recommendation}")
        
        logger.info("---------------------------------------\n")
    except Exception as e:
        logger.error(f"Error logging performance summary: {e}")
        # Non-critical error, just continue