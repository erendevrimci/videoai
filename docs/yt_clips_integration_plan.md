# YouTube Clips Integration Plan

## Current Analysis of yt_clips.py

The current `yt_clips.py` implementation provides functionality for downloading and processing YouTube videos, but lacks several key features needed for robust integration with the main VideoAI system:

1. **Missing Integration with Logging System**
   - Uses print statements instead of structured logging
   - No exception handling framework integration
   - Cannot benefit from centralized logging configuration

2. **No FileManager Integration**
   - Uses direct file operations instead of FileManager class
   - Hardcoded path management
   - No channel-specific file handling
   - Manual error handling for file operations

3. **Lack of Type Hints and Pydantic Models**
   - No type annotations for function parameters/returns
   - No structured data models for configuration or results
   - Missing validation for input parameters

4. **Limited Configuration Options**
   - Hardcoded thresholds and parameters
   - No integration with config.py
   - Limited customization for clip analysis

5. **Basic Error Handling**
   - Simple try/except blocks with print statements
   - No structured error propagation
   - No detailed logging of errors

6. **No Pipeline Integration**
   - Operates as standalone script only
   - Cannot be easily called from main pipeline
   - Limited command-line interface options

7. **No Parallel Processing Support**
   - Not designed to work concurrently with other pipeline processes
   - No async/background operation capabilities
   - No coordination mechanism with the main pipeline
   - No progress tracking for long-running operations

## Proposed Enhancements

### 1. Integration with Logging System

```python
# Current approach
def download_video(video_url, output_path="downloads"):
    try:
        # Download code...
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None, None

# Enhanced approach
from logging_system.logger import Logger
from logging_system.exception_handler import log_exceptions, VideoAIException

# Initialize module logger
logger = Logger.get_logger("yt_clips")

class YouTubeDownloadError(VideoAIException):
    """Exception raised for YouTube download errors."""
    pass

@log_exceptions(logger_instance=logger)
def download_video(video_url: str, output_path: str = "downloads") -> tuple[Optional[Path], Optional[dict]]:
    try:
        # Download code...
    except Exception as e:
        raise YouTubeDownloadError(f"Failed to download video: {video_url}", cause=e)
```

### 2. FileManager Integration

```python
# Current approach
def process_video(query, output_dir="output"):
    # Direct file operations
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_path = os.path.join(output_dir, f"{video_info['id']}_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

# Enhanced approach
from file_manager import FileManager

file_mgr = FileManager()

def process_video(query: str, channel_number: Optional[int] = None) -> Optional[Path]:
    # Use FileManager for all file operations
    output_dir = file_mgr.get_clips_dir(channel_number) if channel_number else Path("downloads")
    
    # FileManager handles directory creation and error handling
    json_path = output_dir / f"{video_info['id']}_metadata.json"
    success = file_mgr.write_json(json_path, metadata)
    
    if not success:
        logger.error(f"Failed to write metadata for video {video_info['id']}")
        return None
        
    return json_path
```

### 3. Pydantic Models for Data Structures

```python
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class VideoInfo(BaseModel):
    """Pydantic model for YouTube video information."""
    id: str
    title: str
    url: HttpUrl
    duration: str
    views: str
    thumbnail: str

class CaptionSegment(BaseModel):
    """Model for caption segments."""
    start: float
    end: float
    text: str

class VideoClip(BaseModel):
    """Model for processed video clips."""
    path: Path
    start_time: float
    end_time: float
    captions: List[CaptionSegment] = Field(default_factory=list)

class VideoMetadata(BaseModel):
    """Model for complete video metadata."""
    video: Dict[str, Any]
    clips: List[Dict[str, Any]]
    download_date: datetime = Field(default_factory=datetime.now)
    processor_version: str = "1.0.0"
```

### 4. Configuration Integration

Create a new configuration model in `config.py`:

```python
class YouTubeClipConfig(BaseModel):
    """Configuration for YouTube clip downloading and processing."""
    max_results: int = 5
    scene_threshold: float = 30.0
    download_format: str = "best[ext=mp4]"
    whisper_model: str = "base"
    output_directory: Path = Path("downloads")
    auto_analyze_frames: bool = True
    min_clip_duration: float = 1.0
    max_clip_duration: float = 30.0
```

Update main config to include this:

```python
class Config(BaseModel):
    # Existing fields...
    youtube_clips: YouTubeClipConfig = Field(default_factory=YouTubeClipConfig)
```

### 5. Enhanced Command-Line Interface

```python
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and process YouTube videos")
    parser.add_argument("--query", type=str, help="YouTube search query")
    parser.add_argument("--url", type=str, help="Direct YouTube URL to process")
    parser.add_argument("--channel", type=int, help="Channel number for output organization")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum search results")
    parser.add_argument("--threshold", type=float, help="Scene detection threshold")
    parser.add_argument("--analyze-frames", action="store_true", help="Run frame analysis on clips")
    parser.add_argument("--skip-captions", action="store_true", help="Skip caption generation")
    return parser.parse_args()
```

### 6. Pipeline Integration Functions

```python
def download_clip_for_channel(query: str, channel_number: int) -> Optional[Path]:
    """Download a clip for a specific channel - callable from main pipeline."""
    config = get_config()
    yt_config = config.youtube_clips
    
    # Download and process
    metadata_path = process_video(
        query=query, 
        channel_number=channel_number,
        max_results=yt_config.max_results,
        threshold=yt_config.scene_threshold
    )
    
    if metadata_path:
        logger.info(f"Downloaded and processed YouTube clip for channel {channel_number}")
        
        # Optionally run frame analysis
        if yt_config.auto_analyze_frames:
            from frame_analysis_module import analyze_clip_frames
            clips_data = file_mgr.read_json(metadata_path)
            for clip in clips_data.get("clips", []):
                analyze_clip_frames(Path(clip["path"]))
    
    return metadata_path
```

### 7. Performance Monitoring Integration

```python
from logging_system.performance_monitor import timing_decorator, PerformanceMonitor

performance_monitor = PerformanceMonitor("yt_clips")

@timing_decorator(monitor=performance_monitor)
def detect_scenes(video_path: Path, threshold: float = 30.0) -> list:
    """Detect scenes in video with performance monitoring."""
    # Existing code with performance monitoring
```

## Implementation Plan

### Phase 1: Core Refactoring

1. **Add Type Hints and Docstrings**
   - Add complete type hints to all functions
   - Implement proper docstrings with Args/Returns sections
   - Create custom exception types

2. **Integrate Logging System**
   - Replace all print statements with proper logging
   - Add exception handling with custom exceptions
   - Implement log level control

3. **Implement Pydantic Models**
   - Create models for all data structures
   - Add validation for input parameters
   - Create serialization/deserialization methods

### Phase 2: FileManager Integration

1. **Replace Direct File Operations**
   - Use FileManager for all path operations
   - Implement proper error handling
   - Add channel-specific directory support

2. **Add Configuration Support**
   - Create YouTubeClipConfig in config.py
   - Use configuration values throughout code
   - Add command-line override options

3. **Enhance Error Recovery**
   - Implement proper cleanup of temporary files
   - Add retry mechanisms for network operations
   - Create detailed error reporting

### Phase 3: Pipeline Integration

1. **Create Pipeline Interface Functions**
   - Implement functions callable from main.py
   - Add channel-specific processing
   - Create integration with other components

2. **Enhance CLI**
   - Implement comprehensive command-line interface
   - Add configuration file support
   - Create detailed help documentation

3. **Add Performance Monitoring**
   - Integrate with performance monitoring system
   - Add metrics for download speed, processing time
   - Create performance reports

### Phase 4: Timeline-Based Integration

1. **Create Timeline Generator for yt_clips**
   - Implement timeline building from YouTube clips
   - Add metadata and clip information to timeline
   - Create timeline serialization functionality

2. **Develop Timeline Merging Component**
   - Create timeline merger with multiple strategies
   - Implement sequential, interleaved, and picture-in-picture merging
   - Add metadata preservation and enhancement

3. **Add Format Conversion**
   - Integrate with FCPXML and XML exporters
   - Add support for multiple output formats
   - Create timeline validation and correction

4. **Create Process and File Monitoring**
   - Implement file-based completion detection
   - Add timeout and error handling
   - Create notification system for process completion

### Phase 5: Testing and Documentation

1. **Create Unit Tests**
   - Implement tests for each component
   - Add integration tests
   - Create mock objects for external services

2. **Write Documentation**
   - Update project documentation
   - Create usage examples
   - Document configuration options

3. **Create Sample Workflows**
   - Provide example scenarios
   - Document integration patterns
   - Add troubleshooting guides

## Sample Implementation

Here's a sample of what the refactored module structure would look like:

```python
# yt_clips.py
"""
YouTube video downloading and processing module.

This module provides functionality for searching, downloading,
analyzing, and processing YouTube videos into clips suitable for
use in the VideoAI pipeline.
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

import yt_dlp
from youtube_search import YoutubeSearch
from moviepy.editor import VideoFileClip
import whisper
from scenedetect import VideoManager, SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images

from pydantic import BaseModel, HttpUrl, Field

from logging_system.logger import Logger
from logging_system.exception_handler import log_exceptions, VideoAIException
from logging_system.performance_monitor import timing_decorator, PerformanceMonitor
from file_manager import FileManager
from config import get_config

# Initialize module components
logger = Logger.get_logger("yt_clips")
file_mgr = FileManager()
perf_monitor = PerformanceMonitor("yt_clips")

# Custom exceptions
class YouTubeSearchError(VideoAIException):
    """Exception raised for YouTube search errors."""
    pass

class YouTubeDownloadError(VideoAIException):
    """Exception raised for YouTube download errors."""
    pass

class SceneDetectionError(VideoAIException):
    """Exception raised for scene detection errors."""
    pass

class VideoProcessingError(VideoAIException):
    """Exception raised for video processing errors."""
    pass

class CaptionGenerationError(VideoAIException):
    """Exception raised for caption generation errors."""
    pass

# Pydantic models
class VideoInfo(BaseModel):
    """YouTube video information."""
    id: str
    title: str
    url: str
    duration: str
    views: str
    thumbnail: str

class CaptionSegment(BaseModel):
    """Caption segment with timing information."""
    start: float
    end: float
    text: str

class VideoClip(BaseModel):
    """Information about a processed video clip."""
    path: str
    start_time: float
    end_time: float
    duration: float = 0.0
    captions: List[CaptionSegment] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        if self.end_time > self.start_time:
            self.duration = self.end_time - self.start_time

class VideoMetadata(BaseModel):
    """Complete metadata for a processed YouTube video."""
    video: Dict[str, Any]
    clips: List[VideoClip] = Field(default_factory=list)
    download_date: datetime = Field(default_factory=datetime.now)
    processor_version: str = "1.0.0"

# Core functions
@log_exceptions(logger_instance=logger)
def search_youtube(query: str, max_results: int = 5) -> List[VideoInfo]:
    """
    Search YouTube for videos matching the query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of VideoInfo objects
    """
    try:
        logger.info(f"Searching YouTube for: '{query}' (max results: {max_results})")
        results = YoutubeSearch(query, max_results=max_results).to_dict()
        
        videos = [
            VideoInfo(
                id=video['id'],
                title=video['title'],
                url=f"https://www.youtube.com/watch?v={video['id']}",
                duration=video['duration'],
                views=video['views'],
                thumbnail=video['thumbnails'][0]
            ) 
            for video in results
        ]
        
        logger.debug(f"Found {len(videos)} results for query: '{query}'")
        return videos
    except Exception as e:
        raise YouTubeSearchError(f"Failed to search YouTube for '{query}'", cause=e)

# ... Additional functions following the same pattern

# Main processing function
@timing_decorator(monitor=perf_monitor)
def process_video(
    query: str,
    channel_number: Optional[int] = None,
    max_results: int = 5,
    threshold: float = 30.0,
    skip_captions: bool = False
) -> Optional[Path]:
    """
    Main function to process a YouTube video.
    
    Args:
        query: YouTube search query
        channel_number: Optional channel number for organization
        max_results: Maximum search results
        threshold: Scene detection threshold
        skip_captions: Whether to skip caption generation
        
    Returns:
        Path to the generated metadata file, or None if processing failed
    """
    # Implementation using all the enhanced components...
    # ...

# Command line interface
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and process YouTube videos")
    # Add arguments...
    return parser.parse_args()

def main():
    """Main entry point for command line usage."""
    args = parse_args()
    config = get_config()
    
    try:
        # Process command line arguments and execute appropriate functions
        # ...
    except VideoAIException as e:
        logger.error(f"Error processing video: {e}")
        return False
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        return False
    
    # Save performance report
    perf_monitor.save_reports()
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

## Expected Benefits

1. **Improved Reliability**
   - Comprehensive error handling
   - Proper logging of errors and warnings
   - Clean failure recovery

2. **Better Performance Monitoring**
   - Detailed metrics on processing time
   - Identification of bottlenecks
   - Performance comparisons between runs

3. **Seamless Integration**
   - Consistent file management
   - Channel-specific processing
   - Pipeline-compatible interfaces

4. **Enhanced Flexibility**
   - Configurable parameters
   - Command-line overrides
   - Customizable processing steps

5. **Better Code Maintainability**
   - Type hints for IDE support
   - Comprehensive documentation
   - Modular, testable design

6. **Data Validation**
   - Pydantic models for input validation
   - Consistent data structures
   - Error prevention through validation

## Timeline-Based Integration Architecture

To make `yt_clips.py` work as a completely independent process that integrates with the main pipeline through timeline files, we'll implement a timeline-based integration architecture:

1. **Independent Processes**: Both yt_clips and the main pipeline run as independent processes
2. **Timeline-Based Output**: Each process produces a v3 timeline JSON output file
3. **Completion Detection**: A monitoring component detects when both processes complete
4. **Timeline Merging**: When both processes finish, their timelines are merged into a single unified timeline
5. **Format Conversion**: The unified timeline is converted to the required output formats (XML, FCPXML)

This approach has several advantages:
- Complete process isolation for improved stability
- No direct inter-process communication required
- Timeline files serve as the integration point
- Each process can be optimized independently
- The system is more resilient to failures in either process

### Timeline-Based Process Architecture

```
┌─────────────────────┐    ┌──────────────────────┐
│                     │    │                      │
│  Main Pipeline      │    │  yt_clips Process    │
│  Process            │    │                      │
│  ・Generate script   │    │  ・Download videos   │
│  ・Generate voice    │    │  ・Process videos    │
│  ・Create timeline A │    │  ・Create timeline B │
│                     │    │                      │
└─────────┬───────────┘    └──────────┬───────────┘
          │                           │
          ▼                           ▼
┌─────────────────────┐    ┌──────────────────────┐
│                     │    │                      │
│  Timeline A         │    │  Timeline B          │
│  (main content)     │    │  (clips)             │
│                     │    │                      │
└─────────┬───────────┘    └──────────┬───────────┘
          │                           │
          └───────────┬───────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │                       │
          │  Timeline Merger      │
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │                       │
          │  Unified Timeline     │
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │                       │
          │  Format Converter     │
          │  (XML, FCPXML)        │
          │                       │
          └───────────────────────┘
```

## Parallelization and Concurrent Operation

To implement the timeline-based integration architecture, several components are needed:

### 1. Asynchronous Operation Model

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from queue import Queue
import multiprocessing
from typing import Dict, List, Optional

class YouTubeClipProcessor:
    """Class for managing asynchronous YouTube clip processing."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize processor with configuration."""
        self.processing_queue = Queue()
        self.results_queue = Queue()
        self.running = False
        self.worker_thread = None
        self.config = config_path  # Configuration for the processor
        
    def start(self):
        """Start the background processing thread."""
        if self.running:
            logger.warning("Processor is already running")
            return
            
        self.running = True
        self.worker_thread = Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        logger.info("YouTube clip processor started in background")
        
    def stop(self):
        """Stop the background processing thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
        logger.info("YouTube clip processor stopped")
        
    def queue_download(self, query: str, channel_number: int) -> str:
        """
        Queue a YouTube download to be processed asynchronously.
        
        Args:
            query: Search query for YouTube
            channel_number: Channel to associate with this download
            
        Returns:
            Job ID for tracking this request
        """
        import uuid
        job_id = str(uuid.uuid4())
        
        # Add job to processing queue
        self.processing_queue.put({
            'job_id': job_id,
            'type': 'download',
            'query': query,
            'channel': channel_number,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Queued YouTube download: '{query}' for channel {channel_number} (Job ID: {job_id})")
        return job_id
        
    def get_job_status(self, job_id: str) -> Dict:
        """Get the status of a specific job."""
        # This would check a job status tracking structure
        # For a complete implementation, you'd need a job status database
        pass
        
    def _process_queue(self):
        """Worker thread to process queued jobs."""
        while self.running:
            try:
                # Get next job from queue (wait up to 1 second)
                try:
                    job = self.processing_queue.get(timeout=1.0)
                except Queue.Empty:
                    continue
                    
                logger.info(f"Processing job {job.get('job_id')}: {job.get('type')}")
                
                # Process based on job type
                if job.get('type') == 'download':
                    self._handle_download(job)
                # Other job types would be handled here
                
                # Mark job as done
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processor thread: {e}", exc_info=True)
                
    def _handle_download(self, job: Dict):
        """Handle a download job."""
        try:
            # Extract job parameters
            query = job.get('query')
            channel = job.get('channel')
            job_id = job.get('job_id')
            
            # Process the download using channel-specific paths
            options = ProcessingOptions(
                max_results=5,
                output_directory=file_mgr.get_channel_output_path(channel) / "clips",
                download_directory=file_mgr.get_channel_output_path(channel) / "downloads",
            )
            
            # Run the actual processing
            result = process_video(query, options)
            
            # Store the result
            if result:
                json_path, metadata = result
                self.results_queue.put({
                    'job_id': job_id,
                    'status': 'completed',
                    'metadata_path': str(json_path),
                    'clips_count': len(metadata.clips),
                    'timestamp': datetime.now().isoformat()
                })
                logger.info(f"Completed job {job_id}: Downloaded {len(metadata.clips)} clips")
            else:
                self.results_queue.put({
                    'job_id': job_id,
                    'status': 'failed',
                    'error': 'Processing returned no results',
                    'timestamp': datetime.now().isoformat()
                })
                logger.warning(f"Job {job_id} failed: No results returned")
                
        except Exception as e:
            logger.error(f"Error processing download job {job.get('job_id')}: {e}", exc_info=True)
            # Report failure
            self.results_queue.put({
                'job_id': job.get('job_id', 'unknown'),
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
```

### 2. Multiprocess Execution

```python
def launch_youtube_processor(config_path: str = None) -> multiprocessing.Process:
    """
    Launch the YouTube processor as a separate process.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Process object for the YouTube processor
    """
    process = multiprocessing.Process(
        target=run_youtube_processor,
        args=(config_path,),
        daemon=True,
        name="YouTubeProcessor"
    )
    process.start()
    return process
    
def run_youtube_processor(config_path: str = None):
    """
    Main function for the YouTube processor process.
    
    Args:
        config_path: Path to configuration file
    """
    # Set up process-specific logging
    from logging_system import Logger, LogLevel
    Logger.initialize(LoggerConfig(
        log_dir=Path("logs/youtube_processor"),
        log_level=LogLevel.DEBUG
    ))
    logger = Logger.get_logger("youtube_processor")
    
    logger.info("YouTube processor process started")
    
    # Load configuration
    config = load_processor_config(config_path)
    
    # Initialize the processor
    processor = YouTubeClipProcessor(config)
    
    # Create an HTTP server for interprocess communication
    create_api_server(processor, config.get('port', 8080))
    
    # Run until terminated
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("YouTube processor received shutdown signal")
    finally:
        # Clean up
        processor.stop()
        logger.info("YouTube processor shut down")
```

### 3. Inter-Process Communication

```python
from flask import Flask, request, jsonify

def create_api_server(processor: YouTubeClipProcessor, port: int = 8080):
    """
    Create a simple API server for IPC with the YouTube processor.
    
    Args:
        processor: The clip processor instance
        port: Port to listen on (default: 8080)
    """
    app = Flask("youtube_processor")
    
    @app.route('/api/download', methods=['POST'])
    def queue_download():
        """Queue a new download job."""
        data = request.json
        if not data or 'query' not in data or 'channel' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400
            
        job_id = processor.queue_download(
            query=data['query'],
            channel_number=int(data['channel'])
        )
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued'
        })
    
    @app.route('/api/status/<job_id>', methods=['GET'])
    def get_job_status(job_id):
        """Get the status of a specific job."""
        status = processor.get_job_status(job_id)
        return jsonify(status)
    
    @app.route('/api/jobs', methods=['GET'])
    def list_jobs():
        """List all active jobs."""
        jobs = processor.list_active_jobs()
        return jsonify({'jobs': jobs})
    
    # Start the server in a background thread
    def run_server():
        app.run(host='127.0.0.1', port=port)
        
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    
    logger.info(f"YouTube processor API available at http://127.0.0.1:{port}/api/")
    return server_thread
```

### 4. Progress Monitoring and Status Updates

```python
class JobStatus:
    """Status tracking for background jobs."""
    
    def __init__(self, job_id: str, job_type: str, params: Dict):
        """Initialize job status."""
        self.job_id = job_id
        self.job_type = job_type
        self.params = params
        self.status = "pending"
        self.progress = 0.0  # 0-100%
        self.start_time = datetime.now()
        self.end_time = None
        self.result = None
        self.error = None
        self.logs = []
        
    def update_progress(self, progress: float, message: str = None):
        """Update job progress."""
        self.progress = min(100.0, max(0.0, progress))
        if message:
            self.add_log(message)
            
    def complete(self, result: Any = None):
        """Mark job as completed."""
        self.status = "completed"
        self.progress = 100.0
        self.end_time = datetime.now()
        self.result = result
        self.add_log(f"Job completed successfully in {self.duration.total_seconds():.2f} seconds")
        
    def fail(self, error: str):
        """Mark job as failed."""
        self.status = "failed"
        self.end_time = datetime.now()
        self.error = error
        self.add_log(f"Job failed: {error}")
        
    def add_log(self, message: str):
        """Add a log message."""
        self.logs.append({
            'timestamp': datetime.now().isoformat(),
            'message': message
        })
        
    @property
    def duration(self) -> datetime.timedelta:
        """Get job duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return datetime.now() - self.start_time
        
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'job_id': self.job_id,
            'job_type': self.job_type,
            'status': self.status,
            'progress': self.progress,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': str(self.duration),
            'result': self.result,
            'error': self.error,
            'logs': self.logs[:10]  # Return last 10 logs
        }
```

### 5. Client Interface for Main Pipeline

```python
class YouTubeClient:
    """Client for interacting with the YouTube processor."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080/api"):
        """Initialize with the API base URL."""
        self.base_url = base_url
        self.session = requests.Session()
        
    def download_clips(self, query: str, channel_number: int) -> str:
        """
        Request clip download from the processor.
        
        Args:
            query: YouTube search query
            channel_number: Channel number
            
        Returns:
            Job ID for tracking
        """
        response = self.session.post(
            f"{self.base_url}/download",
            json={
                'query': query,
                'channel': channel_number
            }
        )
        response.raise_for_status()
        return response.json()['job_id']
        
    def get_job_status(self, job_id: str) -> Dict:
        """Get the status of a specific job."""
        response = self.session.get(f"{self.base_url}/status/{job_id}")
        response.raise_for_status()
        return response.json()
        
    def wait_for_completion(self, job_id: str, timeout: int = 600, 
                          progress_callback: Callable = None) -> Dict:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds
            progress_callback: Optional callback for progress updates
            
        Returns:
            Final job status
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(status)
                
            # Check if job is done
            if status['status'] in ('completed', 'failed'):
                return status
                
            # Wait before checking again
            time.sleep(2)
            
        # Timeout reached
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
        
    def download_clips_sync(self, query: str, channel_number: int, 
                           timeout: int = 600) -> Dict:
        """
        Download clips and wait for completion (synchronous).
        
        Args:
            query: YouTube search query
            channel_number: Channel number
            timeout: Maximum time to wait in seconds
            
        Returns:
            Final job status with results
        """
        job_id = self.download_clips(query, channel_number)
        return self.wait_for_completion(job_id, timeout)
```

### 6. Timeline Generation in yt_clips

```python
class YouTubeTimelineBuilder:
    """Builds v3 timeline from YouTube clips."""
    
    def __init__(self, timeline_manager: Optional['TimelineManager'] = None):
        """Initialize the timeline builder."""
        self.timeline_mgr = timeline_manager or TimelineManager()
        
    def create_timeline_from_clips(self, 
                               metadata: VideoMetadata,
                               width: int = 1920,
                               height: int = 1080,
                               framerate: float = 30.0) -> 'v3':
        """
        Create a v3 timeline from YouTube clip metadata.
        
        Args:
            metadata: The VideoMetadata containing clips information
            width: Video width
            height: Video height
            framerate: Frame rate
            
        Returns:
            v3 timeline object
        """
        # Create an empty timeline
        timeline = self.timeline_mgr.create_v3_timeline(
            width=width,
            height=height,
            framerate=framerate
        )
        
        # Ensure we have video tracks
        if not timeline.v or len(timeline.v) == 0:
            timeline.v = [[]]
        
        # Add metadata
        if not hasattr(timeline, 'metadata'):
            timeline.metadata = {}
            
        timeline.metadata['source_type'] = 'youtube'
        timeline.metadata['video_id'] = metadata.video.id
        timeline.metadata['video_title'] = metadata.video.title
        timeline.metadata['download_date'] = metadata.download_date.isoformat()
        timeline.metadata['clip_count'] = len(metadata.clips)
        
        # Add each clip to the timeline
        from auto_editor.ffwrapper import initFileInfo
        from auto_editor.timeline import TlVideo
        
        current_frame = 0
        for clip_index, clip in enumerate(metadata.clips):
            # Initialize clip source
            try:
                clip_src = initFileInfo(str(clip.path), None)
                
                # Get clip duration in frames
                clip_duration_frames = int(clip.duration * float(timeline.tb))
                
                # Create video object
                video_obj = TlVideo(
                    start=current_frame,  # Position in timeline
                    dur=clip_duration_frames,  # Duration to use
                    src=clip_src,  # Source file
                    offset=0,  # Start from beginning of clip
                    speed=1.0,  # Normal speed
                    stream=0  # Main video stream
                )
                
                # Add to the first video track
                timeline.v[0].append(video_obj)
                
                # Add caption data if available
                if clip.captions:
                    # Store caption data in timeline metadata for this clip
                    timeline.metadata[f'clip_{clip_index}_captions'] = [
                        {
                            'start': seg.start,
                            'end': seg.end,
                            'text': seg.text
                        } for seg in clip.captions
                    ]
                
                # Update current position
                current_frame += clip_duration_frames
                
            except Exception as e:
                logger.warning(f"Error adding clip {clip_index} to timeline: {e}")
                continue
        
        return timeline
    
    def export_timeline(self, 
                       timeline: 'v3', 
                       output_path: Union[str, Path],
                       channel_number: Optional[int] = None) -> Path:
        """
        Export a timeline to a JSON file.
        
        Args:
            timeline: The v3 timeline to export
            output_path: Output path for the timeline file
            channel_number: Optional channel number
            
        Returns:
            Path to the exported timeline file
        """
        # Ensure proper extension
        output_path = Path(output_path)
        if not output_path.suffix.lower() == '.json':
            output_path = output_path.with_suffix('.json')
            
        # Serialize and save the timeline
        path = self.timeline_mgr.serialize_timeline(
            timeline, 
            output_path,
            description=f"YouTube clips timeline for channel {channel_number}"
        )
        
        return path
```

### 7. Timeline Merging Component

```python
class TimelineMerger:
    """Merges multiple timelines into a unified timeline."""
    
    def __init__(self, timeline_manager: Optional['TimelineManager'] = None):
        """Initialize the timeline merger."""
        self.timeline_mgr = timeline_manager or TimelineManager()
        self.logger = Logger.get_logger("timeline_merger")
        
    def merge_timelines(self, 
                      main_timeline_path: Union[str, Path], 
                      clips_timeline_path: Union[str, Path],
                      merge_strategy: str = "sequential") -> 'v3':
        """
        Merge two timelines into a unified timeline.
        
        Args:
            main_timeline_path: Path to the main pipeline timeline (JSON)
            clips_timeline_path: Path to the YouTube clips timeline (JSON)
            merge_strategy: Strategy for merging ("sequential", "interleaved", "picture-in-picture")
            
        Returns:
            v3 timeline object with merged content
        """
        self.logger.info(f"Merging timelines with strategy: {merge_strategy}")
        self.logger.info(f"Main timeline: {main_timeline_path}")
        self.logger.info(f"Clips timeline: {clips_timeline_path}")
        
        # Load both timelines
        main_timeline = self.timeline_mgr.load_timeline(main_timeline_path)
        clips_timeline = self.timeline_mgr.load_timeline(clips_timeline_path)
        
        if not main_timeline or not clips_timeline:
            raise ValueError("Failed to load one or both timelines")
        
        # Create a new timeline with settings from the main timeline
        merged_timeline = self.timeline_mgr.create_v3_timeline(
            width=main_timeline.w,
            height=main_timeline.h,
            framerate=float(main_timeline.tb)
        )
        
        # Merge metadata
        if not hasattr(merged_timeline, 'metadata'):
            merged_timeline.metadata = {}
            
        merged_timeline.metadata['merged_from'] = [
            str(main_timeline_path),
            str(clips_timeline_path)
        ]
        merged_timeline.metadata['merge_strategy'] = merge_strategy
        merged_timeline.metadata['merge_timestamp'] = datetime.now().isoformat()
        
        # Merge content based on the selected strategy
        if merge_strategy == "sequential":
            # First add the main content, then the clips
            merged_timeline = self._merge_sequential(merged_timeline, main_timeline, clips_timeline)
        elif merge_strategy == "interleaved":
            # Alternate between main content and clips
            merged_timeline = self._merge_interleaved(merged_timeline, main_timeline, clips_timeline)
        elif merge_strategy == "picture-in-picture":
            # Add clips as picture-in-picture over the main content
            merged_timeline = self._merge_pip(merged_timeline, main_timeline, clips_timeline)
        else:
            raise ValueError(f"Unsupported merge strategy: {merge_strategy}")
        
        self.logger.info(f"Timeline merge complete. New timeline has {len(merged_timeline.v[0])} video segments")
        return merged_timeline
    
    def _merge_sequential(self, merged_timeline: 'v3', main_timeline: 'v3', clips_timeline: 'v3') -> 'v3':
        """Merge timelines sequentially (main content followed by clips)."""
        # Ensure video tracks exist
        if not merged_timeline.v or len(merged_timeline.v) == 0:
            merged_timeline.v = [[]]
        
        # Copy main timeline video segments
        current_frame = 0
        if main_timeline.v and len(main_timeline.v) > 0:
            for segment in main_timeline.v[0]:
                # Create a copy of the segment
                new_segment = copy.deepcopy(segment)
                new_segment.start = current_frame
                
                # Add to merged timeline
                merged_timeline.v[0].append(new_segment)
                
                # Update position
                current_frame += new_segment.dur
        
        # Copy clips timeline video segments
        if clips_timeline.v and len(clips_timeline.v) > 0:
            for segment in clips_timeline.v[0]:
                # Create a copy of the segment
                new_segment = copy.deepcopy(segment)
                new_segment.start = current_frame
                
                # Add to merged timeline
                merged_timeline.v[0].append(new_segment)
                
                # Update position
                current_frame += new_segment.dur
        
        # Merge audio tracks (similar approach to video)
        # [Audio merging code would be here]
        
        return merged_timeline
    
    def _merge_interleaved(self, merged_timeline: 'v3', main_timeline: 'v3', clips_timeline: 'v3') -> 'v3':
        """Merge timelines with clips interleaved between main content segments."""
        # Implementation would alternate between main content and clips
        # [Interleaving implementation would be here]
        return merged_timeline
    
    def _merge_pip(self, merged_timeline: 'v3', main_timeline: 'v3', clips_timeline: 'v3') -> 'v3':
        """Merge timelines with clips as picture-in-picture over main content."""
        # Implementation would place clips in a smaller window over main content
        # [PiP implementation would be here]
        return merged_timeline
    
    def export_merged_timeline(self, 
                             timeline: 'v3', 
                             output_path: Union[str, Path],
                             channel_number: Optional[int] = None) -> Path:
        """Export the merged timeline to a JSON file."""
        # Ensure proper extension
        output_path = Path(output_path)
        if not output_path.suffix.lower() == '.json':
            output_path = output_path.with_suffix('.json')
            
        # Serialize and save the timeline
        path = self.timeline_mgr.serialize_timeline(
            timeline, 
            output_path,
            description=f"Merged timeline for channel {channel_number}"
        )
        
        self.logger.info(f"Merged timeline exported to: {path}")
        return path
        
    def convert_to_formats(self, 
                         timeline: 'v3',
                         output_dir: Union[str, Path],
                         formats: List[str] = ['fcpxml', 'xml'],
                         channel_number: Optional[int] = None) -> Dict[str, Path]:
        """Convert the merged timeline to various output formats."""
        output_paths = {}
        output_dir = Path(output_dir)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            try:
                # Format-specific output path
                output_path = output_dir / f"merged_timeline_{channel_number}.{fmt}"
                
                # Use the appropriate exporter based on format
                if fmt.lower() == 'fcpxml':
                    from auto_editor.formats.fcp11 import save as save_fcpxml
                    save_fcpxml(timeline, str(output_path))
                    output_paths['fcpxml'] = output_path
                elif fmt.lower() == 'xml':
                    from auto_editor.formats.fcp7 import save as save_xml
                    save_xml(timeline, str(output_path))
                    output_paths['xml'] = output_path
                    
                self.logger.info(f"Exported timeline to {fmt} format: {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to export to {fmt} format: {e}", exc_info=True)
        
        return output_paths
```

### 8. Completion Monitoring and Timeline Integration

```python
def monitor_and_merge_timelines(channel_number: int, timeout_seconds: int = 3600) -> bool:
    """
    Monitor for completion of both processes and merge timelines when ready.
    
    Args:
        channel_number: Channel number to process
        timeout_seconds: Maximum time to wait for both processes
        
    Returns:
        True if successful, False otherwise
    """
    logger = Logger.get_logger("timeline_monitor")
    file_mgr = FileManager()
    timeline_mgr = TimelineManager()
    
    # Define expected timeline paths
    main_timeline_path = file_mgr.get_timeline_path(f"main_pipeline_{channel_number}", channel_number)
    clips_timeline_path = file_mgr.get_timeline_path(f"yt_clips_{channel_number}", channel_number)
    merged_timeline_path = file_mgr.get_timeline_path(f"merged_{channel_number}", channel_number)
    
    # Output directory for exported formats
    output_dir = file_mgr.get_channel_output_path(channel_number) / "exports"
    
    logger.info(f"Monitoring for timelines: \n- {main_timeline_path} \n- {clips_timeline_path}")
    
    # Monitor for timeline files
    start_time = time.time()
    main_ready = False
    clips_ready = False
    
    while time.time() - start_time < timeout_seconds:
        # Check for main timeline
        if not main_ready and file_mgr.file_exists(main_timeline_path):
            main_ready = True
            logger.info(f"Main pipeline timeline is ready: {main_timeline_path}")
        
        # Check for clips timeline
        if not clips_ready and file_mgr.file_exists(clips_timeline_path):
            clips_ready = True
            logger.info(f"YouTube clips timeline is ready: {clips_timeline_path}")
        
        # If both are ready, merge them
        if main_ready and clips_ready:
            logger.info("Both timelines are ready. Starting merge process...")
            
            try:
                # Create merger and merge timelines
                merger = TimelineMerger(timeline_mgr)
                merged_timeline = merger.merge_timelines(
                    main_timeline_path,
                    clips_timeline_path,
                    merge_strategy="sequential"  # Could be configurable
                )
                
                # Export merged timeline
                merger.export_merged_timeline(merged_timeline, merged_timeline_path, channel_number)
                
                # Export to different formats
                exported_paths = merger.convert_to_formats(
                    merged_timeline,
                    output_dir,
                    formats=['fcpxml', 'xml'],
                    channel_number=channel_number
                )
                
                logger.info(f"Timeline merge and export complete. Exported formats: {list(exported_paths.keys())}")
                return True
                
            except Exception as e:
                logger.error(f"Error merging timelines: {e}", exc_info=True)
                return False
        
        # Wait before checking again
        time.sleep(5)
    
    # Timeout reached
    logger.warning(f"Timeout reached after {timeout_seconds} seconds. Main ready: {main_ready}, Clips ready: {clips_ready}")
    return False
```

## Next Steps After Implementation

1. **Integration with frame_analysis_module.py**
   - Automatically analyze downloaded clips
   - Add to clip database
   - Enhance search capabilities

2. **Integration with video_edit.py**
   - Use downloaded clips as source material
   - Match clips to script segments
   - Enhance variety of visual content

3. **Custom Download Scheduling**
   - Regular downloads of trending content
   - Topic-specific clip collection
   - Content refresh automation

4. **Quality Metrics**
   - Implement clip quality scoring
   - Filter out low-quality content
   - Prioritize high-quality matches