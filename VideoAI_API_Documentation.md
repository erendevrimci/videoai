# VideoAI API Documentation

## Overview

VideoAI is a powerful video automation platform designed to create, edit, and manage videos using AI-powered tools. This document outlines the core functionalities and modules that can be integrated into a UI layer.

## Core Components

### 1. Video Generation Pipeline

The VideoAI platform provides an end-to-end video creation pipeline with the following stages:

1. **Script Generation** - Creates video scripts using OpenAI's GPT models
2. **Voice-over Generation** - Converts scripts to natural-sounding voice using ElevenLabs
3. **Caption Generation** - Creates accurate captions from audio
4. **Video Editing** - Assembles clips, captions, and audio into final videos
5. **Title & Description Generation** - Creates engaging titles and descriptions for videos
6. **YouTube Upload** - Publishes finished videos to YouTube channels

### 2. Timeline Management System

The timeline is a central concept in VideoAI, representing a video editing sequence:

- **Timeline Creation** - Create new timelines from scratch or YouTube clips
- **Timeline Editing** - Add, remove, and modify video/audio elements
- **Timeline Serialization** - Save and load timelines in various formats
- **Timeline Visualization** - Visual representation of timeline structures
- **Timeline Export** - Convert timelines to various editing software formats
- **Timeline Rendering** - Render timelines into final video files

### 3. YouTube Video Processing

Advanced YouTube video processing functionalities:

- **YouTube Search** - Search for videos based on keywords
- **Video Download** - Download videos for processing
- **Scene Detection** - Automatically detect scene changes in videos 
- **Clip Extraction** - Extract meaningful clips from videos
- **Caption Generation** - Generate accurate captions for clips
- **Timeline Generation** - Create editable timelines from clips

## API Modules

### FileManager (`file_manager.py`)

The FileManager handles file system operations across the platform.

```python
# Key methods
get_abs_path(rel_path: str) -> Path
ensure_dir_exists(directory: Path) -> bool
read_json(path: Path) -> Dict
write_json(path: Path, data: Dict) -> bool
get_channel_output_path(channel_number: int) -> Path
```

### TimelineManager (`timeline_manager.py`)

The TimelineManager creates and manages video editing timelines.

```python
# Key methods
create_v3_timeline(source_path=None, width=None, height=None, framerate=None) -> v3
clip_sequence_to_timeline(clip_sequence, output_width=1080, output_height=1920) -> v3
serialize_timeline(timeline, path=None, description="") -> Dict
deserialize_timeline(path) -> v3
visualize_timeline(timeline, width=None, detail_level=None) -> str
add_captions_to_timeline(timeline, captions_path, font="Arial") -> v3
save_timeline(timeline, timeline_name, description="") -> bool
load_timeline(timeline_name) -> v3
```

### VideoEdit (`video_edit.py`)

The VideoEdit module handles video assembly and rendering.

```python
# Key methods
load_clips_metadata() -> List[Dict]
match_clips_to_script(script, clips, target_duration=None) -> List[Dict]
create_video_sequence(clip_sequence, clips_metadata=None) -> bool
create_timeline(clip_sequence, channel_number=None) -> v3
merge_voice_with_video(video_path, voice_path, output_path) -> bool
burn_subtitles(video_path, srt_path, output_path) -> bool
render_timeline(timeline, output_path) -> bool
```

### YouTubeTimelineBuilder (`youtube_timeline_builder.py`)

Creates timelines from YouTube videos.

```python
# Key methods
create_timeline_from_clips(metadata, width=1920, height=1080) -> v3
save_timeline(timeline, output_path=None, timeline_name=None) -> Path
```

### YouTube Clips Processor (`yt_clips.py`)

Processes YouTube videos into clips with captions.

```python
# Key methods
search_youtube(query, max_results=5) -> List[VideoSearchResult]
download_video(video_url, output_path="downloads") -> Tuple[Path, VideoInfo]
detect_scenes(video_path, threshold=30.0) -> List
split_video_by_scenes(video_path, scene_list, min_duration=1.0) -> List[VideoClip]
generate_captions(video_path, model_name=WhisperModelSize.BASE) -> List[CaptionSegment]
process_video(query, options=None) -> Tuple[Path, VideoMetadata, Path]
```

### Configuration (`config.py`)

The config module provides configuration management for the platform.

```python
# Key configuration objects
config.openai - OpenAI API settings
config.elevenlabs - ElevenLabs API settings
config.youtube - YouTube API settings
config.video_edit - Video editing settings
config.script_generation - Script generation settings
config.timeline - Timeline settings
config.channels - Channel-specific configurations
```

## Data Models

### VideoMetadata

```python
class VideoMetadata(BaseModel):
    video: VideoSearchResult
    video_info: VideoInfo
    download_date: datetime
    clips: List[VideoClip]
    processor_version: str = "1.0.0"
```

### VideoClip

```python
class VideoClip(BaseModel):
    path: Path
    start_time: float
    end_time: float
    duration: float = 0.0
    captions: List[CaptionSegment]
```

### CaptionSegment

```python
class CaptionSegment(BaseModel):
    start: float
    end: float
    text: str
```

### ProcessingOptions

```python
class ProcessingOptions(BaseModel):
    max_results: int = 5
    output_directory: Path = Path("output")
    download_directory: Path = Path("downloads")
    scene_threshold: float = 15.0
    whisper_model: WhisperModelSize = WhisperModelSize.BASE
    min_clip_duration: float = 1.0
    skip_captions: bool = False
    generate_timeline: bool = True
    timeline_name: Optional[str] = None
    timeline_width: int = 1920
    timeline_height: int = 1080
    timeline_framerate: float = 30.0
    channel_number: Optional[int] = None
```

## Timeline Export Formats

VideoAI supports exporting timelines to various editing software formats:

1. **JSON** - Native VideoAI timeline format
2. **FCP7** - Final Cut Pro 7 XML format
3. **FCP11** - Final Cut Pro X XML format
4. **Shotcut** - Shotcut MLT format

## Rendering and Performance Monitoring

The platform includes advanced rendering capabilities with performance monitoring:

```python
render_timeline(timeline, output_path, channel_number=None, force_fallback=False) -> bool
```

Performance metrics tracked during rendering:
- Memory usage
- CPU utilization
- Rendering time
- Frame rate
- Output file size

## Multi-Channel Support

VideoAI supports multiple channels, each with its own configuration:

```python
get_channel_config(channel_number) -> ChannelConfig
get_timeline_config(channel_number) -> TimelineConfig
get_export_config(channel_number) -> ExportConfig
```

## Configuration Flags and Settings for UI Development

Below is a comprehensive list of all configuration flags and settings that should be exposed in a user interface:

### 1. General Application Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `base_directory` | Path | Current directory | Base directory for all operations |
| `working_mode` | Enum | "standard" | Operational mode ("standard", "performance", "compatibility") |
| `enable_logging` | Boolean | True | Enable detailed logging |
| `log_level` | Enum | "INFO" | Log level (DEBUG, INFO, WARNING, ERROR) |
| `auto_save_interval` | Integer | 5 | Auto-save interval in minutes |
| `max_parallel_processes` | Integer | 4 | Maximum number of parallel processes |
| `enable_ui_notifications` | Boolean | True | Enable notifications for long-running processes |
| `temp_files_cleanup` | Boolean | True | Automatically clean up temporary files |
| `language` | String | "en" | Interface language |

### 2. AI Model Settings

#### OpenAI Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `api_key` | String | From ENV | OpenAI API key |
| `script_model` | String | "gpt-4o" | Model for script generation |
| `title_desc_model` | String | "gpt-4o" | Model for title/description generation |
| `video_edit_model` | String | "o3-mini" | Model for video editing assistance |
| `temperature` | Float | 0.3 | Temperature for model randomness (0.0-1.0) |
| `max_tokens` | Integer | 4000 | Maximum tokens for completion |
| `prompt_cache_enabled` | Boolean | True | Enable caching of prompts for speed |
| `safety_filter` | Boolean | True | Enable content filtering |

#### ElevenLabs Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `api_key` | String | From ENV | ElevenLabs API key |
| `model_id` | String | "eleven_multilingual_v2" | Voice model |
| `default_voice_id` | String | "UgBBYS2sOqTuMpoF3BR0" | Default voice ID |
| `stability` | Float | 0.35 | Voice stability (0.0-1.0) |
| `similarity_boost` | Float | 0.55 | Voice similarity boost (0.0-1.0) |
| `style` | Float | 0.10 | Voice style parameter (0.0-1.0) |
| `use_speaker_boost` | Boolean | True | Enhance speaker voice quality |
| `optimize_streaming_latency` | Integer | 0 | Latency optimization level (0-4) |
| `voice_cache_enabled` | Boolean | True | Cache generated voice files |

### 3. YouTube Integration Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `api_key` | String | From ENV | YouTube API key |
| `client_id` | String | From ENV | YouTube OAuth client ID |
| `client_secret` | String | From ENV | YouTube OAuth client secret |
| `privacy_status` | Enum | "private" | Video privacy ("private", "unlisted", "public") |
| `category_id` | String | "28" | YouTube category ID |
| `upload_chunk_size` | Integer | 1048576 | Chunk size for uploads in bytes |
| `enable_resumable_uploads` | Boolean | True | Allow resuming interrupted uploads |
| `auto_publish` | Boolean | False | Automatically publish upon upload |
| `thumbnail_upload` | Boolean | True | Upload custom thumbnails |
| `enable_monetization` | Boolean | False | Enable monetization settings |
| `tags_auto_generate` | Boolean | True | Auto-generate video tags |
| `save_credentials` | Boolean | True | Save YouTube credentials |
| `max_retries` | Integer | 3 | Maximum upload retry attempts |

### 4. Video Processing Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `max_clip_duration` | Integer | 10 | Maximum clip duration in seconds |
| `default_clip_duration` | Integer | 10 | Default clip duration in seconds |
| `voice_volume` | Float | 1.4 | Voice-over volume level |
| `background_music_volume` | Float | 0.4 | Background music volume |
| `subtitle_font` | String | "DIN Condensed Bold" | Font for subtitles |
| `subtitle_font_size` | Integer | 13 | Font size for subtitles |
| `subtitle_position` | Enum | "bottom" | Subtitle position ("top", "middle", "bottom") |
| `subtitle_color` | String | "#FFFFFF" | Subtitle color (hex) |
| `subtitle_background` | String | "#00000080" | Subtitle background color (hex with alpha) |
| `subtitle_style` | Enum | "outline" | Subtitle style ("outline", "drop_shadow", "box") |
| `video_resolution` | Enum | "1080p" | Video resolution ("720p", "1080p", "2K", "4K") |
| `video_framerate` | Integer | 30 | Video framerate |
| `video_codec` | String | "h264" | Video codec ("h264", "h265", "vp9") |
| `audio_codec` | String | "aac" | Audio codec ("aac", "mp3", "opus") |
| `audio_sample_rate` | Integer | 48000 | Audio sample rate in Hz |
| `audio_channels` | Integer | 2 | Audio channels (1=mono, 2=stereo) |
| `enable_stabilization` | Boolean | False | Enable video stabilization |
| `enable_noise_reduction` | Boolean | True | Enable audio noise reduction |
| `enable_auto_levels` | Boolean | True | Enable automatic color levels |
| `enable_auto_contrast` | Boolean | True | Enable automatic contrast |
| `scene_detection_threshold` | Float | 15.0 | Scene detection sensitivity threshold |
| `min_scene_duration` | Float | 1.0 | Minimum scene duration in seconds |
| `clip_padding` | Float | 0.5 | Padding added to clip boundaries in seconds |
| `clip_overlap` | Float | 0.0 | Overlap between sequential clips in seconds |

### 5. Timeline Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `default_format` | Enum | "json" | Default timeline format |
| `default_preset` | Enum | "default" | Export preset ("default", "compatibility", "professional", "minimal") |
| `auto_analyze` | Boolean | True | Automatically analyze imported timelines |
| `validate_before_export` | Boolean | True | Validate timeline structure before export |
| `on_incompatible_elements` | Enum | "ask" | Handling of incompatible elements ("ask", "skip", "convert") |
| `create_sidecar_files` | Boolean | True | Create metadata sidecar files |
| `normalize_paths` | Boolean | True | Normalize media file paths |
| `convert_custom_elements` | Boolean | True | Convert custom elements to format-compatible ones |
| `relative_paths` | Boolean | True | Use relative paths when possible |
| `resolve_media_paths` | Boolean | True | Attempt to resolve missing media files |
| `storage_directory` | Path | "timelines/" | Directory for timeline storage |
| `visualization_directory` | Path | "timeline_visualizations/" | Directory for visualizations |
| `default_detail_level` | Enum | "normal" | Visualization detail level ("minimal", "normal", "detailed") |
| `default_framerate` | Integer | 30 | Default timeline framerate |
| `default_width` | Integer | 1920 | Default timeline width |
| `default_height` | Integer | 1080 | Default timeline height |
| `default_background` | String | "#000000" | Default timeline background color |
| `enable_performance_monitoring` | Boolean | True | Enable performance monitoring |
| `track_memory_usage` | Boolean | True | Track memory usage |
| `save_performance_reports` | Boolean | True | Save performance reports |
| `performance_output_dir` | Path | "outputs/performance" | Performance reports directory |

### 6. Voice and Caption Generation Flags

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `generate_captions` | Boolean | True | Generate caption files |
| `caption_format` | Enum | "srt" | Caption file format ("srt", "vtt", "ass") |
| `caption_language` | String | "en" | Caption language code |
| `caption_sync_mode` | Enum | "auto" | Caption synchronization mode ("auto", "manual", "force") |
| `caption_word_level` | Boolean | False | Generate word-level timing information |
| `whisper_model_size` | Enum | "base" | Whisper model size ("tiny", "base", "small", "medium", "large") |
| `enable_voice_clone` | Boolean | False | Enable voice cloning features |
| `voice_emotion` | Enum | "neutral" | Voice emotion ("neutral", "happy", "sad", "excited") |
| `speech_pace` | Float | 1.0 | Speed of generated speech (0.5-2.0) |
| `insert_pauses` | Boolean | True | Insert natural pauses in speech |
| `pause_threshold` | Float | 0.5 | Threshold for automatic pauses in seconds |
| `emphasis_detection` | Boolean | True | Detect and emphasize important words |

### 7. Channel-Specific Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `name` | String | "Default Channel" | Channel name |
| `voice_id` | String | - | Channel-specific voice ID |
| `youtube_credentials_file` | String | - | Channel YouTube credentials file |
| `youtube_info_file` | String | - | Channel YouTube info file |
| `privacy_status` | Enum | "private" | Channel default privacy status |
| `export_directory` | Path | - | Channel-specific export directory |
| `default_resolution` | Enum | "1080p" | Channel default resolution |
| `default_aspect_ratio` | Enum | "16:9" | Channel default aspect ratio |
| `enable_watermark` | Boolean | False | Enable channel watermark |
| `watermark_file` | Path | - | Channel watermark file |
| `watermark_opacity` | Float | 0.7 | Watermark opacity |
| `watermark_position` | Enum | "bottom-right" | Watermark position |
| `end_screen_duration` | Integer | 5 | End screen duration in seconds |
| `end_screen_template` | String | "default" | End screen template |
| `auto_schedule` | Boolean | False | Auto-schedule uploads |
| `schedule_time` | String | "12:00" | Default upload time |
| `schedule_day` | Enum | "any" | Default upload day |

### 8. Script Generation Flags

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `min_words` | Integer | 50 | Minimum script word count |
| `max_words` | Integer | 100 | Maximum script word count |
| `target_audience` | String | "fact enthusiasts" | Target audience description |
| `tone` | String | "informative yet conversational" | Script tone |
| `style` | String | "clear, engaging, and accessible" | Script style |
| `keyword_inclusion` | Boolean | True | Force include keywords |
| `keywords` | List[String] | [] | List of keywords to include |
| `include_intro` | Boolean | True | Include formal introduction |
| `include_outro` | Boolean | True | Include clear call-to-action outro |
| `facts_per_minute` | Integer | 3 | Target number of facts per minute |
| `narrative_structure` | Enum | "hook-body-conclusion" | Script structure type |

## Integration Recommendations

For UI integration, consider exposing these key functionalities:

1. **Timeline Editor** - Visual timeline editing interface
2. **YouTube Clip Browser** - Search and extract clips from YouTube
3. **Rendering Dashboard** - Monitor rendering progress and performance
4. **Export Panel** - Export timelines to various formats
5. **Channel Management** - Configure and manage multiple channels
6. **Voice & Caption Controls** - Manage voice and caption generation
7. **Settings Panel** - Comprehensive settings management with categories
8. **Project Management** - Save, load, and manage projects
9. **Media Browser** - Browse and manage media assets
10. **Render Queue** - Manage multiple rendering jobs
11. **Script Editor** - Create and edit scripts with AI assistance
12. **Performance Monitor** - View system resources and processing statistics
13. **Log Viewer** - Access application logs for troubleshooting

## UI Component Recommendations

To effectively implement the VideoAI functionality, the following UI components are recommended:

1. **Timeline Visualization**
   - Multiple tracks display (video, audio, captions)
   - Drag-and-drop clip arrangement
   - Zoom controls and time markers
   - Split/trim/extend tools for clips
   - Waveform visualization for audio

2. **Video Preview Player**
   - Real-time preview rendering
   - Frame-by-frame navigation
   - Transport controls (play, pause, seek)
   - In/out point marking
   - Timecode display

3. **YouTube Search Interface**
   - Search bar with filters
   - Results grid with thumbnails and metadata
   - Preview capability
   - Multi-select for batch processing
   - Scene detection visualization

4. **Channel Dashboard**
   - Channel performance metrics
   - Upload schedule calendar
   - Content library grid
   - Quick access to channel settings
   - Analytics visualization

5. **Rendering Progress Interface**
   - Progress bar with ETA
   - Real-time performance metrics
   - Resource utilization graphs
   - Cancel/pause capability
   - Output log display

## Error Handling

The platform uses a robust error handling system with specific exception types:

- `YouTubeError` - Base exception for YouTube operations
- `TimelineError` - Base exception for timeline operations
- `RenderingError` - Exception for rendering failures
- `ExportError` - Exception for export failures

Each operation returns clear success/failure indicators and detailed error messages. 

## User Permissions and Access Controls

For multi-user environments, the following permission flags can be implemented:

| Permission | Description |
|------------|-------------|
| `can_upload` | User can upload to YouTube |
| `can_edit_scripts` | User can create/edit scripts |
| `can_manage_channels` | User can manage channel settings |
| `can_export` | User can export timelines |
| `can_edit_timeline` | User can modify timelines |
| `can_download` | User can download YouTube videos |
| `admin_access` | Full administrative access |
| `api_key_management` | Can modify API keys |

## User Experience Guidelines

For optimal user experience, consider implementing:

1. **Progressive Disclosure** - Start with simplified UI and expose advanced options as needed
2. **Task-Based Navigation** - Organize UI around common user tasks rather than system components
3. **State Persistence** - Remember user settings and recent projects
4. **Keyboard Shortcuts** - Support efficient keyboard navigation for common actions
5. **Real-Time Feedback** - Provide immediate feedback for actions with visual indicators
6. **Contextual Help** - Offer tooltips and context-sensitive documentation
7. **Undo/Redo Stack** - Maintain comprehensive action history for error recovery
8. **Responsive Layout** - Support various display sizes with appropriate layout adjustments 