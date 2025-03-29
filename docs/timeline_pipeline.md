# Timeline-Aware Pipeline

## Overview

The timeline-aware pipeline integrates timeline functionality throughout the entire VideoAI workflow, creating a more structured approach to video production. By maintaining a timeline object from the beginning of the process, we can:

1. Preserve metadata and relationships between script segments, voice-over, and clips
2. Enable more advanced editing capabilities
3. Support progressive timeline building
4. Provide better visualizations of the production process

## Architecture

The timeline-aware pipeline creates a timeline object early in the process and progressively builds it through each step:

```
Script Generation → Voice-Over → Captions → Video Editing → Publishing
      ↓                 ↓            ↓             ↓             ↓
   Timeline        Timeline       Timeline      Timeline      Timeline
   (markers)      (voice track)  (caption data) (video tracks) (metadata)
```

## Usage

To use the timeline-aware pipeline, use the `--timeline` flag:

```bash
python main.py --channel 1 --timeline
```

You can also use existing timeline files as a starting point:

```bash
python video_edit.py --channel 1 --timeline --timeline-file my_timeline.json
```

## Timeline-Aware Functions

### Script Generation

The script generation phase adds script segments to the timeline with timing information:

```python
timeline = add_script_segments_to_timeline(timeline, script_segments)
```

### Voice-Over Generation

The voice-over phase adds the generated audio to the timeline:

```python
timeline = add_voice_to_timeline(timeline, voice_file)
```

### Caption Generation

The caption generation phase updates the timeline with precise word timings and synchronization information.

### Video Editing

The video editing phase adds clips to the timeline and handles rendering:

```python
timeline = add_clips_to_timeline(timeline, clip_sequence)
result = video_edit.main(channel_number, timeline=timeline)
```

## Timeline Storage

Throughout the process, the timeline is saved at various stages:

- `initial_timeline.json`: Empty timeline with channel-specific settings
- `script_timeline.json`: Timeline after script generation
- `voice_timeline.json`: Timeline after voice-over generation
- `captions_timeline.json`: Timeline after caption generation
- `final_timeline.json`: Complete timeline after video editing

## Command-Line Options

Several command-line options are available for controlling timeline behavior:

- `--timeline`: Enable timeline-aware processing
- `--timeline-file FILE`: Load a specific timeline file for processing
- `--force-fallback`: Force using fallback rendering even if timeline rendering is available
- `--direct-rendering`: Try to use direct timeline rendering instead of fallback
- `--skip-compatibility`: Skip backward compatibility checks (advanced)

## Features

1. **Progressive Timeline Building**
   - Each stage of the pipeline contributes to the timeline
   - Timeline files capture the state at each stage of production

2. **Timeline Visualization**
   - Detailed timeline visualizations are created automatically
   - Different detail levels are available (minimal, normal, detailed)

3. **Timeline Rendering**
   - Direct timeline-based rendering (when available)
   - Fallback rendering for backward compatibility

4. **Timeline Serialization**
   - JSON-based timeline storage
   - Optional compression for large timelines
   - Automatic backups of timeline files

## Future Enhancements

1. **Advanced Timeline Editing**
   - Multi-track support for video and audio
   - Transition effects between clips
   - Animation and visual effects

2. **Timeline Analysis**
   - Content analysis based on timeline data
   - Performance metrics for rendering

3. **Timeline Templates**
   - Reusable timeline templates for different video styles
   - Channel-specific timeline presets