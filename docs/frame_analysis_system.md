# Frame Analysis System for VideoAI

## Overview

The Frame Analysis System is a module designed to extract meaningful information from video clips using multimodal AI. By analyzing individual frames from each clip, the system creates rich metadata that enables precise matching between script segments and visual content during video generation.

## Key Features

1. **Intelligent Frame Extraction**: Extracts representative frames from video clips using various strategies (uniform sampling, keyframe detection, scene change detection)

2. **Multimodal AI Analysis**: Uses a vision-language model to generate detailed captions and hierarchical labels for each frame

3. **Hierarchical Labeling System**: Organizes visual content in a multi-level taxonomy to support efficient matching while preventing context window overflows

4. **Semantic Matching**: Matches script segments to relevant visual content based on meaning rather than just keywords

5. **Performance Optimization**: Implements parallel processing, caching, and batch operations to handle large clip libraries efficiently

## System Architecture

### Components

1. **FrameExtractor**: Extracts frames from video clips using configurable strategies
   - Uniform sampling (frames at regular intervals)
   - Keyframe detection (capture significant visual changes)
   - Scene change detection (identify distinct visual segments)

2. **FrameAnalyzer**: Analyzes frames using vision-language AI models
   - Generates detailed captions describing frame content
   - Creates hierarchical labels in multiple categories
   - Extracts visual attributes and semantic concepts

3. **ClipFrameAnalysis**: Aggregates frame analyses for a complete clip
   - Generates a concise summary of the clip's content
   - Identifies primary labels that represent the clip
   - Provides metadata for efficient retrieval

4. **FrameAnalysisManager**: Coordinates the extraction and analysis process
   - Manages parallel processing of multiple clips
   - Handles caching and persistence of analysis results
   - Updates the video catalog with analysis metadata

5. **ClipMatcher**: Matches script segments to relevant clips
   - Analyzes script content to extract key concepts
   - Scores clips based on semantic relevance to script segments
   - Filters and ranks potential matches

### Data Structures

1. **FrameData**: Information about a single analyzed frame
   - Caption: Detailed textual description
   - Hierarchical labels: Categorized metadata
   - Visual attributes: Colors, composition, lighting
   - Temporal metadata: Timestamp in the clip

2. **ClipFrameAnalysis**: Aggregated analysis of all frames in a clip
   - Primary labels: Most representative categories
   - Summary: Concise description of content
   - Frame collection: All analyzed frames with timestamps

3. **Hierarchical Label Structure**: Multi-level taxonomy of visual content
   - Level 1: Primary categories (people, animals, nature, objects, actions, emotions, scenes, concepts)
   - Level 2: Subcategories (e.g., people → man, woman, child, crowd, etc.)
   - Level 3: Specific attributes (e.g., woman → professional, casual, formal, etc.)

## Hierarchical Labeling System

The hierarchical labeling system is designed to balance specificity with efficiency, avoiding context window overflows while providing meaningful matches.

### Primary Categories

1. **People**: Human subjects and their attributes
   - Demographic: man, woman, child, baby, elderly, teen
   - Quantity: individual, couple, group, crowd
   - Profession: business, creative, service, technical
   - Activity: working, relaxing, exercising, presenting

2. **Animals**: Animal subjects and their attributes
   - Type: domestic, wild, marine, insect, bird
   - Specific: cat, dog, lion, tiger, dolphin, eagle
   - Activity: running, sleeping, hunting, playing
   - Interaction: with humans, with environment, with other animals

3. **Nature**: Natural settings and elements
   - Landscape: mountain, beach, forest, desert, ocean
   - Weather: sunny, rainy, snowy, cloudy, stormy
   - Time: day, night, sunrise, sunset, dusk, dawn
   - Season: spring, summer, fall, winter

4. **Objects**: Inanimate items in scenes
   - Technology: computer, phone, camera, vehicle
   - Household: furniture, appliance, decoration
   - Tools: manual, electric, professional, casual
   - Food: meal, snack, drink, dessert, ingredient

5. **Actions**: Activities and movements
   - Motion: walking, running, jumping, dancing
   - Interaction: talking, hugging, fighting, helping
   - Work: typing, building, cleaning, cooking
   - Leisure: reading, watching, playing, relaxing

6. **Emotions**: Emotional states and expressions
   - Positive: happy, excited, peaceful, satisfied
   - Negative: sad, angry, scared, frustrated
   - Neutral: focused, contemplative, listening
   - Complex: surprised, curious, nostalgic, determined

7. **Scenes**: Settings and environments
   - Urban: city, street, building, neighborhood
   - Indoor: home, office, store, restaurant
   - Natural: park, beach, mountain, forest
   - Event: party, meeting, ceremony, performance

8. **Concepts**: Abstract ideas and themes
   - Relationship: family, friendship, romance, community
   - State: success, failure, transformation, growth
   - Theme: freedom, isolation, connection, conflict
   - Mood: energy, calm, tension, harmony

## Implementation Guide

### Setup and Configuration

1. **Initial Setup**:
   - Create the `frames/` directory for storing extracted frames
   - Create the `frame_analysis/` directory for storing analysis results
   - Update `config.py` to include frame analysis configuration

2. **Configuration Parameters**:
   - Frame extraction: frames per second, extraction method, quality
   - Vision model: model name, parameters, API settings
   - Labeling system: categories, depth, confidence thresholds

### Workflow

1. **Clip Processing**:
   - Extract frames from clips using the configured method
   - Analyze each frame with the vision-language model
   - Generate summary and primary labels for the clip
   - Save analysis results to JSON file
   - Update video catalog with analysis metadata

2. **Integration with Video Editing**:
   - Modify `video_edit.py` to use the ClipMatcher for finding relevant clips
   - Update `match_clips_to_script` function to leverage frame analysis
   - Implement better variety and relevance in clip selection

3. **Performance Considerations**:
   - Process clips in parallel to maximize throughput
   - Implement caching to avoid redundant processing
   - Use batching for API calls to minimize rate limiting
   - Optimize frame extraction for large clip libraries

### Command Line Interface

```bash
# Process all clips in the clips directory
python frame_analysis_module.py --all

# Process specific clips
python frame_analysis_module.py --clips "clips/animal.mp4" "clips/nature.mp4"

# Update video catalog with new analyses
python frame_analysis_module.py --update-catalog

# Match a script to clips 
python frame_analysis_module.py --match-script "outputs/channel_1/script.txt"
```

## Integration with Existing System

### Update to `video_edit.py`

Modify the `match_clips_to_script` function in `video_edit.py` to leverage the frame analysis system:

```python
def match_clips_to_script(self, script_segments, channel_number):
    """Match script segments to relevant clips using frame analysis."""
    # Initialize clip matcher
    matcher = ClipMatcher(self.file_mgr, self.config)
    
    # Load analyses from frame_analysis directory
    analysis_dir = self.file_mgr.get_abs_path("frame_analysis")
    matcher.load_analyses(analysis_dir)
    
    # Match each segment to clips
    matched_clips = []
    excluded_clips = []
    
    for segment in script_segments:
        # Find matching clips for this segment
        matched = matcher.match_segment_to_clips(
            segment_text=segment["text"],
            num_clips=1,
            duration=segment["duration"],
            excluded_clips=excluded_clips
        )
        
        if matched:
            clip_path = matched[0]
            matched_clips.append({
                "segment": segment,
                "clip_path": clip_path
            })
            excluded_clips.append(clip_path)  # Avoid reusing the same clip
        else:
            # Fallback to existing matching method if no match found
            # ...existing fallback logic...
    
    return matched_clips
```

### Improvements to Existing CSV Format

Enhance the `video-catalog-labeled.csv` format to include frame analysis metadata:

```
path,prompt,short_prompt,aspect_ratio,resolution,fps,duration,labels,primary_categories,frame_summary
clips/animal.mp4,A cat playing with a ball of yarn...,A cat playing...,9:16,1080x1920,30.0,10s,"cat, play, yarn, cute","animals, actions, objects","Playful orange tabby cat batting at a red ball of yarn on wooden floor"
```

## Conclusion

The Frame Analysis System enhances the VideoAI project by enabling more intelligent and contextually relevant clip selection. By leveraging multimodal AI to understand the visual content of each clip, the system can create more engaging and coherent videos that match the script intent more effectively.

The hierarchical labeling system provides a structured approach to organizing visual content, balancing specificity with efficiency to avoid context window overflows while maintaining meaningful matches between textual and visual elements.