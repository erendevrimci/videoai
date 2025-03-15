#!/usr/bin/env python
"""
Frame Analysis Module for VideoAI Project

This module extracts frames from video clips, sends them to a multimodal AI model
for detailed captioning, and organizes the results for efficient clip selection
during video editing. The module implements a hierarchical labeling system
to optimize matching between script segments and visual content while preventing
context window overflows.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import cv2
import numpy as np
from pydantic import BaseModel, Field
from tqdm import tqdm

# Import project-specific modules
from file_manager import FileManager
from logging_system.logger import Logger
from config import AppConfig

# Initialize logger
logger = Logger.get_logger(__name__)

class FrameExtractionConfig(BaseModel):
    """Configuration for frame extraction process."""
    frames_per_second: float = Field(0.25, description="Number of frames to extract per second")
    max_frames_per_clip: int = Field(3, description="Maximum number of frames to extract per clip")
    min_frame_distance: float = Field(4.0, description="Minimum distance between frames in seconds")
    extraction_method: str = Field("uniform", description="Method for extracting frames: 'uniform', 'keyframe', or 'scene_change'")
    frame_quality: int = Field(90, description="JPEG quality for saved frames (0-100)")
    skip_existing: bool = Field(True, description="Skip processing clips with existing frame data")
    batch_size: int = Field(10, description="Number of frames to process in a single batch")
    max_workers: int = Field(4, description="Maximum number of worker threads")
    
class VisionModelConfig(BaseModel):
    """Configuration for vision-language AI model."""
    model_name: str = Field("gpt-4o", description="Model to use for frame analysis")
    max_tokens: int = Field(300, description="Maximum tokens for response")
    temperature: float = Field(0.3, description="Temperature for generating captions")
    top_p: float = Field(0.9, description="Top p for sampling")
    presence_penalty: float = Field(0.0, description="Presence penalty for generation")
    frequency_penalty: float = Field(0.0, description="Frequency penalty for generation")

class LabelingConfig(BaseModel):
    """Configuration for frame labeling system."""
    primary_categories: List[str] = Field(
        ["people", "animals", "nature", "objects", "actions", "emotions", "abstract", "scenes"],
        description="Primary categories for classification"
    )
    max_labels_per_frame: int = Field(10, description="Maximum number of labels per frame")
    min_confidence_threshold: float = Field(0.7, description="Minimum confidence threshold for labels")
    hierarchical_depth: int = Field(3, description="Depth of hierarchical labeling (1-3)")
    semantic_grouping: bool = Field(True, description="Use semantic grouping for labels")
    include_scene_description: bool = Field(True, description="Include overall scene description")
    include_objects: bool = Field(True, description="Include objects in frame")
    include_actions: bool = Field(True, description="Include actions in frame")
    include_emotions: bool = Field(True, description="Include emotions in frame")
    include_attributes: bool = Field(True, description="Include visual attributes")
    include_abstract_concepts: bool = Field(False, description="Include abstract concepts")

@dataclass
class FrameData:
    """Data structure for frame information."""
    clip_path: Path
    frame_path: Path
    timestamp: float
    caption: str = ""
    description: str = ""
    labels: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert frame data to dictionary."""
        return {
            "clip_path": str(self.clip_path),
            "frame_path": str(self.frame_path),
            "timestamp": self.timestamp,
            "caption": self.caption,
            "description": self.description,
            "labels": self.labels,
            "metadata": self.metadata,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrameData':
        """Create FrameData from dictionary."""
        return cls(
            clip_path=Path(data["clip_path"]),
            frame_path=Path(data["frame_path"]),
            timestamp=data["timestamp"],
            caption=data["caption"],
            description=data["description"],
            labels=data["labels"],
            metadata=data["metadata"],
            confidence=data["confidence"]
        )

class ClipFrameAnalysis:
    """Analysis results for a single clip."""
    def __init__(self, clip_path: Path):
        self.clip_path = clip_path
        self.frames: List[FrameData] = []
        self.summary: str = ""
        self.primary_labels: Dict[str, List[str]] = {}
        self.metadata: Dict[str, Any] = {}
        
    def add_frame(self, frame: FrameData) -> None:
        """Add analyzed frame to the collection."""
        self.frames.append(frame)
        
    def generate_summary(self) -> str:
        """Generate a concise summary of the clip based on frame analysis."""
        # TODO: Implement summary generation logic based on frame captions
        if not self.frames:
            return "No frames analyzed"
        
        # Extract key concepts from all frames
        all_labels = {}
        for frame in self.frames:
            for category, labels in frame.labels.items():
                if category not in all_labels:
                    all_labels[category] = []
                all_labels[category].extend(labels)
        
        # Count occurrences of each label
        label_counts = {}
        for category, labels in all_labels.items():
            label_counts[category] = {}
            for label in labels:
                if label in label_counts[category]:
                    label_counts[category][label] += 1
                else:
                    label_counts[category][label] = 1
        
        # Get most common labels per category
        top_labels = {}
        for category, counts in label_counts.items():
            sorted_labels = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            top_labels[category] = [label for label, _ in sorted_labels[:5]]
        
        # Format the summary
        summary_parts = []
        for category, labels in top_labels.items():
            if labels:
                summary_parts.append(f"{category.capitalize()}: {', '.join(labels)}")
        
        self.summary = ". ".join(summary_parts)
        self.primary_labels = top_labels
        return self.summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert clip analysis to dictionary."""
        return {
            "clip_path": str(self.clip_path),
            "frames": [frame.to_dict() for frame in self.frames],
            "summary": self.summary,
            "primary_labels": self.primary_labels,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClipFrameAnalysis':
        """Create ClipFrameAnalysis from dictionary."""
        analysis = cls(Path(data["clip_path"]))
        analysis.frames = [FrameData.from_dict(frame_data) for frame_data in data["frames"]]
        analysis.summary = data["summary"]
        analysis.primary_labels = data["primary_labels"]
        analysis.metadata = data["metadata"]
        return analysis

class FrameExtractor:
    """Extracts frames from video clips."""
    
    def __init__(self, config: FrameExtractionConfig, file_manager: FileManager):
        self.config = config
        self.file_manager = file_manager
        
    def extract_frames(self, clip_path: Path) -> List[Tuple[Path, float]]:
        """
        Extract frames from a video clip.
        
        Args:
            clip_path: Path to video clip
            
        Returns:
            List of tuples containing (frame_path, timestamp)
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                logger.error(f"Could not open video file: {clip_path}")
                return []
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Processing {clip_path.name}: {duration:.2f}s, {fps} fps, {total_frames} frames")
            
            # Determine frames to extract based on extraction method
            frames_to_extract = []
            
            if self.config.extraction_method == "uniform":
                # Extract frames at regular intervals
                if self.config.frames_per_second > 0:
                    interval = fps / self.config.frames_per_second
                    frames_to_extract = [int(i * interval) for i in range(int(total_frames / interval) + 1)]
                else:
                    # Use max_frames_per_clip to determine interval
                    interval = total_frames / min(total_frames, self.config.max_frames_per_clip)
                    frames_to_extract = [int(i * interval) for i in range(min(total_frames, self.config.max_frames_per_clip))]
                    
            elif self.config.extraction_method == "keyframe":
                # TODO: Implement keyframe extraction using OpenCV
                # This is a placeholder for future implementation
                logger.warning("Keyframe extraction not fully implemented, using uniform extraction")
                interval = total_frames / min(total_frames, self.config.max_frames_per_clip)
                frames_to_extract = [int(i * interval) for i in range(min(total_frames, self.config.max_frames_per_clip))]
                
            elif self.config.extraction_method == "scene_change":
                # TODO: Implement scene change detection
                logger.warning("Scene change detection not fully implemented, using uniform extraction")
                interval = total_frames / min(total_frames, self.config.max_frames_per_clip)
                frames_to_extract = [int(i * interval) for i in range(min(total_frames, self.config.max_frames_per_clip))]
            
            # Limit maximum number of frames
            if len(frames_to_extract) > self.config.max_frames_per_clip:
                frames_to_extract = frames_to_extract[:self.config.max_frames_per_clip]
            
            # Create output directory for frames
            output_dir = self._get_frames_dir(clip_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract and save frames
            extracted_frames = []
            for i, frame_number in enumerate(frames_to_extract):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Could not read frame {frame_number} from {clip_path.name}")
                    continue
                
                # Calculate timestamp
                timestamp = frame_number / fps
                
                # Save frame
                frame_path = output_dir / f"frame_{i:03d}_{timestamp:.2f}s.jpg"
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.frame_quality])
                
                extracted_frames.append((frame_path, timestamp))
                
            cap.release()
            logger.info(f"Extracted {len(extracted_frames)} frames from {clip_path.name}")
            return extracted_frames
        
        except Exception as e:
            logger.exception(f"Error extracting frames from {clip_path}: {str(e)}")
            return []
    
    def _get_frames_dir(self, clip_path: Path) -> Path:
        """Generate directory path for extracted frames."""
        # Create a directory structure based on clip path
        # Example: clips/animal.mp4 -> frames/clips/animal/
        base_frames_dir = self.file_manager.get_abs_path("frames")
        relative_path = clip_path.relative_to(self.file_manager.base_dir)
        clip_name = relative_path.stem
        return base_frames_dir / relative_path.parent / clip_name

class FrameAnalyzer:
    """Analyzes frames using a vision-language AI model."""
    
    def __init__(
        self, 
        vision_config: VisionModelConfig, 
        labeling_config: LabelingConfig,
        file_manager: FileManager
    ):
        self.vision_config = vision_config
        self.labeling_config = labeling_config
        self.file_manager = file_manager
        self.api_client = None
        
    def initialize_api_client(self, config: AppConfig):
        """Initialize API client based on configuration."""
        # TODO: Implement actual API client initialization based on chosen model
        # This is a placeholder for the actual implementation
        logger.info(f"Initializing vision model: {self.vision_config.model_name}")
        
        if self.vision_config.model_name.startswith("gpt-4o"):
            import openai
            self.api_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            # Handle other model providers
            logger.warning(f"Model {self.vision_config.model_name} not supported yet")
            
    def analyze_frame(self, frame_path: Path, frame_data: FrameData) -> FrameData:
        """
        Analyze a single frame using vision-language model.
        
        Args:
            frame_path: Path to frame image
            frame_data: Existing frame data object
            
        Returns:
            Updated frame data with analysis results
        """
        try:
            logger.debug(f"Analyzing frame: {frame_path.name}")
            
            # Build prompt based on labeling configuration
            prompt = self._build_analysis_prompt(frame_path)
            
            # Call vision-language model API
            response = self._call_vision_api(frame_path, prompt)
            
            # Parse and structure the response
            return self._parse_model_response(frame_data, response)
            
        except Exception as e:
            logger.exception(f"Error analyzing frame {frame_path}: {str(e)}")
            return frame_data
    
    def _build_analysis_prompt(self, frame_path: Path) -> str:
        """Build prompt for vision-language model based on configuration."""
        # The prompt format will vary depending on the model being used
        config = self.labeling_config
        
        prompt_parts = [
            "Analyze this image in detail and provide the following information:",
            "1. A detailed caption describing the scene (2-3 sentences)",
        ]
        
        if config.include_scene_description:
            prompt_parts.append("2. Overall scene context and setting")
            
        if config.include_objects:
            prompt_parts.append("3. Main objects, people, and elements visible in the frame")
            
        if config.include_actions:
            prompt_parts.append("4. Actions and movements occurring in the scene")
            
        if config.include_emotions:
            prompt_parts.append("5. Emotional tone and atmosphere of the scene")
            
        if config.include_attributes:
            prompt_parts.append("6. Visual attributes (colors, lighting, composition)")
            
        prompt_parts.append(f"""
        7. Hierarchical labels in these categories: {', '.join(config.primary_categories)}
        Format your response as JSON with the following structure:
        {{
            "caption": "Detailed caption here",
            "description": "Overall scene description",
            "labels": {{
                "category1": ["label1", "label2"],
                "category2": ["label3", "label4"]
            }}
        }}
        
        Only include high-confidence labels. Be specific and accurate.
        """)
        
        return "\n".join(prompt_parts)
    
    def _call_vision_api(self, frame_path: Path, prompt: str) -> Dict[str, Any]:
        """Call vision-language model API with the given frame and prompt."""
        # Implementation will depend on the specific API being used
        if not self.api_client:
            logger.error("API client not initialized")
            return {"error": "API client not initialized"}
        
        try:
            # Placeholder for actual API call
            # This implementation would change based on the chosen model
            
            if self.vision_config.model_name.startswith("gpt-4"):
                import base64
                import imghdr
                import cv2
                import time
                
                # Verify the image exists
                if not frame_path.exists():
                    logger.error(f"Image file does not exist: {frame_path}")
                    return {"error": "Image file not found"}
                
                # Check if the image can be opened with OpenCV
                try:
                    img = cv2.imread(str(frame_path))
                    if img is None:
                        logger.error(f"Image could not be loaded with OpenCV: {frame_path}")
                        # Try to fix the image by resaving it
                        logger.info(f"Attempting to fix image by resaving it: {frame_path}")
                        # Extract frames directly from the video at the specific timestamp
                        clip_path = Path(str(frame_path).split('frame_')[0].replace('/frames/', '/clips/'))
                        if clip_path.exists():
                            timestamp = 0.0
                            try:
                                # Extract timestamp from filename (e.g., frame_000_0.00s.jpg)
                                timestamp_str = str(frame_path.name).split('_')[-1].replace('s.jpg', '')
                                timestamp = float(timestamp_str)
                            except:
                                pass
                            
                            # Extract frame from video
                            cap = cv2.VideoCapture(str(clip_path))
                            if cap.isOpened():
                                # Set position to timestamp
                                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                                ret, new_frame = cap.read()
                                if ret:
                                    # Save the frame to the same location
                                    cv2.imwrite(str(frame_path), new_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                    logger.info(f"Successfully extracted and saved new frame from video at timestamp {timestamp}s")
                                    img = new_frame
                                else:
                                    logger.error(f"Failed to extract frame at timestamp {timestamp}s from {clip_path}")
                                cap.release()
                
                except Exception as img_err:
                    logger.error(f"Error checking image with OpenCV: {str(img_err)}")
                    return {"error": f"Image validation failed: {str(img_err)}"}
                
                # Make sure frame_path is a valid image format (jpg, png, etc.)
                # OpenAI supports PNG, JPEG, GIF, and WebP formats
                is_valid = str(frame_path).lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))
                if not is_valid:
                    logger.warning(f"Image format may not be supported: {frame_path}")
                
                # Check image size and dimensions
                file_size = frame_path.stat().st_size
                if file_size > 20 * 1024 * 1024:  # 20 MB
                    logger.warning(f"Image is very large ({file_size / 1024 / 1024:.2f} MB): {frame_path}")
                    # Resize the image if too large
                    try:
                        img = cv2.imread(str(frame_path))
                        if img is not None:
                            height, width = img.shape[:2]
                            if width > 2000 or height > 2000:
                                # Resize to something more reasonable
                                scale = min(2000/width, 2000/height)
                                new_width = int(width * scale)
                                new_height = int(height * scale)
                                resized = cv2.resize(img, (new_width, new_height))
                                # Save to a temporary file
                                temp_path = frame_path.with_name(f"temp_{frame_path.name}")
                                cv2.imwrite(str(temp_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                # Replace the original with the temp file
                                temp_path.replace(frame_path)
                                logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
                    except Exception as resize_err:
                        logger.error(f"Error resizing image: {str(resize_err)}")
                
                # Read the image file and encode as base64
                try:
                    with open(frame_path, "rb") as img_file:
                        image_bytes = img_file.read()
                        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                except Exception as read_err:
                    logger.error(f"Error reading image file: {str(read_err)}")
                    return {"error": f"Image reading failed: {str(read_err)}"}
                
                # Determine MIME type based on file extension
                mime_type = "image/jpeg"  # Default mime type
                if str(frame_path).lower().endswith('.png'):
                    mime_type = "image/png"
                elif str(frame_path).lower().endswith('.gif'):
                    mime_type = "image/gif"
                elif str(frame_path).lower().endswith('.webp'):
                    mime_type = "image/webp"
                
                # Try to call the API, with retry on failure
                max_retries = 3
                retry_delay = 2  # seconds
                
                for retry in range(max_retries):
                    try:
                        response = self.api_client.chat.completions.create(
                            model=self.vision_config.model_name,
                            messages=[
                                {"role": "system", "content": "You are a detailed image analysis assistant. Analyze images thoroughly and respond in the exact format requested."},
                                {"role": "user", "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}
                                ]}
                            ],
                            max_tokens=self.vision_config.max_tokens,
                            temperature=self.vision_config.temperature
                        )
                        
                        return {"content": response.choices[0].message.content}
                    
                    except Exception as api_err:
                        if retry < max_retries - 1:
                            logger.warning(f"API call failed (attempt {retry+1}/{max_retries}): {str(api_err)}. Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            logger.error(f"API call failed after {max_retries} attempts: {str(api_err)}")
                            return {"error": str(api_err)}
            
            # Mock response for development/testing
            logger.warning("Using mock response for vision API")
            return self._generate_mock_response(frame_path)
            
        except Exception as e:
            logger.exception(f"Error calling vision API: {str(e)}")
            return {"error": str(e)}
    
    def _generate_mock_response(self, frame_path: Path) -> Dict[str, Any]:
        """Generate a mock response for testing."""
        # This is only for development and testing
        categories = self.labeling_config.primary_categories
        mock_labels = {category: [f"{category}_{i}" for i in range(3)] for category in categories}
        
        mock_response = {
            "content": json.dumps({
                "caption": f"Mock caption for {frame_path.name}",
                "description": f"Mock description of the scene in {frame_path.name}",
                "labels": mock_labels
            })
        }
        
        return mock_response
    
    def _parse_model_response(self, frame_data: FrameData, response: Dict[str, Any]) -> FrameData:
        """Parse and structure the model response."""
        try:
            if "error" in response:
                logger.error(f"Error in model response: {response['error']}")
                return frame_data
            
            content = response.get("content", "")
            
            # Try to extract JSON from the response
            try:
                # Find JSON part in the response (handle cases where model includes extra text)
                import re
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    json_str = json_match.group(1)
                    parsed = json.loads(json_str)
                else:
                    parsed = json.loads(content)
                    
                # Update frame data with parsed information
                frame_data.caption = parsed.get("caption", "")
                frame_data.description = parsed.get("description", "")
                frame_data.labels = parsed.get("labels", {})
                frame_data.confidence = 0.9  # Placeholder for actual confidence score
                
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from response: {content[:100]}...")
                # Fallback to basic text parsing if JSON parsing fails
                frame_data.caption = content[:200] if content else ""
                
            return frame_data
            
        except Exception as e:
            logger.exception(f"Error parsing model response: {str(e)}")
            return frame_data

class FrameAnalysisManager:
    """Manager class that coordinates frame extraction and analysis."""
    
    def __init__(
        self,
        extraction_config: FrameExtractionConfig,
        vision_config: VisionModelConfig,
        labeling_config: LabelingConfig,
        file_manager: FileManager,
        config: AppConfig
    ):
        self.extraction_config = extraction_config
        self.vision_config = vision_config
        self.labeling_config = labeling_config
        self.file_manager = file_manager
        self.config = config
        
        # Initialize components
        self.extractor = FrameExtractor(extraction_config, file_manager)
        self.analyzer = FrameAnalyzer(vision_config, labeling_config, file_manager)
        self.analyzer.initialize_api_client(config)
        
        # Create necessary directories
        self._initialize_directories()
    
    def _initialize_directories(self):
        """Create necessary directories for frame analysis."""
        frames_dir = self.file_manager.get_abs_path("frames")
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_dir = self.file_manager.get_abs_path("frame_analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)
    
    def process_clip(self, clip_path: Path) -> Optional[ClipFrameAnalysis]:
        """
        Process a single clip: extract frames and analyze them.
        
        Args:
            clip_path: Path to video clip
            
        Returns:
            ClipFrameAnalysis object with results, or None if processing failed
        """
        try:
            logger.info(f"Processing clip: {clip_path}")
            
            # Skip if already processed and skip_existing is True
            analysis_path = self._get_analysis_path(clip_path)
            if analysis_path.exists() and self.extraction_config.skip_existing:
                logger.info(f"Clip already processed: {clip_path.name}. Loading existing analysis.")
                return self._load_existing_analysis(analysis_path)
            
            # Extract frames
            frame_paths = self.extractor.extract_frames(clip_path)
            if not frame_paths:
                logger.warning(f"No frames extracted from {clip_path.name}")
                return None
            
            # Create analysis object
            analysis = ClipFrameAnalysis(clip_path)
            
            # Analyze frames
            with ThreadPoolExecutor(max_workers=self.extraction_config.max_workers) as executor:
                futures = []
                
                for frame_path, timestamp in frame_paths:
                    # Create frame data object
                    frame_data = FrameData(
                        clip_path=clip_path,
                        frame_path=frame_path,
                        timestamp=timestamp
                    )
                    
                    # Submit frame analysis task
                    future = executor.submit(self.analyzer.analyze_frame, frame_path, frame_data)
                    futures.append((future, frame_data))
                
                # Collect results
                for future, frame_data in tqdm(futures, desc=f"Analyzing frames from {clip_path.name}"):
                    try:
                        analyzed_frame = future.result()
                        analysis.add_frame(analyzed_frame)
                    except Exception as e:
                        logger.exception(f"Error processing frame {frame_data.frame_path}: {str(e)}")
            
            # Generate summary
            analysis.generate_summary()
            
            # Save analysis
            self._save_analysis(analysis, analysis_path)
            
            return analysis
            
        except Exception as e:
            logger.exception(f"Error processing clip {clip_path}: {str(e)}")
            return None
    
    def process_clips(self, clip_paths: List[Path]) -> Dict[Path, ClipFrameAnalysis]:
        """
        Process multiple clips in parallel.
        
        Args:
            clip_paths: List of paths to video clips
            
        Returns:
            Dictionary mapping clip paths to analysis results
        """
        results = {}
        with ThreadPoolExecutor(max_workers=self.extraction_config.max_workers) as executor:
            future_to_clip = {executor.submit(self.process_clip, clip_path): clip_path for clip_path in clip_paths}
            
            for future in tqdm(as_completed(future_to_clip), total=len(clip_paths), desc="Processing clips"):
                clip_path = future_to_clip[future]
                try:
                    analysis = future.result()
                    if analysis:
                        results[clip_path] = analysis
                except Exception as e:
                    logger.exception(f"Error processing clip {clip_path}: {str(e)}")
        
        return results
    
    def update_video_catalog(self, analyses: Dict[Path, ClipFrameAnalysis], catalog_path: Path, force_update: bool = False) -> None:
        """
        Update video catalog with frame analysis results.
        
        Args:
            analyses: Dictionary mapping clip paths to analysis results
            catalog_path: Path to video catalog CSV file
            force_update: If True, overwrite existing prompt/short_prompt even if already set
        """
        try:
            import pandas as pd
            
            # Load existing catalog
            if catalog_path.exists():
                catalog_df = pd.read_csv(catalog_path)
            else:
                catalog_df = pd.DataFrame(columns=["path", "prompt", "short_prompt", "aspect_ratio", "resolution", "fps", "duration", "labels"])
            
            # Add image caption columns if they don't exist
            max_frames = max([len(analysis.frames) for analysis in analyses.values()]) if analyses else 3
            for i in range(max_frames):
                column_name = f"image_{i+1}_caption"
                if column_name not in catalog_df.columns:
                    catalog_df[column_name] = ""
            
            # Update catalog with analysis results
            for clip_path, analysis in analyses.items():
                # Find row for this clip (if it exists)
                clip_rel_path = clip_path.relative_to(self.file_manager.base_dir)
                row_idx = catalog_df.index[catalog_df["path"] == str(clip_rel_path)].tolist()
                
                # Prepare a dictionary with image captions
                image_captions = {}
                for i, frame in enumerate(analysis.frames):
                    column_name = f"image_{i+1}_caption"
                    image_captions[column_name] = frame.caption if frame.caption else "No caption available"
                
                if not row_idx:
                    # Add new row if clip not in catalog
                    new_row = {
                        "path": str(clip_rel_path),
                        "prompt": analysis.summary,
                        "short_prompt": analysis.summary[:100] + "..." if len(analysis.summary) > 100 else analysis.summary,
                        "aspect_ratio": "",  # These would be filled from video metadata
                        "resolution": "",
                        "fps": "",
                        "duration": "",
                        "labels": self._format_labels_for_catalog(analysis.primary_labels)
                    }
                    # Add image captions
                    new_row.update(image_captions)
                    
                    catalog_df = pd.concat([catalog_df, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    # Update existing row
                    idx = row_idx[0]
                    catalog_df.at[idx, "labels"] = self._format_labels_for_catalog(analysis.primary_labels)
                    
                    # Update prompts if the current analysis has valid data or if the existing fields are empty
                    has_valid_summary = analysis.summary and analysis.summary.strip()
                    existing_prompt_empty = not catalog_df.at[idx, "prompt"] or catalog_df.at[idx, "prompt"] == ""
                    
                    if has_valid_summary and (existing_prompt_empty or force_update):
                        catalog_df.at[idx, "prompt"] = analysis.summary
                        catalog_df.at[idx, "short_prompt"] = analysis.summary[:100] + "..." if len(analysis.summary) > 100 else analysis.summary
                    
                    # Update image captions
                    for column, caption in image_captions.items():
                        catalog_df.at[idx, column] = caption
            
            # Save updated catalog
            catalog_df.to_csv(catalog_path, index=False)
            logger.info(f"Updated video catalog: {catalog_path}")
            
        except Exception as e:
            logger.exception(f"Error updating video catalog: {str(e)}")
    
    def _format_labels_for_catalog(self, primary_labels: Dict[str, List[str]]) -> str:
        """Format labels dictionary as a comma-separated string for catalog."""
        flat_labels = []
        for category, labels in primary_labels.items():
            flat_labels.extend(labels)
        
        return ", ".join(flat_labels)
    
    def _get_analysis_path(self, clip_path: Path) -> Path:
        """Generate path for saving frame analysis results."""
        analysis_dir = self.file_manager.get_abs_path("frame_analysis")
        relative_path = clip_path.relative_to(self.file_manager.base_dir)
        return analysis_dir / f"{relative_path.parent.name}_{relative_path.stem}.json"
    
    def _save_analysis(self, analysis: ClipFrameAnalysis, path: Path) -> None:
        """Save analysis results to JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(analysis.to_dict(), f, indent=2)
            logger.info(f"Saved analysis to {path}")
        except Exception as e:
            logger.exception(f"Error saving analysis to {path}: {str(e)}")
    
    def _load_existing_analysis(self, path: Path) -> Optional[ClipFrameAnalysis]:
        """Load existing analysis from JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return ClipFrameAnalysis.from_dict(data)
        except Exception as e:
            logger.exception(f"Error loading analysis from {path}: {str(e)}")
            return None

class ClipMatcher:
    """Matches script segments to clips based on frame analysis."""
    
    def __init__(
        self,
        file_manager: FileManager,
        config: AppConfig
    ):
        self.file_manager = file_manager
        self.config = config
        self.analysis_cache = {}
        
    def load_analyses(self, analysis_dir: Path) -> None:
        """Load all available frame analyses into cache."""
        try:
            analysis_files = list(analysis_dir.glob("*.json"))
            logger.info(f"Loading {len(analysis_files)} frame analyses")
            
            for path in analysis_files:
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    analysis = ClipFrameAnalysis.from_dict(data)
                    self.analysis_cache[Path(analysis.clip_path)] = analysis
                except Exception as e:
                    logger.warning(f"Error loading analysis {path}: {str(e)}")
            
            logger.info(f"Loaded {len(self.analysis_cache)} frame analyses into cache")
            
        except Exception as e:
            logger.exception(f"Error loading analyses from {analysis_dir}: {str(e)}")
    
    def match_segment_to_clips(
        self,
        segment_text: str,
        num_clips: int = 1,
        duration: float = 5.0,
        excluded_clips: List[Path] = None
    ) -> List[Path]:
        """
        Match a script segment to the most relevant clips.
        
        Args:
            segment_text: Text of the script segment
            num_clips: Number of clips to return
            duration: Desired total duration
            excluded_clips: List of clips to exclude from matching
            
        Returns:
            List of clip paths sorted by relevance
        """
        try:
            if not self.analysis_cache:
                logger.warning("No frame analyses in cache")
                return []
            
            excluded_clips = excluded_clips or []
            
            # Extract keywords from segment text
            keywords = self._extract_keywords(segment_text)
            
            # Score each clip based on keyword matches
            clip_scores = []
            for clip_path, analysis in self.analysis_cache.items():
                if clip_path in excluded_clips:
                    continue
                    
                score = self._calculate_match_score(keywords, analysis)
                clip_scores.append((clip_path, score, analysis))
            
            # Sort by score (descending)
            clip_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top clips
            return [clip_path for clip_path, _, _ in clip_scores[:num_clips]]
            
        except Exception as e:
            logger.exception(f"Error matching segment to clips: {str(e)}")
            return []
    
    def _extract_keywords(self, segment_text: str) -> List[str]:
        """Extract relevant keywords from segment text."""
        try:
            # Simple keyword extraction based on nouns, verbs, and adjectives
            # In a production system, this would use NLP for better extraction
            words = segment_text.lower().split()
            
            # Remove stopwords and punctuation
            stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of", "is", "are", "was", "were"}
            keywords = [word.strip(".,;:!?\"'()[]{}") for word in words if word.lower() not in stopwords]
            
            return keywords
            
        except Exception as e:
            logger.exception(f"Error extracting keywords: {str(e)}")
            return []
    
    def _calculate_match_score(self, keywords: List[str], analysis: ClipFrameAnalysis) -> float:
        """Calculate match score between keywords and clip analysis."""
        try:
            score = 0.0
            
            # Check for keyword matches in labels
            for category, labels in analysis.primary_labels.items():
                for label in labels:
                    for keyword in keywords:
                        if keyword in label.lower():
                            score += 1.0
            
            # Check for keyword matches in summary
            summary_words = analysis.summary.lower().split()
            for keyword in keywords:
                if keyword in summary_words:
                    score += 0.5
            
            # TODO: Implement more sophisticated matching logic
            # This could include semantic similarity, visual concepts, etc.
            
            return score
            
        except Exception as e:
            logger.exception(f"Error calculating match score: {str(e)}")
            return 0.0

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Frame Analysis Module for VideoAI")
    
    parser.add_argument("--all", action="store_true", help="Process all clips in the clips directory")
    parser.add_argument("--clips-dir", default="clips/not_filtered", help="Directory containing clips to process")
    parser.add_argument("--clips", nargs="+", help="Process specific clips (provide relative paths)")
    parser.add_argument("--update-catalog", action="store_true", help="Update video catalog with analysis results")
    parser.add_argument("--add-captions", action="store_true", help="Add frame captions to existing catalog without reprocessing")
    parser.add_argument("--reprocess-failed", action="store_true", help="Reprocess clips with empty captions")
    parser.add_argument("--limit", type=int, default=10, help="Limit the number of clips to process")
    parser.add_argument("--catalog-path", default="video/video-catalog-analyzed-not_filtered.csv", help="Path to custom catalog file")
    
    return parser.parse_args()

def add_captions_to_catalog(file_manager: FileManager, catalog_path: Path, force_update_prompts: bool = False):
    """
    Add frame captions to the catalog without reprocessing.
    
    Args:
        file_manager: FileManager instance for file operations
        catalog_path: Path to the catalog file
        force_update_prompts: If True, update prompts even if they already exist
    """
    try:
        import pandas as pd
        
        # Load existing catalog
        if not catalog_path.exists():
            logger.error(f"Catalog file does not exist: {catalog_path}")
            return
            
        catalog_df = pd.read_csv(catalog_path)
        
        # Get all frame analysis files
        analysis_dir = file_manager.get_abs_path("frame_analysis")
        if not analysis_dir.exists():
            logger.error(f"Frame analysis directory does not exist: {analysis_dir}")
            return
            
        analysis_files = list(analysis_dir.glob("*.json"))
        logger.info(f"Found {len(analysis_files)} frame analysis files")
        
        # Add image caption columns if they don't exist
        max_frames = 3  # Assuming at least 3 frames per clip
        for i in range(max_frames):
            column_name = f"image_{i+1}_caption"
            if column_name not in catalog_df.columns:
                catalog_df[column_name] = ""
        
        # Process each analysis file
        updates = 0
        for analysis_path in analysis_files:
            try:
                with open(analysis_path, "r") as f:
                    data = json.load(f)
                
                analysis = ClipFrameAnalysis.from_dict(data)
                clip_path = Path(analysis.clip_path)
                
                # Get the relative path for catalog lookup
                clip_rel_path = clip_path.relative_to(file_manager.base_dir)
                
                # Find the row in the catalog
                row_idx = catalog_df.index[catalog_df["path"] == str(clip_rel_path)].tolist()
                
                if row_idx:
                    idx = row_idx[0]
                    
                    # Add frame captions
                    for i, frame in enumerate(analysis.frames):
                        if i < max_frames:  # Only add up to max_frames
                            column_name = f"image_{i+1}_caption"
                            caption = frame.caption if frame.caption else "No caption available"
                            catalog_df.at[idx, column_name] = caption
                    
                    # Update labels if empty or forced update
                    if not catalog_df.at[idx, "labels"] or catalog_df.at[idx, "labels"] == "":
                        catalog_df.at[idx, "labels"] = ", ".join([item for sublist in analysis.primary_labels.values() for item in sublist])
                    
                    # Update prompts if empty or forced update
                    has_valid_summary = analysis.summary and analysis.summary.strip()
                    if has_valid_summary:
                        if not catalog_df.at[idx, "prompt"] or catalog_df.at[idx, "prompt"] == "" or force_update_prompts:
                            catalog_df.at[idx, "prompt"] = analysis.summary
                            catalog_df.at[idx, "short_prompt"] = analysis.summary[:100] + "..." if len(analysis.summary) > 100 else analysis.summary
                    
                    updates += 1
            except Exception as e:
                logger.error(f"Error processing analysis file {analysis_path}: {str(e)}")
        
        # Save updated catalog
        catalog_df.to_csv(catalog_path, index=False)
        logger.info(f"Updated {updates} entries in catalog with frame captions")
        
    except Exception as e:
        logger.exception(f"Error adding captions to catalog: {str(e)}")

def main():
    """Main function to run the frame analysis module."""
    args = parse_args()
    
    # Initialize components
    file_manager = FileManager()
    config = AppConfig()  # This would load from config.py
    
    # Get catalog path
    catalog_path = file_manager.get_abs_path(args.catalog_path)
    
    # If just adding captions to catalog
    if args.add_captions:
        logger.info(f"Adding frame captions to catalog: {catalog_path}")
        add_captions_to_catalog(file_manager, catalog_path, force_update_prompts=args.reprocess_failed)
        return
    
    # Create configuration objects
    extraction_config = FrameExtractionConfig(
        frames_per_second=0.25,
        max_frames_per_clip=3,
        extraction_method="uniform",
        frame_quality=90,
        skip_existing=not args.reprocess_failed,  # Skip existing unless reprocessing failed
        batch_size=30,
        max_workers=6
    )
    
    vision_config = VisionModelConfig(
        model_name="gpt-4o",
        max_tokens=300,
        temperature=0.3
    )
    
    labeling_config = LabelingConfig(
        primary_categories=[
            "people", "animals", "nature", "objects", 
            "actions", "emotions", "scenes", "concepts"
        ],
        max_labels_per_frame=10,
        hierarchical_depth=2
    )
    
    # Create manager
    manager = FrameAnalysisManager(
        extraction_config=extraction_config,
        vision_config=vision_config,
        labeling_config=labeling_config,
        file_manager=file_manager,
        config=config
    )
    
    # Determine which clips to process
    clip_paths = []
    
    if args.clips:
        # Process specific clips
        for clip_path in args.clips:
            abs_path = file_manager.get_abs_path(clip_path)
            if abs_path.exists():
                clip_paths.append(abs_path)
            else:
                logger.warning(f"Clip not found: {clip_path}")
    elif args.reprocess_failed:
        # Reprocess clips with empty captions
        analysis_dir = file_manager.get_abs_path("frame_analysis")
        for analysis_path in analysis_dir.glob("*.json"):
            try:
                with open(analysis_path, "r") as f:
                    data = json.load(f)
                
                # Check if any frames have empty captions
                frames = data.get("frames", [])
                if not frames or any(not frame.get("caption") for frame in frames):
                    clip_path = Path(data.get("clip_path", ""))
                    if clip_path.exists():
                        clip_paths.append(clip_path)
                        logger.info(f"Will reprocess clip with empty captions: {clip_path.name}")
            except Exception as e:
                logger.error(f"Error checking analysis file {analysis_path}: {str(e)}")
    else:
        # Process all clips in the specified directory
        clips_dir = file_manager.get_abs_path(args.clips_dir)
        clip_paths = list(clips_dir.glob("*.mp4"))
    
    if not clip_paths:
        logger.warning(f"No clips found to process")
        return
    
    # Limit the number of clips if specified and not processing all
    if not args.all and args.limit > 0 and args.limit < len(clip_paths):
        clip_paths = clip_paths[:args.limit]
    
    logger.info(f"Processing {len(clip_paths)} clips")
    
    # Process clips
    results = manager.process_clips(clip_paths)
    
    # Update video catalog if requested
    if args.update_catalog or args.all:
        # Force update if reprocessing failed clips
        force_update = args.reprocess_failed
        manager.update_video_catalog(results, catalog_path, force_update=force_update)
    
    logger.info(f"Processed {len(results)} clips successfully")

if __name__ == "__main__":
    main()