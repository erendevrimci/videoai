"""
Performance monitoring system for VideoAI.

This module provides tools for tracking and analyzing performance metrics
in the VideoAI pipeline, with a focus on timeline rendering operations.
"""

import time
import os
import statistics
import functools
import tracemalloc
import typing as t
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

from logging_system.logger import Logger

# Initialize module logger
logger = Logger.get_logger("performance")

class MetricType(Enum):
    """Types of performance metrics that can be collected."""
    DURATION = "duration"  # Time measurements
    MEMORY = "memory"      # Memory usage
    IO = "io"              # I/O operations
    CPU = "cpu"            # CPU usage
    FRAMES = "frames"      # Frame processing stats
    CUSTOM = "custom"      # Custom metrics


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    type: MetricType
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    context: dict = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Collection of performance metrics with analysis."""
    metrics: t.List[PerformanceMetric] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    name: str = ""
    
    def add_metric(self, metric: PerformanceMetric) -> None:
        """Add a metric to the report."""
        self.metrics.append(metric)
    
    def get_duration(self) -> float:
        """Get total duration of the monitoring period."""
        if self.end_time == 0:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def complete(self) -> None:
        """Mark the report as complete."""
        self.end_time = time.time()
    
    def get_metrics_by_type(self, metric_type: MetricType) -> t.List[PerformanceMetric]:
        """Get all metrics of a specific type."""
        return [m for m in self.metrics if m.type == metric_type]
    
    def get_metrics_by_name(self, name: str) -> t.List[PerformanceMetric]:
        """Get all metrics with a specific name."""
        return [m for m in self.metrics if m.name == name]
    
    def get_statistics(self, metric_name: str = None, metric_type: MetricType = None) -> dict:
        """Calculate statistics for metrics matching the given name and/or type."""
        filtered_metrics = self.metrics
        
        if metric_name:
            filtered_metrics = [m for m in filtered_metrics if m.name == metric_name]
        
        if metric_type:
            filtered_metrics = [m for m in filtered_metrics if m.type == metric_type]
        
        if not filtered_metrics:
            return {}
        
        values = [m.value for m in filtered_metrics]
        
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "count": len(values),
            "unit": filtered_metrics[0].unit
        }
    
    def to_dict(self) -> dict:
        """Convert the report to a dictionary for serialization."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time if self.end_time > 0 else time.time(),
            "duration": self.get_duration(),
            "metrics": [
                {
                    "type": m.type.value,
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp,
                    "context": m.context
                }
                for m in self.metrics
            ]
        }
    
    def compare_with(self, other: 'PerformanceReport', metric_name: str = None) -> dict:
        """Compare this report with another and return statistics on differences."""
        if not metric_name:
            # Get all unique metric names from both reports
            self_names = {m.name for m in self.metrics}
            other_names = {m.name for m in other.metrics}
            all_names = self_names.union(other_names)
            
            # Compare each metric
            return {
                name: self.compare_with(other, name)
                for name in all_names
            }
        
        # Get statistics for the named metric in both reports
        self_stats = self.get_statistics(metric_name)
        other_stats = other.get_statistics(metric_name)
        
        if not self_stats or not other_stats:
            return {"error": "Metric not found in one of the reports"}
        
        # Calculate differences
        return {
            "self_mean": self_stats["mean"],
            "other_mean": other_stats["mean"],
            "absolute_diff": self_stats["mean"] - other_stats["mean"],
            "percent_diff": ((self_stats["mean"] - other_stats["mean"]) / other_stats["mean"]) * 100 
                          if other_stats["mean"] != 0 else float('inf'),
            "unit": self_stats["unit"]
        }


class PerformanceMonitor:
    """
    Monitor and track performance metrics throughout the application.
    
    This class provides tools for measuring execution time, memory usage,
    and other performance characteristics, with a focus on timeline operations.
    """
    
    def __init__(self, name: str = "", enable_memory_tracking: bool = False):
        """
        Initialize a new performance monitor.
        
        Args:
            name: Name for this monitoring session
            enable_memory_tracking: Whether to track memory usage
        """
        self.name = name or f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_memory_tracking = enable_memory_tracking
        self.current_report = PerformanceReport(name=self.name)
        self.reports: t.List[PerformanceReport] = []
        
        if enable_memory_tracking:
            tracemalloc.start()
            logger.debug(f"Memory tracking enabled for {self.name}")
    
    def start_tracking(self, report_name: str = "") -> None:
        """
        Start a new tracking session.
        
        Args:
            report_name: Optional name for the tracking report
        """
        if self.current_report and len(self.current_report.metrics) > 0:
            self.complete_tracking()
        
        self.current_report = PerformanceReport(name=report_name or self.name)
        logger.debug(f"Started performance tracking: {self.current_report.name}")
    
    def add_metric(self, name: str, value: float, metric_type: MetricType, 
                  unit: str, context: dict = None) -> None:
        """
        Add a custom metric to the current report.
        
        Args:
            name: Name of the metric
            value: Value of the metric
            metric_type: Type of metric
            unit: Unit of measurement
            context: Additional contextual information
        """
        metric = PerformanceMetric(
            type=metric_type,
            name=name,
            value=value,
            unit=unit,
            context=context or {}
        )
        self.current_report.add_metric(metric)
        logger.debug(f"Added metric {name}: {value} {unit}")
    
    def measure_memory(self, name: str = "memory_usage") -> float:
        """
        Measure current memory usage and add as a metric.
        
        Args:
            name: Name for the memory metric
            
        Returns:
            Current memory usage in MB
        """
        if not self.enable_memory_tracking:
            logger.warning("Memory tracking not enabled - enable it in the constructor")
            return 0.0
        
        current, peak = tracemalloc.get_traced_memory()
        memory_mb = current / (1024 * 1024)  # Convert to MB
        
        self.add_metric(
            name=name,
            value=memory_mb,
            metric_type=MetricType.MEMORY,
            unit="MB",
            context={"peak_mb": peak / (1024 * 1024)}
        )
        
        return memory_mb
    
    def complete_tracking(self) -> PerformanceReport:
        """
        Complete the current tracking session and store the report.
        
        Returns:
            The completed performance report
        """
        self.current_report.complete()
        self.reports.append(self.current_report)
        logger.info(f"Completed performance tracking: {self.current_report.name} "
                   f"({self.current_report.get_duration():.2f}s, "
                   f"{len(self.current_report.metrics)} metrics)")
        
        completed_report = self.current_report
        self.current_report = PerformanceReport(name=self.name)
        
        return completed_report
    
    def save_reports(self, file_path: t.Optional[Path] = None) -> Path:
        """
        Save all performance reports to a JSON file.
        
        Args:
            file_path: Path to save the reports, or None to use default
            
        Returns:
            Path where the reports were saved
        """
        if not file_path:
            # Get output directory from environment or use default
            output_dir = Path(os.environ.get('PERFORMANCE_OUTPUT_DIR', 'outputs/performance'))
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Include current report if it has metrics
        reports_to_save = self.reports.copy()
        if self.current_report and len(self.current_report.metrics) > 0:
            reports_to_save.append(self.current_report)
        
        # Convert to serializable format
        data = {
            "monitor_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "reports": [report.to_dict() for report in reports_to_save]
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(reports_to_save)} performance reports to {file_path}")
        return file_path
    
    def compare_reports(self, report1_index: int, report2_index: int) -> dict:
        """
        Compare two reports and generate comparison statistics.
        
        Args:
            report1_index: Index of the first report
            report2_index: Index of the second report
            
        Returns:
            Dictionary of comparison statistics
        """
        if report1_index >= len(self.reports) or report2_index >= len(self.reports):
            logger.error(f"Invalid report indices: {report1_index}, {report2_index}")
            return {"error": "Invalid report indices"}
        
        report1 = self.reports[report1_index]
        report2 = self.reports[report2_index]
        
        return {
            "comparison": report1.compare_with(report2),
            "report1_name": report1.name,
            "report2_name": report2.name,
            "report1_duration": report1.get_duration(),
            "report2_duration": report2.get_duration(),
        }
    
    def summarize_reports(self) -> dict:
        """
        Generate a summary of all reports.
        
        Returns:
            Dictionary with summary statistics
        """
        all_reports = self.reports.copy()
        if self.current_report and len(self.current_report.metrics) > 0:
            all_reports.append(self.current_report)
        
        if not all_reports:
            return {"error": "No reports available"}
        
        # Get all unique metric names
        all_metric_names = set()
        for report in all_reports:
            all_metric_names.update(m.name for m in report.metrics)
        
        # Calculate statistics for each metric across all reports
        summary = {}
        for name in all_metric_names:
            all_values = []
            for report in all_reports:
                metrics = report.get_metrics_by_name(name)
                if metrics:
                    all_values.extend([m.value for m in metrics])
            
            if all_values:
                summary[name] = {
                    "min": min(all_values),
                    "max": max(all_values),
                    "mean": statistics.mean(all_values),
                    "median": statistics.median(all_values),
                    "stdev": statistics.stdev(all_values) if len(all_values) > 1 else 0,
                    "count": len(all_values),
                    "unit": metrics[0].unit if metrics else "unknown"
                }
        
        return {
            "monitor_name": self.name,
            "report_count": len(all_reports),
            "metrics": summary
        }


def timing_decorator(func=None, *, name=None, monitor=None, track_memory=False):
    """
    Decorator to measure and record function execution time.
    
    Args:
        func: The function to decorate
        name: Custom name for the metric (defaults to function name)
        monitor: PerformanceMonitor instance to use (creates new one if None)
        track_memory: Whether to track memory before and after function execution
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create monitor
            nonlocal monitor
            func_monitor = monitor or PerformanceMonitor(name=func.__name__)
            metric_name = name or func.__name__
            
            # Track memory before (if enabled)
            if track_memory and func_monitor.enable_memory_tracking:
                func_monitor.measure_memory(f"{metric_name}_before")
            
            # Measure execution time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise e
            finally:
                duration = time.time() - start_time
                
                # Record execution time
                func_monitor.add_metric(
                    name=metric_name,
                    value=duration,
                    metric_type=MetricType.DURATION,
                    unit="seconds",
                    context={
                        "function": func.__name__,
                        "success": success,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    }
                )
                
                # Track memory after (if enabled)
                if track_memory and func_monitor.enable_memory_tracking:
                    func_monitor.measure_memory(f"{metric_name}_after")
            
            return result
        
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


class RenderingPerformanceTracker:
    """
    Specialized performance tracker for rendering operations.
    
    This class extends the base PerformanceMonitor with rendering-specific
    metrics and analysis capabilities.
    """
    
    def __init__(self, name: str = "rendering", enable_memory_tracking: bool = True):
        """
        Initialize a rendering performance tracker.
        
        Args:
            name: Name for this tracking session
            enable_memory_tracking: Whether to track memory usage
        """
        self.monitor = PerformanceMonitor(name=name, enable_memory_tracking=enable_memory_tracking)
        self.direct_report = None
        self.fallback_report = None
        self.frame_count = 0
        self.timeline_complexity = 0
        
    def start_direct_rendering_tracking(self, timeline=None) -> None:
        """
        Start tracking performance for direct timeline rendering.
        
        Args:
            timeline: Optional timeline object to analyze complexity
        """
        self.monitor.start_tracking("direct_rendering")
        
        # Analyze timeline complexity if provided
        if timeline:
            self._analyze_timeline_complexity(timeline)
    
    def start_fallback_rendering_tracking(self, timeline=None) -> None:
        """
        Start tracking performance for fallback rendering.
        
        Args:
            timeline: Optional timeline object to analyze complexity
        """
        self.monitor.start_tracking("fallback_rendering")
        
        # Analyze timeline complexity if provided
        if timeline:
            self._analyze_timeline_complexity(timeline)
    
    def _analyze_timeline_complexity(self, timeline) -> None:
        """
        Analyze and record timeline complexity.
        
        Args:
            timeline: Timeline object to analyze
        """
        try:
            # Count video clips
            video_clips = sum(len(track) for track in timeline.v)
            
            # Count audio clips
            audio_clips = sum(len(track) for track in timeline.a)
            
            # Calculate overall complexity score
            complexity = video_clips + (audio_clips * 0.5)
            
            self.timeline_complexity = complexity
            
            # Record as a metric
            self.monitor.add_metric(
                name="timeline_complexity",
                value=complexity,
                metric_type=MetricType.CUSTOM,
                unit="score",
                context={
                    "video_clips": video_clips,
                    "audio_clips": audio_clips,
                    "video_tracks": len(timeline.v),
                    "audio_tracks": len(timeline.a)
                }
            )
            
            logger.debug(f"Timeline complexity: {complexity} "
                        f"({video_clips} video clips, {audio_clips} audio clips)")
        
        except Exception as e:
            logger.warning(f"Error analyzing timeline complexity: {e}")
    
    def record_frame_processed(self, frame_index: int, processing_time: float) -> None:
        """
        Record metrics for a processed video frame.
        
        Args:
            frame_index: Index of the processed frame
            processing_time: Time taken to process the frame (seconds)
        """
        self.frame_count += 1
        
        self.monitor.add_metric(
            name="frame_processing",
            value=processing_time,
            metric_type=MetricType.FRAMES,
            unit="seconds",
            context={
                "frame_index": frame_index,
                "frame_count": self.frame_count
            }
        )
    
    def record_audio_processing(self, duration: float, sample_count: int) -> None:
        """
        Record metrics for audio processing.
        
        Args:
            duration: Time taken for audio processing (seconds)
            sample_count: Number of audio samples processed
        """
        self.monitor.add_metric(
            name="audio_processing",
            value=duration,
            metric_type=MetricType.DURATION,
            unit="seconds",
            context={
                "sample_count": sample_count
            }
        )
    
    def record_io_operation(self, operation: str, size_mb: float, duration: float) -> None:
        """
        Record metrics for I/O operations.
        
        Args:
            operation: Type of I/O operation (e.g., 'read', 'write')
            size_mb: Size of data processed in MB
            duration: Time taken for the operation (seconds)
        """
        self.monitor.add_metric(
            name=f"io_{operation}",
            value=duration,
            metric_type=MetricType.IO,
            unit="seconds",
            context={
                "size_mb": size_mb,
                "throughput_mbps": size_mb / duration if duration > 0 else 0
            }
        )
    
    def complete_direct_rendering(self) -> PerformanceReport:
        """
        Complete tracking for direct rendering and return the report.
        
        Returns:
            Performance report for direct rendering
        """
        self.direct_report = self.monitor.complete_tracking()
        return self.direct_report
    
    def complete_fallback_rendering(self) -> PerformanceReport:
        """
        Complete tracking for fallback rendering and return the report.
        
        Returns:
            Performance report for fallback rendering
        """
        self.fallback_report = self.monitor.complete_tracking()
        return self.fallback_report
    
    def compare_rendering_approaches(self) -> dict:
        """
        Compare direct and fallback rendering performance.
        
        Returns:
            Dictionary with comparison statistics
        """
        if not self.direct_report or not self.fallback_report:
            logger.warning("Cannot compare rendering approaches: missing reports")
            return {"error": "Missing one or both rendering reports"}
        
        comparison = self.direct_report.compare_with(self.fallback_report)
        
        # Add overall speed comparison
        direct_duration = self.direct_report.get_duration()
        fallback_duration = self.fallback_report.get_duration()
        
        # Calculate speedup
        if fallback_duration > 0:
            speedup = fallback_duration / direct_duration if direct_duration > 0 else float('inf')
            speedup_percent = (1 - (direct_duration / fallback_duration)) * 100
        else:
            speedup = 0
            speedup_percent = 0
        
        # Add summary
        summary = {
            "direct_duration": direct_duration,
            "fallback_duration": fallback_duration,
            "speedup_factor": speedup,
            "speedup_percent": speedup_percent,
            "timeline_complexity": self.timeline_complexity,
            "frame_count": self.frame_count,
            "efficiency": speedup / self.timeline_complexity if self.timeline_complexity > 0 else 0
        }
        
        return {
            "summary": summary,
            "detailed_comparison": comparison
        }
    
    def save_reports(self, file_path: t.Optional[Path] = None) -> Path:
        """
        Save performance reports to file.
        
        Args:
            file_path: Optional custom file path
            
        Returns:
            Path where reports were saved
        """
        return self.monitor.save_reports(file_path)
    
    def generate_performance_report(self) -> dict:
        """
        Generate a comprehensive performance report for the rendering process.
        
        Returns:
            Dictionary with detailed performance statistics and recommendations
        """
        # Check if we have both reports
        have_both_reports = self.direct_report and self.fallback_report
        comparison = self.compare_rendering_approaches() if have_both_reports else None
        
        # Get all reports
        all_reports = []
        if self.direct_report:
            all_reports.append(self.direct_report)
        if self.fallback_report:
            all_reports.append(self.fallback_report)
        
        if not all_reports:
            return {"error": "No performance data available"}
        
        # Calculate frame-specific statistics
        frame_metrics = []
        for report in all_reports:
            frame_stats = report.get_statistics(metric_type=MetricType.FRAMES)
            if frame_stats:
                frame_metrics.append({
                    "report_name": report.name,
                    "stats": frame_stats
                })
        
        # Calculate I/O statistics
        io_metrics = []
        for report in all_reports:
            io_stats = {}
            for metric in report.metrics:
                if metric.type == MetricType.IO:
                    if metric.name not in io_stats:
                        io_stats[metric.name] = []
                    io_stats[metric.name].append(metric.value)
            
            if io_stats:
                io_metrics.append({
                    "report_name": report.name,
                    "stats": {
                        name: {
                            "mean": statistics.mean(values),
                            "total": sum(values)
                        }
                        for name, values in io_stats.items()
                    }
                })
        
        # Generate optimization recommendations
        recommendations = []
        
        if comparison and "summary" in comparison:
            summary = comparison["summary"]
            
            # Look for significant speedup or slowdown
            if summary["speedup_factor"] > 1.5:
                recommendations.append(
                    "Direct rendering shows significant performance advantage "
                    f"({summary['speedup_percent']:.1f}% faster)."
                )
            elif summary["speedup_factor"] < 0.9:
                recommendations.append(
                    "Fallback rendering is currently faster. Consider optimizing "
                    "direct rendering or using fallback for this type of timeline."
                )
            
            # Check frame processing times
            if frame_metrics and len(frame_metrics) > 1:
                direct_frames = next((fm for fm in frame_metrics if fm["report_name"] == "direct_rendering"), None)
                fallback_frames = next((fm for fm in frame_metrics if fm["report_name"] == "fallback_rendering"), None)
                
                if direct_frames and fallback_frames:
                    direct_mean = direct_frames["stats"].get("mean", 0)
                    fallback_mean = fallback_frames["stats"].get("mean", 0)
                    
                    if direct_mean > fallback_mean * 1.5:
                        recommendations.append(
                            "Frame processing is significantly slower in direct rendering. "
                            "Consider optimizing the render_av function implementation."
                        )
        
        # Generate final report
        return {
            "timestamp": datetime.now().isoformat(),
            "timeline_complexity": self.timeline_complexity,
            "frame_count": self.frame_count,
            "rendering_comparison": comparison if have_both_reports else "Only one approach measured",
            "frame_processing": frame_metrics,
            "io_operations": io_metrics,
            "recommendations": recommendations
        }