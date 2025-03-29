# Performance Monitoring System

The VideoAI project includes a comprehensive performance monitoring system that helps track and analyze the performance of timeline rendering operations. This document provides a detailed guide on how to use, configure, and extend the performance monitoring capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
   - [Global Configuration](#global-configuration)
   - [Channel-Specific Configuration](#channel-specific-configuration)
   - [Command-Line Options](#command-line-options)
   - [Environment Variables](#environment-variables)
3. [Core Components](#core-components)
   - [PerformanceMonitor](#performancemonitor)
   - [RenderingPerformanceTracker](#renderingperformancetracker)
   - [Performance Decorators](#performance-decorators)
4. [Using Performance Monitoring](#using-performance-monitoring)
   - [Enabling Monitoring](#enabling-monitoring)
   - [Tracking Custom Metrics](#tracking-custom-metrics)
   - [Tracking Memory Usage](#tracking-memory-usage)
   - [Rendering Performance](#rendering-performance)
5. [Performance Reports](#performance-reports)
   - [Report Structure](#report-structure)
   - [Saving and Loading Reports](#saving-and-loading-reports)
   - [Analyzing Results](#analyzing-results)
6. [Extending the System](#extending-the-system)
   - [Adding New Metric Types](#adding-new-metric-types)
   - [Creating Custom Trackers](#creating-custom-trackers)
7. [Best Practices](#best-practices)
   - [When to Enable Monitoring](#when-to-enable-monitoring)
   - [Optimizing Performance](#optimizing-performance)
   - [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

## Overview

The performance monitoring system provides tools for measuring and analyzing various performance metrics in the VideoAI pipeline, with a special focus on timeline rendering operations. It allows tracking execution time, memory usage, frame processing rates, and I/O operations.

Key features include:

- **Conditional Monitoring**: Enable monitoring only when needed to avoid overhead in production
- **Memory Usage Tracking**: Monitor memory consumption during rendering operations
- **Frame-by-Frame Analysis**: Track processing time for individual video frames
- **Rendering Comparison**: Compare direct and fallback rendering approaches
- **Performance Reports**: Generate detailed reports with statistics and recommendations
- **Custom Metrics**: Track any custom performance metrics relevant to your workflow

## Configuration

### Global Configuration

Performance monitoring settings are defined in the `TimelineRenderingConfig` class within `config.py`:

```python
class TimelineRenderingConfig(BaseModel):
    # ... other settings ...
    
    # Performance monitoring settings
    enable_performance_monitoring: bool = Field(default=False)  # Enable performance monitoring
    track_memory_usage: bool = Field(default=True)  # Track memory usage during rendering
    save_performance_reports: bool = Field(default=True)  # Save performance reports to disk
    performance_output_dir: str = Field(default="outputs/performance")  # Directory for reports
```

These settings control whether monitoring is enabled globally, whether memory tracking is active, whether reports are saved to disk, and where reports are stored.

### Channel-Specific Configuration

You can override monitoring settings for specific channels by modifying the `ChannelTimelineConfig` for that channel:

```python
channels = {
    1: ChannelConfig(
        name="Channel 1",
        # ... other settings ...
        timeline=ChannelTimelineConfig(
            # ... other overrides ...
            enable_performance_monitoring=True,  # Enable just for this channel
            track_memory_usage=True,
            performance_output_dir="outputs/channel1/performance"
        )
    )
}
```

### Command-Line Options

When running scripts that support timeline rendering, you can enable or configure performance monitoring using command-line arguments:

```bash
python video_edit.py --channel 1 --enable-monitoring --track-memory --save-perf-reports --performance-dir outputs/custom_perf
```

Available command-line options:

- `--enable-monitoring`: Enable performance monitoring for rendering
- `--track-memory`: Track memory usage during rendering
- `--save-perf-reports`: Save performance reports to disk
- `--performance-dir`: Directory for performance reports

### Environment Variables

You can also use environment variables to control performance monitoring:

- `PERFORMANCE_OUTPUT_DIR`: Directory where performance reports are saved
- `PERFORMANCE_MONITORING_ENABLED`: Set to "1" to enable monitoring globally

Example:
```bash
PERFORMANCE_OUTPUT_DIR=outputs/performance_tests python video_edit.py --channel 1
```

## Core Components

### PerformanceMonitor

The `PerformanceMonitor` class is the foundation of the performance monitoring system. It provides tools for measuring, tracking, and analyzing performance metrics throughout the application.

Key functionality:

- Creating and managing performance reports
- Adding custom metrics
- Measuring memory usage
- Saving reports to disk
- Comparing different reports
- Generating statistical summaries

Example usage:

```python
from logging_system.performance_monitor import PerformanceMonitor, MetricType

# Create a monitor
monitor = PerformanceMonitor(name="my_function", enable_memory_tracking=True)

# Start tracking a specific operation
monitor.start_tracking("important_operation")

# Record a custom metric
monitor.add_metric(
    name="processing_time",
    value=1.234,
    metric_type=MetricType.DURATION,
    unit="seconds"
)

# Measure memory usage
memory_mb = monitor.measure_memory()

# Complete tracking and get the report
report = monitor.complete_tracking()

# Save reports to disk
report_path = monitor.save_reports()
```

### RenderingPerformanceTracker

The `RenderingPerformanceTracker` class extends `PerformanceMonitor` with rendering-specific metrics and analysis capabilities. It's designed specifically for tracking timeline rendering performance.

Key functionality:

- Tracking direct vs. fallback rendering approaches
- Analyzing timeline complexity
- Recording frame processing times
- Tracking audio processing
- Monitoring I/O operations
- Comparing rendering approaches
- Generating optimization recommendations

Example usage:

```python
from logging_system.performance_monitor import RenderingPerformanceTracker

# Create a tracker for a specific channel
tracker = RenderingPerformanceTracker(
    name=f"rendering_channel_1", 
    enable_memory_tracking=True
)

# Start tracking direct rendering
tracker.start_direct_rendering_tracking(timeline)

# Record frame processing
tracker.record_frame_processed(frame_index=1, processing_time=0.05)

# Record I/O operation
tracker.record_io_operation(
    operation="write", 
    size_mb=120.5, 
    duration=1.5
)

# Complete direct rendering tracking
tracker.complete_direct_rendering()

# Generate and save report
performance_report = tracker.generate_performance_report()
tracker.save_reports()
```

### Performance Decorators

The `timing_decorator` allows you to easily track the execution time of any function:

```python
from logging_system.performance_monitor import timing_decorator

# Basic usage
@timing_decorator
def my_function(arg1, arg2):
    # Function body
    pass

# Advanced usage with custom name and memory tracking
@timing_decorator(name="custom_metric_name", track_memory=True)
def another_function():
    # Function body
    pass

# Using with a specific monitor
monitor = PerformanceMonitor(name="module_monitor")

@timing_decorator(monitor=monitor)
def tracked_function():
    # Function body
    pass
```

## Using Performance Monitoring

### Enabling Monitoring

To enable performance monitoring for timeline rendering:

1. **Through configuration**:
   Edit `config.py` to set `enable_performance_monitoring=True` in `TimelineRenderingConfig`

2. **For a specific channel**:
   Add `enable_performance_monitoring=True` to the channel's `ChannelTimelineConfig`

3. **Through command-line**:
   Run with `--enable-monitoring` flag

```python
# In code
from config import get_timeline_config

# Get configuration with monitoring enabled
timeline_config = get_timeline_config(channel_number=1)
if timeline_config.rendering.enable_performance_monitoring:
    # Monitoring is enabled
    pass
```

### Tracking Custom Metrics

You can track any custom metrics relevant to your specific use case:

```python
from logging_system.performance_monitor import PerformanceMonitor, MetricType

monitor = PerformanceMonitor(name="custom_tracking")

# Start tracking
monitor.start_tracking("my_custom_process")

# Add various types of metrics
monitor.add_metric(
    name="processing_chunks",
    value=25,
    metric_type=MetricType.CUSTOM,
    unit="chunks",
    context={"chunk_size": 1024}
)

monitor.add_metric(
    name="compression_ratio",
    value=0.75,
    metric_type=MetricType.CUSTOM,
    unit="ratio"
)

# Complete tracking
report = monitor.complete_tracking()

# Save reports
monitor.save_reports()
```

### Tracking Memory Usage

The monitoring system can track memory usage during operations:

```python
# Create monitor with memory tracking enabled
monitor = PerformanceMonitor(name="memory_tracking", enable_memory_tracking=True)

# Start tracking
monitor.start_tracking("memory_intensive_operation")

# Record the baseline memory
monitor.measure_memory("baseline")

# Perform memory-intensive operation
# ...

# Measure again
monitor.measure_memory("after_operation")

# Complete tracking
report = monitor.complete_tracking()

# Get memory statistics
memory_stats = report.get_statistics(metric_type=MetricType.MEMORY)
print(f"Average memory usage: {memory_stats['mean']} MB")
```

### Rendering Performance

The specialized `RenderingPerformanceTracker` is integrated with the timeline rendering system in `perf_render_timeline.py`. When performance monitoring is enabled, this tracker automatically collects metrics during rendering operations.

Key metrics captured:

- Timeline complexity assessment
- Frame-by-frame processing times
- Audio processing duration
- I/O operations for reading and writing
- Memory usage during rendering
- Comparative analysis between direct and fallback rendering methods

## Performance Reports

### Report Structure

Performance reports contain comprehensive information about the monitored operations:

```json
{
  "monitor_name": "rendering_channel_1",
  "timestamp": "2025-03-19T10:15:30.123456",
  "reports": [
    {
      "name": "direct_rendering",
      "start_time": 1711019730.123,
      "end_time": 1711019735.456,
      "duration": 5.333,
      "metrics": [
        {
          "type": "duration",
          "name": "frame_processing",
          "value": 0.033,
          "unit": "seconds",
          "timestamp": 1711019730.156,
          "context": {
            "frame_index": 1,
            "frame_count": 1
          }
        },
        // Additional metrics...
      ]
    },
    // Additional reports...
  ]
}
```

### Saving and Loading Reports

Reports are automatically saved to the directory specified in the configuration (`performance_output_dir`). The file names include timestamps for easy identification.

You can manually save reports using:

```python
tracker.save_reports()  # Uses default location from config

# Or specify a custom path
from pathlib import Path
custom_path = Path("custom/path/report.json")
tracker.save_reports(file_path=custom_path)
```

### Analyzing Results

The `PerformanceReport` class provides methods for analyzing and comparing results:

```python
# Get statistics for a specific metric
frame_stats = report.get_statistics(metric_name="frame_processing")
print(f"Average frame processing time: {frame_stats['mean']} seconds")

# Compare two reports
comparison = report1.compare_with(report2)
for metric_name, diff in comparison.items():
    print(f"{metric_name}: {diff['percent_diff']}% difference")
```

The `RenderingPerformanceTracker` provides specialized methods for rendering analysis:

```python
# Compare direct and fallback rendering
comparison = tracker.compare_rendering_approaches()
summary = comparison["summary"]

print(f"Direct rendering: {summary['direct_duration']:.2f} seconds")
print(f"Fallback rendering: {summary['fallback_duration']:.2f} seconds")
print(f"Speedup: {summary['speedup_factor']:.2f}x ({summary['speedup_percent']:.1f}%)")

# Get performance recommendations
report = tracker.generate_performance_report()
for recommendation in report["recommendations"]:
    print(f"- {recommendation}")
```

## Extending the System

### Adding New Metric Types

To add new metric types, extend the `MetricType` enum in `performance_monitor.py`:

```python
class MetricType(Enum):
    """Types of performance metrics that can be collected."""
    DURATION = "duration"  # Time measurements
    MEMORY = "memory"      # Memory usage
    IO = "io"              # I/O operations
    CPU = "cpu"            # CPU usage
    FRAMES = "frames"      # Frame processing stats
    CUSTOM = "custom"      # Custom metrics
    GPU = "gpu"            # New metric type for GPU operations
```

### Creating Custom Trackers

You can create specialized trackers for specific parts of your application:

```python
from logging_system.performance_monitor import PerformanceMonitor, MetricType

class AudioProcessingTracker:
    """Specialized tracker for audio processing operations."""
    
    def __init__(self, name="audio_processing", enable_memory_tracking=False):
        self.monitor = PerformanceMonitor(name=name, enable_memory_tracking=enable_memory_tracking)
        
    def start_processing(self, audio_file=None):
        """Start tracking audio processing."""
        self.monitor.start_tracking("audio_processing")
        
        # Record audio file details if provided
        if audio_file:
            self.monitor.add_metric(
                name="audio_file_size",
                value=audio_file.stat().st_size / (1024 * 1024),
                metric_type=MetricType.IO,
                unit="MB",
                context={"file_path": str(audio_file)}
            )
    
    def record_sample_processing(self, sample_count, processing_time):
        """Record sample processing metrics."""
        self.monitor.add_metric(
            name="sample_processing",
            value=processing_time,
            metric_type=MetricType.DURATION,
            unit="seconds",
            context={"sample_count": sample_count}
        )
    
    def complete_processing(self):
        """Complete tracking and return the report."""
        return self.monitor.complete_tracking()
```

## Best Practices

### When to Enable Monitoring

Performance monitoring adds some overhead, so it's best to enable it selectively:

- **During development**: Enable monitoring to identify bottlenecks
- **For benchmarking**: Enable when comparing different approaches
- **For optimization**: Enable when optimizing specific operations
- **For troubleshooting**: Enable when investigating performance issues

Disable monitoring in production unless you're specifically investigating an issue.

### Optimizing Performance

Based on monitoring results, you can optimize rendering performance:

1. **Direct vs. Fallback**: Compare direct and fallback rendering to determine which is faster for your specific timelines
2. **Frame Processing**: Look for spikes in frame processing time that might indicate bottlenecks
3. **Memory Usage**: Monitor memory consumption to avoid out-of-memory errors
4. **I/O Operations**: Optimize read/write operations, especially for large files
5. **Timeline Complexity**: Simplify complex timelines or break them into smaller parts

### Troubleshooting

If you encounter issues with the monitoring system:

1. **Memory Tracking Errors**: Ensure tracemalloc is imported and started before using
2. **Report Saving Failures**: Check permissions for the output directory
3. **High Overhead**: Disable memory tracking if it's causing too much overhead
4. **Missing Metrics**: Verify that tracking starts and completes properly

## API Reference

For complete API details, refer to the source code in `logging_system/performance_monitor.py`.

### Key Classes

- `PerformanceMonitor`: Base class for performance monitoring
- `RenderingPerformanceTracker`: Specialized tracker for rendering operations
- `PerformanceMetric`: Individual performance measurement
- `PerformanceReport`: Collection of performance metrics with analysis

### Key Functions

- `timing_decorator`: Decorator for tracking function execution time

### Enums

- `MetricType`: Types of performance metrics (DURATION, MEMORY, IO, CPU, FRAMES, CUSTOM)