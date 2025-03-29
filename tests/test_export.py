"""
Test script for export functionality
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from timeline_manager import TimelineManager, TlText
from export_manager import ExportManager, FORMAT_JSON, FORMAT_FCP7, FORMAT_FCP11, FORMAT_SHOTCUT

# Create output directory
output_dir = Path("tests/test_output")
if not output_dir.exists():
    os.makedirs(output_dir)

# Create timeline manager and export manager
timeline_manager = TimelineManager()
export_manager = ExportManager()

# Create a test timeline
timeline = timeline_manager.create_v3_timeline(
    width=1080,
    height=1920,
    framerate=30
)

# Add a text element
text_obj = TlText(
    start=0,
    dur=90,  # 3 seconds at 30fps
    text="Test Text",
    x=540,
    y=960,
    font="Arial",
    font_size=36,
    color="#FFFFFF"
)

# Add to the timeline
timeline.v[0].append(text_obj)

# Add videoai_metadata to make it valid
timeline.videoai_metadata = {
    "channel": None,
    "version": "1.0",
    "type": "v3",
    "created_at": datetime.now().isoformat(),
    "description": "Test timeline for export testing"
}

# Save the timeline
timeline_path = output_dir / "test_timeline_export.json"
timeline_manager.serialize_timeline(timeline, timeline_path)
print(f"Created test timeline: {timeline_path}")

# Export to different formats
formats = [FORMAT_JSON, FORMAT_FCP7, FORMAT_FCP11, FORMAT_SHOTCUT]

for fmt in formats:
    try:
        output_path = output_dir / f"test_timeline_export.{fmt}"
        result = export_manager.export_timeline(timeline, fmt, output_path)
        if result:
            print(f"Successfully exported to {fmt} format: {result}")
        else:
            print(f"Failed to export to {fmt} format")
    except Exception as e:
        print(f"Error exporting to {fmt} format: {e}")

print("Done testing exports")