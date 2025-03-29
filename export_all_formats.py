#!/usr/bin/env python3
"""
Export a timeline to all supported formats.

Usage:
    python export_all_formats.py --channel 1 --timeline final_timeline 
    
This will export the specified timeline to all supported formats:
- JSON
- FCP7 XML
- FCP11 FCPXML
- Shotcut MLT
"""

import argparse
from pathlib import Path
from export_manager import ExportManager
from timeline_manager import TimelineManager
from file_manager import FileManager

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Export a timeline to all supported formats")
    parser.add_argument("--channel", type=int, default=1, help="Channel number")
    parser.add_argument("--timeline", type=str, default="final_timeline", 
                        help="Timeline name (e.g., final_timeline, voice_timeline)")
    parser.add_argument("--output_dir", type=str, help="Optional output directory")
    args = parser.parse_args()
    
    # Initialize managers
    file_mgr = FileManager()
    timeline_mgr = TimelineManager(channel_number=args.channel)
    export_mgr = ExportManager(channel_number=args.channel)
    
    # Get the timeline path
    timeline_path = timeline_mgr.get_timeline_path(args.timeline, args.channel)
    
    if not file_mgr.file_exists(timeline_path):
        print(f"Timeline not found: {timeline_path}")
        return
    
    print(f"Exporting timeline: {timeline_path}")
    
    # Prepare output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = file_mgr.get_channel_output_path(args.channel) / "exports"
    
    file_mgr.ensure_dir_exists(output_dir)
    
    # Get all supported formats
    formats = export_mgr.get_supported_formats()
    print(f"Supported formats: {formats}")
    
    # Export to all formats
    results = {}
    
    for format_type in formats:
        print(f"Exporting to {format_type}...")
        output_path = output_dir / f"{args.timeline}_{format_type}{export_mgr._get_format_extension(format_type)}"
        
        try:
            result_path = export_mgr.export_timeline(
                timeline=timeline_path,
                format_type=format_type,
                output_path=output_path
            )
            
            if result_path:
                print(f"✅ Successfully exported to {result_path}")
                results[format_type] = str(result_path)
            else:
                print(f"❌ Failed to export to {format_type}")
                results[format_type] = None
        except Exception as e:
            print(f"❌ Error exporting to {format_type}: {e}")
            results[format_type] = None
    
    # Print summary
    print("\nExport Summary:")
    for format_type, path in results.items():
        status = "✅ Success" if path else "❌ Failed"
        print(f"  {format_type}: {status}")
        if path:
            print(f"    Output: {path}")
    
    print("\nTo use exported timeline files:")
    print("- JSON: Can be imported back into VideoAI or used for interchange")
    print("- FCP7 XML: Import into Adobe Premiere Pro or Final Cut Pro 7")
    print("- FCP11 FCPXML: Import into Final Cut Pro X")
    print("- Shotcut MLT: Import into Shotcut editor")

if __name__ == "__main__":
    main()