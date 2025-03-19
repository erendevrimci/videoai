#!/usr/bin/env python3
"""
Script to fill clips_label.md with data from video-catalog-labeled.csv
"""
import csv
import os

def read_csv(csv_file_path):
    """Read the CSV file and return the data as a list of dictionaries."""
    data = []
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def generate_markdown_entries(csv_data):
    """Generate markdown entries from CSV data."""
    entries = []
    
    for i, row in enumerate(csv_data):
        # Skip the first row which is already in the file as an example
        if i == 0:
            continue
            
        # Ensure we have valid data
        if not all(key in row for key in ['path', 'short_prompt', 'labels', 'duration']):
            print(f"Warning: Row {i+1} is missing required fields. Skipping.")
            continue
            
        entry = (
            f"Name: {row['path']}.mp4\n"
            f"Description: {row['short_prompt']}\n"
            f"Notes: {row['labels']}\n"
            f"Length: {row['duration']}\n\n"
            f"---\n"
        )
        entries.append(entry)
    
    return entries

def update_markdown_file(md_file_path, entries):
    """Update the markdown file with generated entries."""
    # Keep the first entry as an example
    with open(md_file_path, 'r', encoding='utf-8') as mdfile:
        lines = mdfile.readlines()
    
    # Find the end of the first entry
    first_entry_end = 0
    separator_count = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            separator_count += 1
            if separator_count == 1:
                first_entry_end = i + 1
                break
    
    # Combine the first entry with the new entries
    updated_content = ''.join(lines[:first_entry_end]) + '\n' + '\n'.join(entries)
    
    # Write the updated content back to the file
    with open(md_file_path, 'w', encoding='utf-8') as mdfile:
        mdfile.write(updated_content)

def main():
    """Main function to coordinate the process."""
    # Define file paths
    csv_file_path = '/Volumes/drev-ventura/video-repos/videoai/clips/video-catalog-labeled.csv'
    md_file_path = '/Volumes/drev-ventura/video-repos/videoai/clips/clips_label1.md'
    
    # Check if files exist
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    
    if not os.path.exists(md_file_path):
        print(f"Error: Markdown file not found at {md_file_path}")
        return
    
    # Process the files
    csv_data = read_csv(csv_file_path)
    entries = generate_markdown_entries(csv_data)
    update_markdown_file(md_file_path, entries)
    
    print(f"Successfully updated {md_file_path} with {len(entries)} entries from {csv_file_path}")

if __name__ == "__main__":
    main()