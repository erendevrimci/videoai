import pandas as pd
import os
import shutil
from pathlib import Path

# Define paths for easier configuration
base_dir = Path(__file__).resolve().parent
catalog_path = base_dir / 'video' / 'video-catalog-labeled.csv'
filtered_catalog_path = base_dir / 'video' / 'video-catalog-labeled-filtered.csv'
target_folder = base_dir / 'filtered_videos'

# Common video extensions to check
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm']

def filter_video_catalog():
    """
    Filter videos from the catalog by aspect ratio and copy the matching videos
    to the target folder. Updates the CSV with new path information.
    """
    print(f"Reading catalog from {catalog_path}")
    df = pd.read_csv(catalog_path)
    
    # Extract aspect ratio components
    df['a'] = df['aspect_ratio'].str.split(':', expand=True)[0].astype(float)
    df['b'] = df['aspect_ratio'].str.split(':', expand=True)[1].astype(float)
    
    # Filter for horizontal videos (width < height, like mobile videos)
    filtered_df = df[df['a'] < df['b']]
    filtered_df = filtered_df.drop(columns=['a', 'b'], axis=1)
    
    print(f"Found {len(filtered_df)} videos matching filter criteria")
    
    # Move videos and update paths
    filtered_df['path'] = filtered_df['path'].apply(lambda row: move_videos(row))
    
    # Save filtered catalog
    filtered_df.to_csv(filtered_catalog_path, index=False)
    print(f"Saved filtered catalog to {filtered_catalog_path}")

def find_video_file(path_str):
    """
    Find a video file at the given path, trying multiple approaches:
    1. Check exact path
    2. Check path relative to base directory
    3. Check path with different extensions
    
    Args:
        path_str: String path to the video file
    
    Returns:
        Path object if found, None otherwise
    """
    # Check if it's already a full path
    path = Path(path_str)
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return path
    
    # Check if it's relative to base directory
    base_path = base_dir / path
    if base_path.is_file() and base_path.suffix.lower() in VIDEO_EXTENSIONS:
        return base_path
    
    # Check inside clips directory
    clips_path = base_dir / 'clips' / path.name
    if clips_path.is_file() and clips_path.suffix.lower() in VIDEO_EXTENSIONS:
        return clips_path
    
    # Check inside video directory
    video_path = base_dir / 'video' / path.name
    if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
        return video_path
    
    # Try different extensions if the path doesn't have one
    if not path.suffix:
        for ext in VIDEO_EXTENSIONS:
            # Check with extension added to original path
            ext_path = Path(f"{path_str}{ext}")
            if ext_path.is_file():
                return ext_path
            
            # Check with extension added to base path
            base_ext_path = base_dir / f"{path}{ext}"
            if base_ext_path.is_file():
                return base_ext_path
            
            # Check clips directory with extension
            clips_ext_path = base_dir / 'clips' / f"{path.name}{ext}"
            if clips_ext_path.is_file():
                return clips_ext_path
            
            # Check video directory with extension
            video_ext_path = base_dir / 'video' / f"{path.name}{ext}"
            if video_ext_path.is_file():
                return video_ext_path
    
    # If we still haven't found it, check for similar filenames
    try:
        clips_dir = base_dir / 'clips'
        if clips_dir.exists():
            for file in clips_dir.glob("*"):
                if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS:
                    # Check if the filename is similar (contains the path's stem)
                    if path.stem.lower() in file.stem.lower():
                        return file
    except Exception as e:
        print(f"Error searching for similar files: {e}")
    
    return None

def move_videos(row):
    """
    Copy a video file to the target folder and return the new path.
    
    Args:
        row: Path string from the CSV
    
    Returns:
        New path if copied successfully, original path otherwise
    """
    # Ensure target folder exists
    target_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        # Find the source video file
        source_file = find_video_file(row)
        
        if source_file is not None:
            # Create the target path using the original filename
            video_name = source_file.name
            target_path = target_folder / video_name
            
            # Copy the file
            shutil.copy2(source_file, target_path)
            print(f"Successfully copied: {video_name}")
            
            # Return relative path to keep CSV portable
            return str(Path('filtered_videos') / video_name)
        else:
            print(f"Warning: Could not find video file: {row}")
            return row  # Keep original path if file not found
    except Exception as e:
        print(f"Error copying {row}: {e}")
        return row  # Keep original path on error

if __name__ == "__main__":
    filter_video_catalog()