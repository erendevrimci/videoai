#\!/bin/bash

# Script to remove duplicate files from clips/not_filtered that already exist in clips

# Get the list of files in the main clips directory
main_clips=$(ls -1 clips | grep -v "not_filtered" | grep -v "filtered_videos" | grep -v "sample_clips" | grep -v "\.DS_Store")

# Counter for removed files
removed=0

# Loop through each file in the main clips directory
for file in $main_clips; do
  # Check if the file exists in the not_filtered directory
  if [ -f "clips/not_filtered/$file" ]; then
    # Remove the duplicate file from not_filtered
    rm "clips/not_filtered/$file"
    echo "Removed duplicate: $file"
    ((removed++))
  fi
done

echo "Removed $removed duplicate files from clips/not_filtered"
