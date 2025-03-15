#\!/bin/bash

# Create a list of all unique filenames in the clips directory (excluding subdirectories)
find clips -maxdepth 1 -type f -not -path "*/\.*" | sort > clips_files.txt

# Count how many files we need to check
total_files=$(cat clips_files.txt | wc -l)
echo "Found $total_files files in main clips directory to check for duplicates"

# Counter for removed files
removed=0

# Process each file in the main clips directory
while read filepath; do
  # Extract just the filename
  filename=$(basename "$filepath")
  
  # Check if this file exists in not_filtered
  if [ -f "clips/not_filtered/$filename" ]; then
    # Remove the duplicate from not_filtered
    rm "clips/not_filtered/$filename"
    echo "Removed duplicate: $filename"
    ((removed++))
  fi
done < clips_files.txt

echo "Removed $removed duplicate files from clips/not_filtered"

# Clean up
rm clips_files.txt
