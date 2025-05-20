import csv
import json

def process_csv_to_bulk_insert(csv_path, output_file, batch_size=500):
    """
    Processes a CSV file into a SQL script that performs bulk inserts
    """
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip header row
        
        # Build the INSERT statement template
        columns = ", ".join(headers)
        
        # Open the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write the opening of the transaction
            f.write("BEGIN;\n\n")
            
            batch_count = 0
            row_count = 0
            values_parts = []
            
            for row in reader:
                # Skip empty rows
                if not any(row):
                    continue
                
                # Process values
                processed_values = []
                for i, val in enumerate(row):
                    if headers[i] == 'fps' and val.strip():
                        # Handle numeric column
                        processed_values.append(val)
                    elif val.strip():
                        # Escape single quotes and use SQL string literal
                        escaped_val = val.replace("'", "''")
                        processed_values.append(f"'{escaped_val}'")
                    else:
                        processed_values.append("NULL")
                
                # Add value tuple to the batch
                values_parts.append(f"({', '.join(processed_values)})")
                row_count += 1
                
                # When batch size is reached, write the INSERT statement
                if len(values_parts) >= batch_size:
                    f.write(f"-- Batch {batch_count + 1}: {len(values_parts)} rows\n")
                    f.write(f"INSERT INTO public.video_clips ({columns}) VALUES\n")
                    f.write(",\n".join(values_parts))
                    f.write(";\n\n")
                    
                    values_parts = []
                    batch_count += 1
            
            # Write any remaining rows
            if values_parts:
                f.write(f"-- Batch {batch_count + 1}: {len(values_parts)} rows\n")
                f.write(f"INSERT INTO public.video_clips ({columns}) VALUES\n")
                f.write(",\n".join(values_parts))
                f.write(";\n\n")
                batch_count += 1
            
            # Write the commit and report count
            f.write("COMMIT;\n\n")
            f.write("-- Check the count\n")
            f.write("SELECT COUNT(*) FROM public.video_clips;\n")
            
            print(f"Generated SQL import script with {batch_count} batches and {row_count} total rows")

# Create the bulk insert script
process_csv_to_bulk_insert(
    '/Volumes/drev-ventura/video-repos/videoai/video/video-catalog-adjusted.csv',
    '/Volumes/drev-ventura/video-repos/videoai/bulk_insert.sql',
    500  # 500 rows per batch
)