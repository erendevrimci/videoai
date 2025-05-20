import csv
import re
import os

def escape_sql_string(s):
    if s is None or s == '':
        return 'NULL'
    # Replace single quotes with two single quotes for SQL escape
    s = s.replace("'", "''")
    return f"'{s}'"

def process_csv_to_sql_batches(csv_path, output_dir, batch_size=50):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip header row
        
        batch_count = 0
        row_count = 0
        batch = []
        
        for row in reader:
            # Skip empty rows
            if not any(row):
                continue
                
            # Process numeric values
            values = []
            for j, val in enumerate(row):
                if headers[j] == 'fps' and val.strip():
                    # Handle numeric column
                    values.append(val)
                else:
                    values.append(escape_sql_string(val))
            
            # Create insert statement
            columns = ", ".join(headers)
            values_str = ", ".join(values)
            insert_stmt = f"INSERT INTO public.video_clips ({columns}) VALUES ({values_str});"
            batch.append(insert_stmt)
            row_count += 1
            
            # Output batch when it reaches batch_size
            if len(batch) >= batch_size:
                batch_file = os.path.join(output_dir, f"batch_{batch_count}.sql")
                with open(batch_file, 'w', encoding='utf-8') as f:
                    f.write(f"-- Batch {batch_count} ({len(batch)} rows)\n\n")
                    f.write("\n".join(batch))
                    f.write(f"\n\n-- End of batch {batch_count}\n")
                
                batch = []
                batch_count += 1
        
        # Output any remaining statements in the last batch
        if batch:
            batch_file = os.path.join(output_dir, f"batch_{batch_count}.sql")
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write(f"-- Batch {batch_count} ({len(batch)} rows)\n\n")
                f.write("\n".join(batch))
                f.write(f"\n\n-- End of batch {batch_count}\n")
            
            batch_count += 1
        
        print(f"Generated {batch_count} batch files with {row_count} total rows")
        
        # Create a batch execution script
        with open(os.path.join(output_dir, "execute_all.sql"), 'w', encoding='utf-8') as f:
            f.write("-- Execute all batches\n\n")
            for i in range(batch_count):
                f.write(f"\\i batch_{i}.sql\n")

# Process the CSV file
process_csv_to_sql_batches(
    '/Volumes/drev-ventura/video-repos/videoai/video/video-catalog-adjusted.csv',
    '/Volumes/drev-ventura/video-repos/videoai/sql_batches',
    100  # 100 rows per batch
)