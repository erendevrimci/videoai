import csv
import os
import sys

def escape_sql_string(s):
    if s is None or s == '':
        return 'NULL'
    # Escape single quotes
    escaped_val = s.replace("'", "''")
    return f"'{escaped_val}'"

def split_csv_to_batches(csv_path, output_dir, batch_size=100, start_from=0):
    """
    Split CSV data into smaller batch files for easier import
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip header row
        
        # Skip rows we've already processed
        for _ in range(start_from):
            next(reader, None)
        
        batch_number = 0
        row_count = 0
        batch_rows = []
        
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
            
            # Add to batch
            value_str = f"({', '.join(processed_values)})"
            batch_rows.append(value_str)
            row_count += 1
            
            # Write batch file when we reach the batch size
            if len(batch_rows) >= batch_size:
                batch_file = os.path.join(output_dir, f"batch_{batch_number}.sql")
                with open(batch_file, 'w', encoding='utf-8') as f:
                    cols = ", ".join(headers)
                    f.write(f"-- Batch {batch_number} ({len(batch_rows)} rows)\n")
                    f.write(f"INSERT INTO public.video_clips ({cols}) VALUES\n")
                    f.write(",\n".join(batch_rows))
                    f.write(";\n")
                
                batch_rows = []
                batch_number += 1
                print(f"Created batch {batch_number} with {batch_size} rows")
        
        # Write any remaining rows
        if batch_rows:
            batch_file = os.path.join(output_dir, f"batch_{batch_number}.sql")
            with open(batch_file, 'w', encoding='utf-8') as f:
                cols = ", ".join(headers)
                f.write(f"-- Batch {batch_number} ({len(batch_rows)} rows)\n")
                f.write(f"INSERT INTO public.video_clips ({cols}) VALUES\n")
                f.write(",\n".join(batch_rows))
                f.write(";\n")
            
            print(f"Created batch {batch_number} with {len(batch_rows)} rows")
            batch_number += 1
        
        print(f"Created {batch_number} batch files with {row_count} total rows")
        print(f"Files are in {output_dir}/")

# Process the CSV data
if __name__ == "__main__":
    start_from = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    split_csv_to_batches(
        '/Volumes/drev-ventura/video-repos/videoai/video/video-catalog-adjusted.csv',
        '/Volumes/drev-ventura/video-repos/videoai/sql_batches',
        100,  # 100 rows per batch
        start_from
    )