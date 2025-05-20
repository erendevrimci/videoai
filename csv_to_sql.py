import csv
import re

def escape_sql_string(s):
    if s is None or s == '':
        return 'NULL'
    # Replace single quotes with two single quotes for SQL escape
    s = s.replace("'", "''")
    return f"'{s}'"

def process_csv_to_sql_inserts(csv_path, output_path, limit=100):
    with open(csv_path, 'r', encoding='utf-8') as csvfile, open(output_path, 'w', encoding='utf-8') as outfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip header row
        
        # Write SQL statements
        outfile.write("-- Insert statements for video_clips table\n\n")
        
        count = 0
        for row in reader:
            if count >= limit:
                break
                
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
            outfile.write(insert_stmt + "\n")
            
            count += 1
        
        outfile.write(f"\n-- Generated {count} insert statements\n")

# Process the CSV file
process_csv_to_sql_inserts(
    '/Volumes/drev-ventura/video-repos/videoai/video/video-catalog-adjusted.csv',
    '/Volumes/drev-ventura/video-repos/videoai/video_clips_insert.sql',
    100  # Limit to first 100 rows
)