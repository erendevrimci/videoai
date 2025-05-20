import csv
import os

def create_copy_statement(csv_path, table_name):
    # Read the first row to get column names
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
    
    # Create a COPY command that will load data from CSV file
    columns = ", ".join(headers)
    
    # Generate temp path for CSV (will be the same file but with proper location)
    csv_basename = os.path.basename(csv_path)
    temp_path = f"/tmp/{csv_basename}"
    
    # Create the COPY command
    copy_cmd = f"""
-- Create a temporary function to load data from CSV
CREATE OR REPLACE FUNCTION pg_temp.load_csv_data() RETURNS void AS $$
BEGIN
    COPY {table_name} ({columns})
    FROM '{temp_path}'
    WITH (FORMAT csv, HEADER true);
END;
$$ LANGUAGE plpgsql;

-- Execute the function
SELECT pg_temp.load_csv_data();

-- Report count
SELECT COUNT(*) FROM {table_name};
"""
    
    # Write the COPY command to a file
    output_file = f"import_{table_name}.sql"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(copy_cmd)
    
    print(f"Created SQL import script: {output_file}")
    print(f"You will need to copy your CSV file to: {temp_path}")
    print(f"Then run the SQL script with: psql -f {output_file}")

# Create the import script
create_copy_statement(
    '/Volumes/drev-ventura/video-repos/videoai/video/video-catalog-adjusted.csv',
    'public.video_clips'
)