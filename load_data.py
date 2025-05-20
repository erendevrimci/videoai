import csv
import sys
import os

def escape_sql_string(s):
    if s is None or s == '':
        return 'NULL'
    # Replace single quotes with two single quotes for SQL escape
    s = s.replace("'", "''")
    return f"'{s}'"

def create_insert_statement(row, headers):
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
    return f"INSERT INTO public.video_clips ({columns}) VALUES ({values_str});"

def main():
    # Parameters
    csv_path = '/Volumes/drev-ventura/video-repos/videoai/video/video-catalog-adjusted.csv'
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # Start index (0-based)
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10     # Number of rows to process
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip header row
        
        # Skip to the starting position
        for _ in range(start_idx):
            next(reader, None)
        
        # Generate insert statements for the specified number of rows
        inserts = []
        for i, row in enumerate(reader):
            if i >= count:
                break
                
            # Skip empty rows
            if not any(row):
                continue
                
            insert_stmt = create_insert_statement(row, headers)
            inserts.append(insert_stmt)
        
        # Print the SQL statements
        for stmt in inserts:
            print(stmt)

if __name__ == "__main__":
    main()