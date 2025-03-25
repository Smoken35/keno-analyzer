#!/usr/bin/env python3
import os
import sys
from data_processor import KenoDataProcessor

def print_usage():
    print("""
Keno Data Update Tool
--------------------
Usage:
    python update_data.py <file_path>
    
    file_path: Path to the CSV or JSON file containing Keno results
    
Example:
    python update_data.py keno_results.csv
    python update_data.py keno_results.json
    """)

def main():
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist")
        sys.exit(1)
        
    processor = KenoDataProcessor()
    success, message = processor.process_uploaded_file(file_path)
    
    print(message)
    if success:
        print("\nLatest results:")
        latest = processor.get_latest_results()
        for result in latest:
            print(f"Draw {result['draw_number']}: {result['draw_time']}")

if __name__ == '__main__':
    main() 