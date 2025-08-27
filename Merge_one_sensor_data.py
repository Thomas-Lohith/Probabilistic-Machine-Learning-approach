import os
import time
import argparse
import pandas as pd

def merge_csv_from_folders(parent_folder, sensor_col, output_file):
    """
    Merge 'time' and a chosen sensor column from all CSV files inside date-named folders.
    """
    start_time = time.time()
    merged_data = []

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(folder_path, file_name)
                    
                    try:
                        df = pd.read_csv(file_path, usecols=["time", sensor_col])
                        merged_data.append(df)
                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")

    if merged_data:
        final_df = pd.concat(merged_data, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        end_time = time.time()
        print(f" Merged CSV saved to {output_file}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    else:
        print(" No data merged. Check column names or folder structure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge time + sensor data from multiple CSVs in dated folders")
    parser.add_argument("--parent_folder", type=str, required=True, help="Path to parent folder containing dated subfolders")
    parser.add_argument("--sensor", type=str, required=True, help="Sensor column name (e.g., 03091002_x)")
    parser.add_argument("--output", type=str, required=True, help="Path to save merged CSV")

    args = parser.parse_args()

    merge_csv_from_folders(args.parent_folder, args.sensor, args.output)


    ###python merge_csv.py --parent_folder /Users/thomas/data --sensor 03091002_x --output merged_output.csv###