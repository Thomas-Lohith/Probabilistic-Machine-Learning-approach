import os
import time
import argparse
import pandas as pd

def merge_csv_in_date_folder(date_folder, sensor_col):
    """
    Merge 'time' and a chosen sensor column from CSV files in csv_acc folder of a given date folder.
    Save as parquet file named <date>.parquet in the output_dir.
    """
    start_time = time.time()
    csv_acc_path = os.path.join(date_folder, "csv_acc")
    
    
    if not os.path.exists(csv_acc_path):
        print(f"Skipping {date_folder} (no csv_acc folder)")
        return
    
 # Use generator instead of list append → saves memory and time
    dfs = (
        pd.read_csv(os.path.join(csv_acc_path, f), usecols=["time", sensor_col], sep=";")
        for f in os.listdir(csv_acc_path) if f.endswith(".csv")
    )
    
    try:
        final_df = pd.concat(dfs, ignore_index=True)
    except ValueError:
        print(f"No valid CSVs found in {csv_acc_path}")
        return
    # Convert time column to datetime if possible
    try:
        final_df["time"] = pd.to_datetime(final_df["time"])
    except Exception:
        print(f"Could not convert time column in {date_folder}, saving as-is")

    # Sort by time
    final_df = final_df.sort_values(by="time").reset_index(drop=True)


    date_name = os.path.basename(date_folder.rstrip("/"))
    output_file = os.path.join(date_folder, f"{date_name}.parquet")
    final_df.to_parquet(output_file, index=False)
    end_time = time.time()
    print(f" Saved {output_file} ({len(final_df)} rows)")
    print(f"Time taken: {end_time - start_time:.2f} seconds")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge sensor data from CSVs in csv_acc folders into parquet files per date")
    parser.add_argument("--parent", type=str, required=True, help="Path to parent folder containing date folders")
    parser.add_argument("--sensor", type=str, required=True, help="Sensor column name (e.g., 03091002_x)")
    #parser.add_argument("--output", type=str, required=True, help="Directory to save parquet files")

    args = parser.parse_args()

    # Process each date folder
    date_folders = [f for f in os.listdir(args.parent) if os.path.isdir(os.path.join(args.parent, f))]
    total_dates = len(date_folders)

    for idx, folder_name in enumerate(sorted(date_folders), start=1):
        date_folder = os.path.join(args.parent, folder_name)
        print(f"\n=== [{idx}/{total_dates}] Processing date folder: {folder_name} ===")
        merge_csv_in_date_folder(date_folder, args.sensor)


# ex: python merge_csv.py --parent /Users/thomas/data --sensor 03091002_x --output /Users/thomas/merged_parquet