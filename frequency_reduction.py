import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar

def process_sensor_files(root_dir, sensor_name, output_csv, chunk_size=100):
    """
    Reduce frequency of sensor data by computing mean and variance every 'chunk_size' samples.
    Applies log-normal transform on variance.

    Parameters:
    - root_dir (str): Root folder containing date subfolders with hourly CSVs.
    - sensor_name (str): Sensor column name to process (e.g., "acc1_z").
    - output_csv (str): Path to save the aggregated results.
    - chunk_size (int): Number of samples to group together (default=100).
    """

    results = []

    # Walk through date folders
    for day_folder in sorted(os.listdir(root_dir)):
        day_path = os.path.join(root_dir, day_folder)
        if not os.path.isdir(day_path):
            continue

        # --- enter csv_acc folder ---
        csv_acc_path = os.path.join(day_path, "csv_acc")
        if not os.path.exists(csv_acc_path):
            print(f"Skipping {day_folder}: no csv_acc folder found")
            continue

        print(f"\nProcessing {day_folder} ...")

        # Process hourly CSV files inside csv_acc folder
        for file in tqdm(sorted(os.listdir(csv_acc_path)), desc=f"{day_folder}", unit="file"):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(csv_acc_path, file)

            # Read only the sensor column (and time if present)
            try:
                print('\n\n\n', file_path)
                df = pd.read_csv(file_path, usecols=["time", f'{sensor_name}'], sep=';')
            except ValueError:
                # If time not present, fallback
                df = pd.read_csv(file_path, usecols=[sensor_name], sep=';')
                df["time"] = range(len(df))

            # Process in chunks
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    continue  # skip incomplete chunk at end

                mean_val = chunk[sensor_name].mean()
                var_val = chunk[sensor_name].var()
                log_var = log_normal_variance(var_val)

                results.append({
                    "day": day_folder,
                    "hour_file": file,
                    "start_time": chunk["time"].iloc[0],
                    "end_time": chunk["time"].iloc[-1],
                    "mean": mean_val,
                    "variance": var_val,
                    "log_variance": log_var,
                })

    # Save results to a single CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\n Reduced dataset saved to {output_csv} ({len(results_df)} rows)")
    else:
        print("\n No data processed.")


def log_normal_variance(variance):
    """Apply log-normal transform safely to variance values."""
    if variance <= 0 or pd.isna(variance):
        return np.nan
    return np.log(variance)


def main():
    parser = argparse.ArgumentParser(description="Reduce frequency of sensor CSV data (per 100 samples)")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to parent folder containing date folders")
    parser.add_argument("--sensor_channel", type=str, required=True, help="Sensor column name (e.g., 03091002_x)")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of samples per averaging chunk")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")

    args = parser.parse_args()

    process_sensor_files(args.root_dir, args.sensor_channel, args.output, args.chunk_size)



#examplehow to use the script: python3 frequency_reduction.py --root_dir /Users/thomas/Data/Data_sensors --sensor_channel 030911EF_x --output /Users/thomas/Data/Data_sensors/output.csv

if __name__ == "__main__":
    main()