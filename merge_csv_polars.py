import os
import time
import argparse
import polars as pl

def merge_csv_in_date_folder(date_folder, sensor_col):
    """
    Merge 'time' and a chosen sensor column from CSV files in csv_acc folder of a given date folder.
    Save as parquet file named <date>.parquet in the same folder.
    """
    start_time = time.time()
    csv_acc_path = os.path.join(date_folder, "csv_acc")

    if not os.path.exists(csv_acc_path):
        print(f"Skipping {date_folder} (no csv_acc folder)")
        return

    dfs = []
    for f in os.listdir(csv_acc_path):
        if f.endswith(".csv"):
            file_path = os.path.join(csv_acc_path, f)
            try:
                df = pl.read_csv(
                    file_path,
                    separator=";",
                    columns=["time", sensor_col]  # only load required cols
                )
                # 🔑Ensure sensor column is numeric
                df = df.with_columns(
                    pl.col(sensor_col).cast(pl.Float32, strict=False))
                dfs.append(df)
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")

    if dfs:
        # concatenate all frames
        final_df = pl.concat(dfs)

        # try to convert "time" to datetime if possible
        try:
            final_df = final_df.with_columns(
                pl.col("time").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M:%S:%3f", strict=False)
            )
        except Exception:
            print(f"Could not convert time column in {date_folder}, saving as-is")

        # sort by time
        final_df = final_df.sort("time")

        date_name = os.path.basename(date_folder.rstrip("/"))
        output_file = os.path.join(date_folder, f"{date_name}.parquet")
        final_df.write_parquet(output_file)
        end_time = time.time()
        print(f" Saved {output_file} ({final_df.height} rows)")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    else:
        print(f" No data merged for {date_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge sensor data from CSVs in csv_acc folders into parquet files per date"
    )
    parser.add_argument("--parent", type=str, required=True,
                        help="Path to parent folder containing date folders")
    parser.add_argument("--sensor", type=str, required=True,
                        help="Sensor column name (e.g., 03091002_x)")

    args = parser.parse_args()

    # process each date folder
    date_folders = [f for f in os.listdir(args.parent)
                    if os.path.isdir(os.path.join(args.parent, f))]
    total_dates = len(date_folders)

    for idx, folder_name in enumerate(sorted(date_folders), start=1):
        date_folder = os.path.join(args.parent, folder_name)
        print(f"\n=== [{idx}/{total_dates}] Processing date folder: {folder_name} ===")
        merge_csv_in_date_folder(date_folder, args.sensor)


# Example:
#  python3 merge_csv_polars.py --parent /data/pool/c8x-98x/test_merge_script/ --sensor 03091002_x 