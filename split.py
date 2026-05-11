import argparse
from pathlib import Path

import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-dir")
    parser.add_argument("--num-split", type=int, default=4)
    parser.add_argument("--lod", type=int, default=7)
    args = parser.parse_args()

    object_dir = Path(args.object_dir)
    csv_filepath = object_dir / "ground_truth.csv"

    # Main part ========================================================
    meta_md_header = pd.read_csv(csv_filepath,
                                 nrows=1,  # from metadata header to mass distr header
                                 )

    meta_md_header.to_csv(csv_filepath.with_stem(f"metadata_{csv_filepath.stem}").with_suffix(".csv"),
                          #header=False,
                          index=False,
                          )

    md_df= pd.read_csv(csv_filepath,
                       skiprows=2,  # exclude the metadata and its header
                       )

    print(f"{md_df.to_numpy().shape=}")

    # load the .csv and plot w/ matplotlib =============================
    num_rows = md_df.shape[0]
    lod = args.lod
    resolution = 2**lod
    num_voxels = resolution**3
    num_split = args.num_split

    num_rows_per_split = num_voxels / num_split
    for j in range(num_split, 0, -1):
        print(f"slice range: ("
              f"{int(num_rows_per_split * (j - 1))}, "
              f"{int(num_rows_per_split * j)})")
        start = num_rows_per_split * (j - 1)
        stop = 0 if 0 == j else num_rows_per_split * j

        new_csv_filepath = csv_filepath.with_stem(f"_{csv_filepath.stem}_{j}").with_suffix(".csv")
        md_df[int(start):int(stop)].to_csv(new_csv_filepath,
                                           #header=False,
                                           index=False,
                                           )

