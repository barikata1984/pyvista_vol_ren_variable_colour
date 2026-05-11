import argparse
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-dir")
    args = parser.parse_args()

    print(f"{args=}\n")

    def check_pathobj(item, filename):
        return item.is_file() and item.name.startswith(filename) and item.name.endswith(".csv")

    object_dir = Path(args.object_dir)  # Ensure directory path is a Path object
    csv_filename = "_ground_truth_"
    csv_files = [str(item) 
                 for item 
                 in object_dir.iterdir() 
                 if check_pathobj(item, csv_filename)]
    csv_files.sort()

    print(f"{csv_files=}")

    df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files],
                   ignore_index=True)

    print(f"{df=}")

    csv_filepath = (object_dir / csv_filename).with_suffix(".csv")
    df.to_csv(csv_filepath, index=False)

    with open(object_dir / "joined_ground_truth.csv", 'w') as target:
        
        with open(object_dir / "metadata_ground_truth.csv", 'r') as metadata:
            metadata_lines= metadata.readlines()
       
            with open(csv_filepath, 'r') as mass_distr:
                content = mass_distr.readlines()
                for metadata_line in reversed(metadata_lines):
                    content.insert(0, metadata_line)
        
        target.writelines(content)

    Path.unlink(csv_filepath)
