import argparse
import json
import os

import pandas as pd
from datasets import load_dataset
from verl.utils.hdfs_io import copy, makedirs


from system_prompts import SWE_SYSTEM_PROMPT, SWE_REBENCH_USER_PROMPT, R2EGYM_SYSTEM_PROMPT, R2EGYM_USER_PROMPT


"""
SWE_DATASETS = [
    "R2E-Gym/R2E-Gym-Subset",
    "R2E-Gym/R2E-Gym-Lite",
    "R2E-Gym/R2E-Gym-V1",
    "R2E-Gym/SWE-Bench-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    "r2e-edits/SweSmith-RL-Dataset",
]
"""
SWE_DATASETS = ["/wh-test/tianrenbao/datasets/swe-rebench-v2/"]


def create_problem_statement(row):
    problem_statement = row['problem_statement']
    requirement = row['requirements']
    interface = row['interface']

    return f"""{problem_statement}

Requirements:
{requirement}

New interfaces introduced:
{interface}"""


def main():
    parser = argparse.ArgumentParser(description="Generate trajectories using specified environment and policy.")
    parser.add_argument("--local_dir", default=os.path.join('.', "data/swe"))
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    hdfs_dir = args.hdfs_dir

    def make_map_fn():
        def process_fn(row):
            row_dict = dict(row)
            # problem_statement = row_dict.get("problem_statement", "")
            problem_statement = create_problem_statement(row_dict)

            repo = row_dict.get("repo", "")
            # workdir = f"/{repo.split('/')[1]}"
            workdir = repo.split('/')[1]

            image_name = row_dict.get("image_name", "")
            tag = image_name.split('/')[-1].replace(':', "_")
            image_name = f"iregistry.baidu-int.com/ainf-matrix/swe-rebench-v2:{tag}"
            row_dict["docker_image"] = image_name
            row_dict["sample_source"] = "swerebench"
            row_dict["sample_type"] = "swerebench"
            row_dict["working_dir"] = f"/{workdir}"

            return {
                "data_source": "swe",
                "prompt": [{"role": "system", "content": R2EGYM_SYSTEM_PROMPT.format(workspace=workdir)}, {"role": "user", "content": R2EGYM_USER_PROMPT.format(problem_statement=problem_statement, workspace=workdir)}],
                "ability": "swe",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": json.dumps(row_dict),
            }

        return process_fn

    process_fn = make_map_fn()

    with open("pass_ids_less100_rm403", "r", encoding="utf-8") as f:
        wanted_ids = {line.strip() for line in f if line.strip()}
    print(f"Loaded {len(wanted_ids)} wanted instance_ids")

    for dataset_name in SWE_DATASETS:
        print(f"Processing dataset: {dataset_name}")
        try:
            # Load the dataset dictionary (which contains splits like 'train' or 'test')
            dataset_splits = load_dataset(dataset_name)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue

        output_name_base = dataset_name.split("/")[-1].replace("-", "_")  # Use underscore for consistency

        # Determine which split exists ('train' or 'test')
        if "train" in dataset_splits:
            split_name = "train"
            split_data = dataset_splits["train"]
        elif "test" in dataset_splits:
            split_name = "test"
            split_data = dataset_splits["test"]
        else:
            print(f"Skipping {dataset_name} as it contains neither 'train' nor 'test' split.")
            continue

        print(f"Using '{split_name}' split for {dataset_name}")

        # Process the data from the identified split
        processed_data = [
            process_fn(row)
            for row in split_data
            if row["language"] in ["python"] and row["instance_id"] in wanted_ids
        ]  # ["python", "go"]

        # Create DataFrame and save to a single parquet file
        df = pd.DataFrame(processed_data)
        output_filepath = os.path.join(local_dir, f"{output_name_base}.parquet")
        df.to_parquet(output_filepath)
        print(f"Saved {len(df)} records from '{split_name}' split to {output_filepath}")

        # Copy to HDFS if needed
        if hdfs_dir is not None:
            hdfs_filepath = os.path.join(hdfs_dir, f"{output_name_base}.parquet")
            # Ensure HDFS directory exists before copying
            # Assuming makedirs handles potential race conditions or existing dirs
            try:
                makedirs(hdfs_dir)
                copy(src=output_filepath, dst=hdfs_filepath)
                print(f"Copied {output_name_base}.parquet to HDFS: {hdfs_filepath}")
            except Exception as e:
                print(f"Failed to copy {output_filepath} to HDFS {hdfs_filepath}: {e}")


if __name__ == "__main__":
    main()
