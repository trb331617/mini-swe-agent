import json
from datasets import load_dataset

dataset_splits = load_dataset('parquet', data_files='/wh-test/tianrenbao/datasets/SWE-bench_Verified/test-00000-of-00001.parquet')

# Determine which split exists ('train' or 'test')
if "train" in dataset_splits:
    split_name = "train"
    split_data = dataset_splits["train"]
elif "test" in dataset_splits:
    split_name = "test"
    split_data = dataset_splits["test"]
else:
    print(f"Skipping {dataset_name} as it contains neither 'train' nor 'test' split.")


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None) or instance.get("docker_image", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        image_name_baidu = f"iregistry.baidu-int.com/ainf-matrix/swe-bench-verified:sweb.eval.x86_64.{id_docker_compatible}".lower()
    return image_name, image_name_baidu

def make_map_fn():
    def process_fn(row):
        row_dict = dict(row)
        """
        problem_statement = row_dict.get("problem_statement", "")
        return {
            "data_source": "swe",
            "prompt": [{"role": "system", "content": SWE_SYSTEM_PROMPT}, {"role": "user", "content": SWE_USER_PROMPT.format(problem_statement=problem_statement)}],
            "ability": "swe",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": json.dumps(row_dict),
        }
        """
        image_name = get_swebench_docker_image_name(row_dict)
        print(f'{row_dict.get("instance_id", "")} {image_name[0]} {image_name[1]}')

    return process_fn

process_fn = make_map_fn()


# Process the data from the identified split
#process_fn(split_data[0])
#exit()
processed_data = [process_fn(row) for row in split_data]
