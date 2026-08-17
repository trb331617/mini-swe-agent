import os
import sys

import pandas as pd

PARQUET_PATH = "train-00000-of-00001.parquet"
INSTS_FILE = "rebnech_insts"
FILTERED_PARQUET_PATH = "train-filtered.parquet"


def print_image_names():
    """输出所有数据的 instance_id、原始 image_name 及转换后的新 image_name。"""
    df = pd.read_parquet(PARQUET_PATH, columns=["instance_id", "image_name"])
    for instance_id, image_name in zip(df["instance_id"], df["image_name"]):
        assert image_name is not None, f"Task {instance_id} missing image_name."
        tag = image_name.split('/')[-1].replace(':', "_")
        new_image_name = f"iregistry.baidu-int.com/ainf-matrix/swe-rebench-v2-opencode:{tag}"
        print(instance_id, image_name, new_image_name)


def print_patch(target_id):
    """输出指定 instance_id 数据的 patch 字段。"""
    df = pd.read_parquet(PARQUET_PATH, columns=["instance_id", "patch"])
    matched = df[df["instance_id"] == target_id]
    if matched.empty:
        print(f"instance_id not found: {target_id}")
        return
    for patch in matched["patch"]:
        print(patch)


def filter_and_save(insts_file=INSTS_FILE, output_path=FILTERED_PARQUET_PATH):
    """过滤出 insts_file 中列出的 instance_id 数据，保存为新的 parquet 数据集。"""
    with open(insts_file) as f:
        wanted_ids = [line.strip() for line in f if line.strip()]

    df = pd.read_parquet(PARQUET_PATH)
    filtered = df[df["instance_id"].isin(wanted_ids)]

    missing = set(wanted_ids) - set(filtered["instance_id"])
    if missing:
        print(f"Warning: {len(missing)} instance_id 未在数据集中找到: {sorted(missing)}")

    filtered.to_parquet(output_path, index=False)
    print(f"已过滤 {len(filtered)}/{len(wanted_ids)} 条数据, 保存到 {output_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        print_patch(sys.argv[1])
    else:
        filter_and_save()
    # pyarrow(read_parquet 的后端)在解释器退出时的原生清理偶发触发
    # "terminate called without an active exception / Aborted"。
    # 输出已完成，这里刷新缓冲后强制退出，跳过该 teardown 崩溃。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
