import json
import os
import sys

import pandas as pd

PARQUET_PATH = "train-00000-of-00001.parquet"
INSTS_FILE = "rebnech_insts_500"
FILTERED_PARQUET_PATH = "train-filtered.parquet"
LANG_FILE = "rebench_languae_500"
REPORT_FILE = "eval_report_500_1.json"


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


def print_languages(insts_file=INSTS_FILE):
    """读取 insts_file 中的 instance_id 列表，输出这部分数据的 language 字段。"""
    with open(insts_file) as f:
        wanted_ids = [line.strip() for line in f if line.strip()]

    df = pd.read_parquet(PARQUET_PATH, columns=["instance_id", "language"])
    matched = df[df["instance_id"].isin(wanted_ids)]

    missing = set(wanted_ids) - set(matched["instance_id"])
    if missing:
        print(f"Warning: {len(missing)} instance_id 未在数据集中找到: {sorted(missing)}")

    for instance_id, language in zip(matched["instance_id"], matched["language"]):
        print(instance_id, language)


def print_passed_match(lang_file=LANG_FILE, report_file=REPORT_FILE):
    """读取 lang_file(第一列 instance_id, 第二列 language) 与评测报告, 输出对应的 passed_match。"""
    with open(lang_file) as f:
        lang_map = {}
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                lang_map[parts[0]] = parts[1]

    with open(report_file) as f:
        report = json.load(f)
    passed_map = {item["instance_id"]: item["passed_match"] for item in report["items"]}

    for instance_id, language in lang_map.items():
        if instance_id not in passed_map:
            print(f"Warning: {instance_id} 未在 {report_file} 中找到")
            continue
        print(instance_id, language, passed_map[instance_id])

    print_pass_rate_by_language(lang_map, passed_map)


def print_pass_rate_by_language(lang_map, passed_map):
    """按 language 汇总通过率。"""
    stats = {}
    for instance_id, language in lang_map.items():
        if instance_id not in passed_map:
            continue
        total, passed = stats.get(language, (0, 0))
        stats[language] = (total + 1, passed + bool(passed_map[instance_id]))

    print("\n=== 按 language 汇总通过率 ===")
    for language in sorted(stats, key=lambda lang: -stats[lang][0]):
        total, passed = stats[language]
        print(f"{language:10s} {passed:4d}/{total:<4d} {passed / total:.2%}")

    all_total = sum(total for total, _ in stats.values())
    all_passed = sum(passed for _, passed in stats.values())
    print(f"{'TOTAL':10s} {all_passed:4d}/{all_total:<4d} {all_passed / all_total:.2%}")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        print_patch(sys.argv[1])
    else:
        print_passed_match()
    # pyarrow(read_parquet 的后端)在解释器退出时的原生清理偶发触发
    # "terminate called without an active exception / Aborted"。
    # 输出已完成，这里刷新缓冲后强制退出，跳过该 teardown 崩溃。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
