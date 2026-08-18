import json
import argparse
from pathlib import Path
from datasets import load_dataset

def rewrite_image(img: str, target_registry: str, target_repo: str) -> str:
    if not isinstance(img, str) or ":" not in img:
        raise ValueError(f"invalid image_name: {img}")

    # 去除前缀
    no_reg = img[len("docker.io/"):] if img.startswith("docker.io/") else img

    # 拆分路径和标签
    path, tag = no_reg.rsplit(":", 1)
    repo = path.split("/")[-1]

    # 拼接新标签：repo名 + 原标签
    new_tag = f"{repo}_{tag}"
    return f"{target_registry}/{target_repo}:{new_tag}"

def main():
    # 1. 定义命令行选项
    parser = argparse.ArgumentParser(description="SWE-rebench 数据集镜像路径重写工具")
    parser.add_argument("-i", "--input", type=str, required=True, help="输入的 Parquet 文件路径")
    parser.add_argument("-o", "--output", type=str, default="./sub_task.json", help="输出的 JSON 文件路径")
    parser.add_argument("--registry", type=str, default="iregistry.baidu-int.com", help="目标 Registry 地址")
    parser.add_argument("--repo", type=str, default="ainf-matrix/swe-rebench-v2", help="目标 Repository 路径")

    args = parser.parse_args()

    # 2. 加载数据集 (Streaming 模式节省内存)
    print(f"正在读取: {args.input}")
    ds = load_dataset("parquet", data_files={"train": args.input}, streaming=True)["train"]

    first = True
    count = 0

    # 3. 执行转换并写入
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("[\n")
        for ex in ds:
            if "image_name" not in ex:
                print(f"警告: 实例 {ex.get('instance_id', 'unknown')} 缺失 image_name，已跳过")
                continue

            # 调用转换函数
            ex["image_name"] = rewrite_image(ex["image_name"], args.registry, args.repo)

            if not first:
                f.write(",\n")

            f.write(json.dumps(ex, ensure_ascii=False))
            first = False
            count += 1
        f.write("\n]\n")

    print(f"成功导出到 {args.output}，总计任务数: {count}")

if __name__ == "__main__":
    main()
