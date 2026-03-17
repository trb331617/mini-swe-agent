from datasets import load_dataset
import pandas as pd


def download_to_csv():
    print("正在从 Hugging Face 下载 SWE-bench Pro 数据集...")
    dataset = load_dataset("/workspace/mini-swe-agent/SWE-bench_Pro/", split="test")

    df = dataset.to_pandas()

    output_name = "swe_bench_pro_full.csv"
    df.to_csv(output_name, index=False)

    print(f"成功！数据集已保存为: {output_name}")
    print(f"包含实例数量: {len(df)}")
    print(f"列名清单: {df.columns.tolist()}")


if __name__ == "__main__":
    download_to_csv()
