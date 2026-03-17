#!/usr/bin/env python3


import json
from pathlib import Path


def remove_empty_data(output_path: Path):
    output_data = {}
    if not output_path.exists():
        print(f"{output_path} does not exist")
    output_data = json.loads(output_path.read_text())

    new_data = {
        k: v for k, v in output_data.items()
        if v.get("model_patch", "") != ""
    }

    output_path.write_text(json.dumps(new_data, indent=2))


def remove_fail_data(eval_file: str, preds_file: Path):
    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)

    preds_json = json.loads(preds_file.read_text())

    for k,v in eval_results.items():
        if not v:
            del preds_json[k]

    preds_file.write_text(json.dumps(preds_json, indent=2))


if __name__ == "__main__":
    #remove_empty_data(Path("./preds.json"))
    remove_fail_data("./eval_results.json", Path("./preds.json"))



    
