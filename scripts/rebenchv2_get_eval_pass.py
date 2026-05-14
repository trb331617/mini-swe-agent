import json
from pathlib import Path


def get_passed_instances(report_path: str, preds_path: str) -> list[str]:
    """Parse eval report and return instance_ids that passed."""
    old_preds = json.loads(Path(preds_path).read_text(encoding="utf-8"))
    patch_insts = [k for k, v in old_preds.items() if v.get("model_patch", "") != ""]

    passed = []
    if not Path(report_path).exists():
        return passed
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    for item in report["items"]:
        if item.get("passed_match", False) and item["instance_id"] in patch_insts:
            passed.append(item["instance_id"])
    return passed


def main():
    output_dir = Path("/workspace/output")
    eval_file = output_dir / "eval_report.json"
    preds_file = output_dir / "preds.json"
    passed = get_passed_instances(eval_file, preds_file)
    for iid in passed:
        print(iid)


if __name__ == "__main__":
    main()
