#!/usr/bin/env python3
"""
Iterative SWE-rebench-V2 evaluation script.

For each instance in the dataset, repeatedly runs the agent and evaluates
until the instance passes or reaches the maximum retry count (8).

Usage:
    python run_iterative_eval.py \
        --config /workspace/mini-swe-agent/src/minisweagent/config/benchmarks/swerebench_v2.yaml \
        --subset /workspace/swe-rebench-v2-python-200/ \
        --split train \
        --parquet /workspace/swe-rebench-v2-python-200/train-not-in-mm27.parquet \
        --output output_test/ \
        --max-retries 8
"""

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path

WORKSPACE = Path("/workspace")
SWEBENCH_SCRIPT = "mini-swe-agent/src/minisweagent/run/benchmarks/swebench.py"
PATCH_SCRIPT = WORKSPACE / "SWE-rebench-V2" / "patch.py"
EVAL_SCRIPT = WORKSPACE / "SWE-rebench-V2" / "scripts" / "eval.py"
TASK_SCRIPT = WORKSPACE / "SWE-rebench-V2" / "task.py"


def run_cmd(cmd: list[str], env: dict | None = None, cwd: str | None = None) -> int:
    print(f"\n{'='*60}")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, env=env, cwd=cwd)
    return result.returncode


def generate_sub_tasks(parquet_path: str, output_path: str) -> None:
    """Step 2: Convert parquet dataset to sub_tasks.json."""
    cmd = [
        sys.executable, str(TASK_SCRIPT),
        "-i", parquet_path,
        "-o", output_path,
    ]
    rc = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"task.py failed with exit code {rc}")


def run_agent_for_instances(
    config: str,
    subset: str,
    split: str,
    workers: int,
    output_dir: str,
    instance_ids: list[str] | None = None,
) -> int:
    """Step 1: Run swebench agent for specified instances."""
    import os

    env = os.environ.copy()
    env["AGENT"] = "swe"
    env["DATA_TYPE"] = "rebench"
    env['MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT'] = "3"

    filter_regex = ""
    if instance_ids:
        escaped = [id_.replace(".", r"\.") for id_ in instance_ids]
        filter_regex = "^(" + "|".join(escaped) + ")$"

    cmd = [
        "mini-extra", "swebench",
        "--config", config,
        "--output", output_dir,
        "--subset", subset,
        "--split", split,
        "--workers", workers,
        # "--redo-existing",
    ]
    if filter_regex:
        cmd.extend(["--filter", filter_regex])

    return run_cmd(cmd, env=env)


def convert_preds_to_patches(preds_path: str, patches_path: str) -> int:
    """Step 3: Convert preds.json to patches.json."""
    cmd = [
        sys.executable, str(PATCH_SCRIPT),
        "--preds", preds_path,
        "--out", patches_path,
        # "--drop-empty",
    ]
    return run_cmd(cmd)


def run_eval(
    sub_tasks_path: str,
    patches_path: str,
    report_path: str,
    instance_ids: list[str] | None = None,
) -> int:
    """Step 4: Evaluate patches."""
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--json", str(Path(sub_tasks_path).resolve()),
        "--patches", str(Path(patches_path).resolve()),
        "--max-workers", "20",
        "--report-json", str(Path(report_path).resolve()),
    ]
    if instance_ids:
        cmd.extend(["--instance-ids", ",".join(instance_ids)])
    return run_cmd(cmd, cwd=str(WORKSPACE / "SWE-rebench-V2"))


def get_failed_instances(report_path: str) -> list[str]:
    """Parse eval report and return instance_ids that did not pass."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    failed = []
    for item in report["items"]:
        if item.get("error") or not item.get("passed_match", False):
            failed.append(item["instance_id"])
    return failed


def get_passed_instances(report_path: str, preds_path: str) -> list[str]:
    """Parse eval report and return instance_ids that passed."""
    old_preds = json.loads(Path(preds_path).read_text(encoding="utf-8"))
    empty_insts = [k for k, v in old_preds.items() if v.get("model_patch", "") == ""]

    passed = []
    if not Path(report_path).exists():
        return passed
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    for item in report["items"]:
        if not item.get("error") and item.get("passed_match", False) and item["instance_id"] not in empty_insts:
            passed.append(item["instance_id"])
    return passed

def do_pred_file(preds_file: Path, fail_insts: list[str]):
    output_data = {}
    if not preds_file.exists():
        print(f"{preds_file} does not exist")
        return
    output_data = json.loads(preds_file.read_text())

    new_data = {
        k: v for k, v in output_data.items()
        if k not in fail_insts
    }

    preds_file.write_text(json.dumps(new_data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Iterative SWE-rebench-V2 evaluation")
    parser.add_argument("--config", required=True, help="Agent config YAML path")
    parser.add_argument("--subset", required=True, help="Dataset subset path")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--workers", default=1, help="Number of workers")
    parser.add_argument("--parquet", required=True, help="Parquet file for sub_tasks.json generation")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-retries", type=int, default=8, help="Max retries per instance")
    parser.add_argument("--sub-tasks", default="", help="Pre-existing sub_tasks.json (skip generation)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds_path = str(output_dir / "preds.json")
    patches_path = str(output_dir / "patches.json")
    report_path = str(output_dir / "eval_report.json")

    # redo model_patch is empty
    shutil.copy(preds_path, str(output_dir / "preds_16.json"))
    old_preds = json.loads(Path(preds_path).read_text(encoding="utf-8"))
    noempty_preds = {
        k: v for k, v in old_preds.items()
        if v.get("model_patch", "") != ""
    }
    empty_insts = [k for k, v in old_preds.items() if v.get("model_patch", "") == ""]
    Path(preds_path).write_text(json.dumps(noempty_preds, indent=2))

    # Generate sub_tasks.json
    if args.sub_tasks:
        sub_tasks_path = args.sub_tasks
    else:
        sub_tasks_path = str(output_dir / "sub_tasks.json")
        print("\n[INFO] Generating sub_tasks.json from parquet...")
        # generate_sub_tasks(args.parquet, sub_tasks_path)

    # Track per-instance attempt counts and results
    attempt_counts: dict[str, int] = {}
    resolved_instances: set[str] = set()

    # Round 1: run all instances
    print("\n" + "#"*60)
    print(f"# ROUND 1: Running agent on all instances")
    print("#"*60)

    rc = run_agent_for_instances(
        config=args.config,
        subset=args.subset,
        split=args.split,
        workers=args.workers,
        output_dir=str(output_dir),
    )
    shutil.copy(preds_path, str(output_dir / "preds_empty_1.json"))
    shutil.copy(str(output_dir / "minisweagent.log"), str(output_dir / "minisweagent_empty_1.log"))

    # Read all instance_ids from preds.json
    all_instance_ids = empty_insts
    """
    if Path(preds_path).exists():
        preds = json.loads(Path(preds_path).read_text(encoding="utf-8"))
        all_instance_ids = list(preds.keys())
    else:
        print("[ERROR] preds.json not generated, aborting.")
        return 1
    """
    for iid in all_instance_ids:
        attempt_counts[iid] = 1

    # Convert and evaluate
    convert_preds_to_patches(preds_path, patches_path)
    run_eval(sub_tasks_path, patches_path, report_path, instance_ids=all_instance_ids)

    passed = get_passed_instances(report_path, preds_path)
    # failed = get_failed_instances(report_path)
    # failed = [x for x in all_instance_ids if x not in passed]
    resolved_instances.update(passed)
    failed = list(set(all_instance_ids) - resolved_instances)

    print(f"\n[ROUND 1 RESULT] Passed: {len(passed)}, Failed: {len(failed)}, Total: {len(all_instance_ids)}")

    # Iterative retries for failed instances
    for round_num in range(2, args.max_retries + 1):
        if not failed:
            print("\n[INFO] All instances passed!")
            break

        # Filter out instances that have already exhausted max retries
        retryable = [iid for iid in failed if attempt_counts.get(iid, 0) < args.max_retries]
        if not retryable:
            print(f"\n[INFO] All remaining failed instances have reached max retries ({args.max_retries}).")
            break
        do_pred_file(Path(preds_path), retryable)

        print(f"\n{'#'*60}")
        print(f"# ROUND {round_num}: Retrying {len(retryable)} failed instances")
        print(f"#{'#'*59}")
        for iid in retryable:
            print(f"  - {iid} (attempt {attempt_counts[iid] + 1})")

        # Run agent only for failed instances
        rc = run_agent_for_instances(
            config=args.config,
            subset=args.subset,
            split=args.split,
            output_dir=str(output_dir),
            # instance_ids=retryable,
            workers=args.workers,
        )
        shutil.copy(preds_path, str(output_dir / f"preds_empty_{round_num}.json"))
        shutil.copy(str(output_dir / "minisweagent.log"), str(output_dir / f"minisweagent_empty_{round_num}.log"))

        for iid in retryable:
            attempt_counts[iid] += 1

        # Re-convert and re-evaluate only the retried instances
        convert_preds_to_patches(preds_path, patches_path)

        round_report_path = str(output_dir / f"eval_report_round_empty{round_num}.json")
        run_eval(sub_tasks_path, patches_path, round_report_path, instance_ids=retryable)

        newly_passed = get_passed_instances(round_report_path, preds_path)
        # still_failed = get_failed_instances(round_report_path)
        resolved_instances.update(newly_passed)
        still_failed = list(set(all_instance_ids) - resolved_instances)

        print(f"\n[ROUND {round_num} RESULT] Newly passed: {len(newly_passed)}, Still failed: {len(still_failed)}")

        failed = still_failed

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total instances:  {len(all_instance_ids)}")
    print(f"Resolved:         {len(resolved_instances)}")
    print(f"Unresolved:       {len(all_instance_ids) - len(resolved_instances)}")
    print(f"Resolution rate:  {len(resolved_instances)/len(all_instance_ids)*100:.1f}%")
    print()
    print("Per-instance attempts:")
    for iid in sorted(all_instance_ids):
        status = "PASSED" if iid in resolved_instances else "FAILED"
        print(f"  {iid}: {attempt_counts[iid]} attempts - {status}")

    # Save final summary
    summary = {
        "total": len(all_instance_ids),
        "resolved": len(resolved_instances),
        "unresolved": len(all_instance_ids) - len(resolved_instances),
        "resolution_rate": len(resolved_instances) / len(all_instance_ids) if all_instance_ids else 0,
        "max_retries": args.max_retries,
        "instances": {
            iid: {
                "attempts": attempt_counts[iid],
                "resolved": iid in resolved_instances,
            }
            for iid in sorted(all_instance_ids)
        },
    }
    summary_path = output_dir / "iterative_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")

    return 0 if len(resolved_instances) == len(all_instance_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
