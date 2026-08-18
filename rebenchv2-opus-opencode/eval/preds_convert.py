#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_preds_entries(preds: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Returns: list of (instance_id, entry_dict)

    Supports:
      - dict keyed by instance_id (mini-SWE-agent v2)
      - list of dicts with instance_id
    """
    if isinstance(preds, dict):
        out: List[Tuple[str, Dict[str, Any]]] = []
        for k, v in preds.items():
            if isinstance(v, dict):
                iid = v.get("instance_id") or k
                out.append((iid, v))
        return out

    if isinstance(preds, list):
        out: List[Tuple[str, Dict[str, Any]]] = []
        for item in preds:
            if isinstance(item, dict):
                iid = item.get("instance_id")
                if iid:
                    out.append((iid, item))
        return out

    raise TypeError(f"Unsupported preds.json root type: {type(preds)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to mini-swe-agent preds.json")
    ap.add_argument("--out", required=True, help="Output path for SWE-rebench-V2 patches.json")
    ap.add_argument("--drop-empty", action="store_true", help="Drop empty/whitespace-only patches")
    ap.add_argument(
        "--keep-meta",
        action="store_true",
        help="Keep extra fields like model_name_or_path in each entry.",
    )
    args = ap.parse_args()

    preds_path = Path(args.preds)
    out_path = Path(args.out)

    preds = _load_json(preds_path)
    entries = _extract_preds_entries(preds)

    patches_by_iid: Dict[str, Dict[str, Any]] = {}
    empty = 0
    duplicate = 0

    for iid, entry in entries:
        raw_patch = entry.get("model_patch") or entry.get("patch") or entry.get("diff") or ""

        if not isinstance(raw_patch, str):
            print(f"[warn] instance_id={iid}: patch is not a string, got {type(raw_patch)}", file=sys.stderr)
            patch_str = ""
        else:
            patch_str = raw_patch

        if args.drop_empty and not patch_str.strip():
            empty += 1
            continue

        obj: Dict[str, Any] = {
            "instance_id": iid,
            "patch": patch_str,
        }

        if args.keep_meta:
            for k in ("model_name_or_path", "model_patch"):
                if k in entry:
                    obj[k] = entry[k]

        if iid in patches_by_iid:
            duplicate += 1
            print(f"[warn] duplicate instance_id={iid}, keeping the last one", file=sys.stderr)

        patches_by_iid[iid] = obj

    patches_out = [patches_by_iid[iid] for iid in sorted(patches_by_iid)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(patches_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {len(patches_out)} patch entries to: {out_path}")
    if args.drop_empty:
        print(f"Dropped empty patches: {empty}")
    if duplicate:
        print(f"Overwrote duplicate instance_ids: {duplicate}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
