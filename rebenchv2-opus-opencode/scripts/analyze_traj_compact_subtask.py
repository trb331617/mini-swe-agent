#!/usr/bin/env python3
"""批量分析 opencode 轨迹文件（schema_version 2 的 HTTP 录制格式）。

用法:
    python3 analyze_traj.py <文件或目录> [...] [--json out.json]

判定项:
    - 完整性: HTTP 状态、complete 标志、stop_reason、历史是否纯累加、末轮是否 end_turn
    - 上下文压缩: opencode 注入的压缩标记 + 历史长度是否发生回缩
    - subtask: task 工具是否被真实调用（区别于"只是声明了工具"）
"""
import argparse
import json
import os
import sys
from collections import Counter

# opencode 触发压缩后会把摘要作为新的首条 user 消息注入，这些是它的特征串。
# 只在「模型输入」的 text 块里匹配，不在 tool_result 里匹配，避免被仓库文件内容误伤。
COMPACT_MARKERS = (
    "this session is being continued",
    "summary of the conversation",
    "continued from a previous conversation",
    "<conversation_summary>",
    "compacted",
)
TITLE_MARKER = "You are a title generator"


def iter_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".json"):
                    yield os.path.join(p, name)
        else:
            yield p


def sys_text(turn):
    s = turn["request"]["body"]["value"].get("system") or []
    if isinstance(s, list):
        return " ".join(b.get("text", "") for b in s)
    return str(s)


def input_texts(turn):
    """请求里所有 text 块（不含 tool_result），用于压缩标记检测。"""
    out = []
    for m in turn["request"]["body"]["value"].get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        else:
            for b in c or []:
                if b.get("type") == "text":
                    out.append(b.get("text", ""))
    return out


def analyze(path):
    with open(path) as f:
        d = json.load(f)
    turns = d.get("turns", [])
    r = {
        "file": os.path.basename(path),
        "schema_version": d.get("schema_version"),
        "size_mb": round(os.path.getsize(path) / 1e6, 2),
        "n_turns": len(turns),
        "sessions": sorted({t["request"]["headers"].get("x-session-id") for t in turns}),
        "user_agents": sorted({t["request"]["headers"].get("user-agent") for t in turns}),
    }
    if not turns:
        r["verdict"] = "EMPTY"
        return r

    main_idx = [i for i, t in enumerate(turns) if TITLE_MARKER not in sys_text(t)]
    r["n_title_calls"] = len(turns) - len(main_idx)
    r["n_main_turns"] = len(main_idx)

    tools_declared = turns[main_idx[0]]["request"]["body"]["value"].get("tools") or []
    r["tools_declared"] = [t.get("name") for t in tools_declared]

    tool_calls, stops = Counter(), Counter()
    task_calls, tokens, msg_counts = [], [], []
    bad_status, incomplete = [], []
    for i in main_idx:
        t = turns[i]
        resp = t.get("response") or {}
        if resp.get("status_code") != 200:
            bad_status.append((i, resp.get("status_code")))
        if not resp.get("complete"):
            incomplete.append(i)
        body = (resp.get("body") or {}).get("value") or {}
        stops[body.get("stop_reason")] += 1
        tokens.append((body.get("usage") or {}).get("input_tokens"))
        msg_counts.append(len(t["request"]["body"]["value"].get("messages", [])))
        for c in body.get("content") or []:
            if c.get("type") == "tool_use":
                tool_calls[c["name"]] += 1
                if c["name"] == "task":
                    task_calls.append({"turn": i, "input": c.get("input")})

    r["tool_calls"] = dict(tool_calls.most_common())
    r["stop_reasons"] = dict(stops)
    r["input_tokens"] = tokens
    r["max_input_tokens"] = max([x for x in tokens if x] or [0])
    r["msg_counts"] = msg_counts
    r["bad_status"] = bad_status
    r["incomplete"] = incomplete

    # --- subtask ---
    r["task_tool_available"] = "task" in r["tools_declared"]
    r["task_calls"] = task_calls
    r["used_subtask"] = bool(task_calls)

    # --- 上下文压缩 ---
    hits = []
    for i in main_idx:
        for txt in input_texts(turns[i]):
            low = txt.lower()
            for m in COMPACT_MARKERS:
                if m in low:
                    hits.append({"turn": i, "marker": m, "excerpt": txt[:200]})
    # 历史回缩：msg 数量本应单调 +2 递增，压缩会让它突然变小
    shrinks = [
        (main_idx[k - 1], msg_counts[k - 1], main_idx[k], msg_counts[k])
        for k in range(1, len(msg_counts))
        if msg_counts[k] < msg_counts[k - 1]
    ]
    r["compact_markers"] = hits
    r["history_shrinks"] = shrinks
    r["used_compaction"] = bool(hits or shrinks)

    # --- 完整性 ---
    last = (turns[main_idx[-1]].get("response") or {}).get("body", {}).get("value") or {}
    r["last_stop_reason"] = last.get("stop_reason")
    r["final_text"] = " ".join(
        c.get("text", "") for c in last.get("content") or [] if c.get("type") == "text"
    ).strip()
    monotonic = all(b > a for a, b in zip(msg_counts, msg_counts[1:]))
    problems = []
    if bad_status:
        problems.append(f"非200响应 {bad_status}")
    if incomplete:
        problems.append(f"未完成响应 turn {incomplete}")
    if last.get("stop_reason") != "end_turn":
        problems.append(f"末轮 stop_reason={last.get('stop_reason')}（未自然收尾）")
    if stops.get("max_tokens"):
        problems.append(f"{stops['max_tokens']} 轮被 max_tokens 截断")
    if not monotonic and not shrinks:
        problems.append("历史长度非单调，需人工核对")
    if not r["final_text"]:
        problems.append("末轮无文本输出")
    r["problems"] = problems
    r["complete"] = not problems
    return r


def report(r):
    print("=" * 78)
    print(f"{r['file']}  | schema {r['schema_version']} | {r['size_mb']}MB | {r['n_turns']} turns")
    if r.get("verdict") == "EMPTY":
        print("  空轨迹")
        return
    print(f"  session: {r['sessions']}")
    print(f"  ua     : {r['user_agents']}")
    print(f"  主循环 {r['n_main_turns']} 轮（另有 {r['n_title_calls']} 次标题生成旁路调用）")
    print(f"  工具调用: {r['tool_calls']}")
    print(f"  stop_reasons: {r['stop_reasons']} | 末轮: {r['last_stop_reason']}")
    print(f"  input_tokens 峰值: {r['max_input_tokens']} | msg 数序列: {r['msg_counts']}")
    print(f"  完整性: {'完整' if r['complete'] else '存在问题 -> ' + '; '.join(r['problems'])}")
    print(f"  上下文压缩: {'是' if r['used_compaction'] else '否'}"
          f"（标记命中 {len(r['compact_markers'])}，历史回缩 {len(r['history_shrinks'])}）")
    print(f"  subtask: {'是' if r['used_subtask'] else '否'}"
          f"（task 工具{'已' if r['task_tool_available'] else '未'}声明，调用 {len(r['task_calls'])} 次）")
    for c in r["task_calls"]:
        print(f"    - turn {c['turn']}: {json.dumps(c['input'], ensure_ascii=False)[:200]}")
    if r["final_text"]:
        print(f"  末轮摘要: {r['final_text'][:300].replace(chr(10), ' ')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="轨迹 json 文件或目录")
    ap.add_argument("--json", dest="out", help="把结构化结果写到该文件")
    a = ap.parse_args()

    results = []
    for p in iter_files(a.paths):
        try:
            r = analyze(p)
        except Exception as e:  # 坏文件不该中断整批
            r = {"file": os.path.basename(p), "error": f"{type(e).__name__}: {e}"}
            print("=" * 78)
            print(f"{r['file']}  解析失败: {r['error']}")
            results.append(r)
            continue
        results.append(r)
        report(r)

    ok = [r for r in results if r.get("complete")]
    print("=" * 78)
    print(f"合计 {len(results)} 个文件：完整 {len(ok)}，"
          f"用了压缩 {sum(1 for r in results if r.get('used_compaction'))}，"
          f"用了 subtask {sum(1 for r in results if r.get('used_subtask'))}，"
          f"解析失败 {sum(1 for r in results if 'error' in r)}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结构化结果已写入 {a.out}")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
