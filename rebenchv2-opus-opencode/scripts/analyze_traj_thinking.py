#!/usr/bin/env python3
"""批量统计 llm-proxy 轨迹文件（schema_version=2）。

用法:
    python stat_traj.py                          # 统计默认目录 rebenchv2_traj
    python stat_traj.py <dir_or_file> [...]      # 指定目录或文件
    python stat_traj.py <dir> --per-turn         # 额外打印每轮明细
    python stat_traj.py <dir> --json out.json    # 结果导出 json
"""
import argparse
import json
import os
from glob import glob

DEFAULT_DIR = '/sgl-workspace/ms-swift/rebenchv2_traj'


def iter_files(paths):
    for p in paths:
        if os.path.isdir(p):
            yield from sorted(glob(os.path.join(p, '*.json')))
        else:
            yield p


def analyze(path):
    with open(path) as f:
        data = json.load(f)
    turns = data.get('turns') or []
    st = {
        'file': os.path.basename(path),
        'trajectory_id': data.get('trajectory_id'),
        'turns': len(turns),
        'ok_turns': 0,
        'err_turns': 0,
        'status_codes': {},
        'thinking_turn_idx': [],
        'thinking_blocks': 0,
        'thinking_chars': 0,
        'tool_calls': 0,
        'parallel_tool_turns': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'cache_read_tokens': 0,
        'cost': 0.0,
        'models': set(),
        'thinking_cfg': set(),
        'effort_cfg': set(),
        'last_stop_reason': None,
        'per_turn': [],
    }
    for t in turns:
        req = (t.get('request') or {}).get('body', {}).get('value') or {}
        resp = t.get('response') or {}
        code = resp.get('status_code')
        st['status_codes'][code] = st['status_codes'].get(code, 0) + 1
        st['models'].add(req.get('model'))
        st['thinking_cfg'].add(json.dumps(req.get('thinking'), sort_keys=True))
        oc = req.get('output_config') or {}
        st['effort_cfg'].add(oc.get('effort'))

        idx = t.get('turn_index')
        if code != 200:
            st['err_turns'] += 1
            st['per_turn'].append({'turn': idx, 'status': code, 'thinking': False})
            continue
        st['ok_turns'] += 1
        val = (resp.get('body') or {}).get('value') or {}
        content = val.get('content') or []
        n_think = n_think_chars = n_tool = 0
        for c in content:
            if c.get('type') == 'thinking':
                n_think += 1
                n_think_chars += len(c.get('thinking') or '')
            elif c.get('type') == 'tool_use':
                n_tool += 1
        if n_think:
            st['thinking_turn_idx'].append(idx)
        st['thinking_blocks'] += n_think
        st['thinking_chars'] += n_think_chars
        st['tool_calls'] += n_tool
        if n_tool > 1:
            st['parallel_tool_turns'] += 1
        u = val.get('usage') or {}
        st['input_tokens'] += u.get('input_tokens') or 0
        st['output_tokens'] += u.get('output_tokens') or 0
        st['cache_read_tokens'] += u.get('cache_read_input_tokens') or 0
        st['cost'] += u.get('cost') or 0.0
        st['last_stop_reason'] = val.get('stop_reason')
        st['per_turn'].append({
            'turn': idx, 'status': code, 'thinking': bool(n_think),
            'thinking_chars': n_think_chars, 'tool_calls': n_tool,
            'stop_reason': val.get('stop_reason'),
            'in': u.get('input_tokens'), 'out': u.get('output_tokens'),
        })

    st['models'] = sorted(x for x in st['models'] if x)
    st['thinking_cfg'] = sorted(st['thinking_cfg'])
    st['effort_cfg'] = sorted(x for x in st['effort_cfg'] if x)
    st['thinking_turns'] = len(st['thinking_turn_idx'])
    st['thinking_ratio'] = st['thinking_turns'] / st['turns'] if st['turns'] else 0.0
    st['thinking_ratio_ok'] = st['thinking_turns'] / st['ok_turns'] if st['ok_turns'] else 0.0
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='*', default=[DEFAULT_DIR])
    ap.add_argument('--per-turn', action='store_true', help='打印每轮明细')
    ap.add_argument('--json', dest='json_out', help='结果写入 json 文件')
    args = ap.parse_args()

    results = []
    for path in iter_files(args.paths or [DEFAULT_DIR]):
        try:
            results.append(analyze(path))
        except Exception as e:
            print(f'[skip] {path}: {type(e).__name__}: {e}')

    for st in results:
        print(f"\n=== {st['file']}")
        print(f"  轮次: 总 {st['turns']}  成功 {st['ok_turns']}  失败 {st['err_turns']}"
              f"  状态码 {st['status_codes']}")
        print(f"  含 thinking 轮次: {st['thinking_turns']}/{st['turns']}"
              f" = {st['thinking_ratio']:.1%}  (仅成功轮: {st['thinking_ratio_ok']:.1%})"
              f"  轮号 {st['thinking_turn_idx']}")
        print(f"  thinking: {st['thinking_blocks']} block / {st['thinking_chars']} 字符")
        print(f"  工具调用: {st['tool_calls']} 次, 并行轮次 {st['parallel_tool_turns']}")
        print(f"  token: in {st['input_tokens']}  out {st['output_tokens']}"
              f"  cache_read {st['cache_read_tokens']}  cost ${st['cost']:.4f}")
        print(f"  model {st['models']}  thinking_cfg {st['thinking_cfg']}"
              f"  effort {st['effort_cfg']}  末轮 stop_reason {st['last_stop_reason']}")
        if args.per_turn:
            for r in st['per_turn']:
                print(f"    turn {r['turn']:>3} status={r['status']}"
                      f" thinking={'Y' if r['thinking'] else '-'}"
                      f" tools={r.get('tool_calls')} in={r.get('in')} out={r.get('out')}"
                      f" stop={r.get('stop_reason')}")

    if results:
        n = len(results)
        tot_turns = sum(r['turns'] for r in results)
        tot_think = sum(r['thinking_turns'] for r in results)
        tot_ok = sum(r['ok_turns'] for r in results)
        print(f"\n=== 汇总 ({n} 个文件)")
        print(f"  总轮次 {tot_turns}  成功 {tot_ok}  失败 {tot_turns - tot_ok}")
        print(f"  含 thinking 轮次 {tot_think}/{tot_turns} = {tot_think / tot_turns:.1%}"
              f"  (仅成功轮 {tot_think / tot_ok:.1%})" if tot_ok else '')
        print(f"  thinking 字符 {sum(r['thinking_chars'] for r in results)}"
              f"  工具调用 {sum(r['tool_calls'] for r in results)}"
              f"  并行轮次 {sum(r['parallel_tool_turns'] for r in results)}")
        print(f"  token in {sum(r['input_tokens'] for r in results)}"
              f"  out {sum(r['output_tokens'] for r in results)}"
              f"  cost ${sum(r['cost'] for r in results):.4f}")
        print(f"  平均每文件: {tot_turns / n:.1f} 轮, "
              f"thinking 轮占比 {sum(r['thinking_ratio'] for r in results) / n:.1%}")

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'\n已写入 {args.json_out}')


if __name__ == '__main__':
    main()
