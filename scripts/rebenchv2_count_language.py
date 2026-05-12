
from collections import Counter

from datasets import load_dataset

ds = load_dataset("/workspace/nebius_SWE-rebench-V2/", split="train")

counts = Counter(ds["language"])
total = sum(counts.values())

for lang, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"{lang}: {n} ({n / total * 100:.2f}%)")
print(f"Total: {total}")


"""
dataset: swe-rebench-v2

Total: 32079

python: 7243 (22.58%)
go: 6144 (19.15%)
ts: 4204 (13.11%)
js: 4138 (12.90%)
rust: 3123 (9.74%)
java: 1716 (5.35%)
php: 1445 (4.50%)
kotlin: 889 (2.77%)
julia: 793 (2.47%)
elixir: 416 (1.30%)
scala: 411 (1.28%)
swift: 362 (1.13%)
dart: 251 (0.78%)
c: 230 (0.72%)
cpp: 182 (0.57%)
csharp: 173 (0.54%)
r: 157 (0.49%)
clojure: 105 (0.33%)
ocaml: 58 (0.18%)
lua: 39 (0.12%)
"""
