"""Print mean +/- SD tables and paired deltas vs baseline. Usage: python summarize.py [results_dir]"""
import glob, json, os, statistics as st, sys
from collections import defaultdict

base = sys.argv[1] if len(sys.argv) > 1 else "results"
runs = defaultdict(dict)   # (group, n, variant) -> {seed: value}
for d in sorted(glob.glob(os.path.join(base, "*", "*"))):
    try:
        cfg = json.load(open(os.path.join(d, "run_config.json")))
        sm = json.load(open(os.path.join(d, "summary.json")))
    except FileNotFoundError:
        continue   # unfinished run
    group = os.path.basename(os.path.dirname(d))
    if "best_val_acc" in sm:
        runs[(group, cfg["per_class"], cfg["variant"])][cfg["seed"]] = 100 * sm["best_val_acc"]
    elif "best_val_ppl" in sm:
        nu = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
        best = min(nu, key=lambda r: r["val_ppl"])
        runs[(group, "-", cfg["variant"])][cfg["seed"]] = (sm["best_val_ppl"], best["val_nonunk_ppl"])

def fmt(xs):
    return f"{st.mean(xs):7.2f} ± {st.stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:7.2f}"

for grp_n in sorted({k[:2] for k in runs}, key=str):
    group, n = grp_n
    is_gen = n == "-"
    print(f"\n== {group}" + ("" if is_gen else f"  n={n}/class") +
          ("  (val perplexity, lower is better)" if is_gen else "  (val accuracy %)"))
    ref_name = "attn" if is_gen else "baseline"
    ref = runs.get((group, n, ref_name), {})
    for (g, nn, v), seeds in sorted(runs.items(), key=str):
        if (g, nn) != grp_n:
            continue
        vals = [s[0] if is_gen else s for s in seeds.values()]
        line = f"  {v:12s} seeds={sorted(seeds)}  {fmt(vals)}"
        if is_gen:
            line += f"   non-unk ppl {fmt([s[1] for s in seeds.values()])}"
        common = sorted(set(seeds) & set(ref))
        if v != ref_name and common:
            get = (lambda x: x[0]) if is_gen else (lambda x: x)
            diffs = [get(seeds[s]) - get(ref[s]) for s in common]
            line += f"   Δ vs {ref_name} per seed: " + ", ".join(f"{x:+.2f}" for x in diffs)
        print(line)
