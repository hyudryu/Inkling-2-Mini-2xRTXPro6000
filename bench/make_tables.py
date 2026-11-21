#!/usr/bin/env python3
"""Print the README benchmark tables for several runs, oldest on the left.

    python3 make_tables.py baseline v1 v2

Each name is a directory under results/. The last one is the "current" run:
it is bold, the change column compares it with the first, and the per-user /
TTFT columns are its own."""
import csv, datetime, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
runs = sys.argv[1:] or ["baseline"]
label = lambda r: r.capitalize() if r == "baseline" else r.upper()
load = lambda r, f: json.load(open(os.path.join(HERE, "results", r, f)))
pct = lambda new, old: "—" if len(runs) == 1 else f"{(new / old - 1) * 100:+.0f}%"


def row(cells):
    return "| " + " | ".join(cells) + " |"


def header(cols):
    return row(cols) + "\n" + row(["---"] * len(cols))


def versions(values, fmt):
    out = [fmt(v) for v in values]
    out[-1] = f"**{out[-1]}**"
    return out


# Prefill: median of the warm sweeps.
pre = {}
for r in runs:
    sweeps = [load(r, f"results-prefill-warm{i}.json")["prefill"] for i in range(1, 6)]
    pre[r] = {
        s[0]["target_tokens"]: (
            s[0]["prompt_tokens"],
            statistics.median(x["prefill_tps"] for x in s),
            statistics.median(x["ttft_s"] for x in s),
        )
        for s in zip(*sweeps)
    }
print("### prefill\n")
print(header(["Prompt tokens"] + [f"{label(r)} tok/s" for r in runs] + ["Change", "Time to first token (s)"]))
for size in pre[runs[-1]]:
    vals = [pre[r][size][1] for r in runs]
    print(row([f"{pre[runs[-1]][size][0]:,}"] + versions(vals, lambda v: f"{v:,.0f}")
              + [pct(vals[-1], vals[0]), f"{pre[runs[-1]][size][2]:.2f}"]))

# Decode matrix.
dec = {r: {(x["workload"], x["concurrency"]): x for x in load(r, "results-decode.json")["decode"]} for r in runs}
names = {"prose": "Prose (short story)", "coding": "Coding (Python + tests)", "counting": "Counting (1…400 in words)"}
print("\n### decode\n")
print(header(["Workload", "C"] + [f"{label(r)} tok/s" for r in runs] + ["Change", "Per-user tok/s", "TTFT p50 (s)"]))
for w, name in names.items():
    for c in sorted({k[1] for k in dec[runs[-1]]}):
        vals = [dec[r][(w, c)]["steady_aggregate_tps"] for r in runs]
        cur = dec[runs[-1]][(w, c)]
        print(row([name if c == 1 else "", str(c)] + versions(vals, lambda v: f"{v:.1f}")
                  + [pct(vals[-1], vals[0]), f"{cur['per_req_decode_tps_median']:.1f}", f"{cur['ttft_median_s']:.2f}"]))

# Context sweep.
ctx = {r: {(x["context"], x["concurrency"]): x for x in load(r, "results-context.json")["context"] if "skipped" not in x} for r in runs}
print("\n### context\n")
print(header(["Context", "C"] + [f"{label(r)} tok/s" for r in runs] + ["Change", "Per-user tok/s", "TTFT p50 (s)"]))
for (n, c), cur in sorted(ctx[runs[-1]].items()):
    vals = [ctx[r].get((n, c), {}).get("steady_aggregate_tps") for r in runs]
    cells = [("—" if v is None else f"{v:.1f}") for v in vals]
    cells[-1] = f"**{cells[-1]}**"
    change = pct(vals[-1], vals[0]) if vals[0] and vals[-1] else "—"
    print(row([f"{n // 1024}K", str(c)] + cells + [change, f"{cur['per_req_decode_tps_median']:.1f}", f"{cur['ttft_median_s']:.1f}"]))

# Power for the current run.
r = runs[-1]
phases = {}
for line in open(os.path.join(HERE, "results", r, "phases.txt")):
    name, edge, ts = line.split()
    phases.setdefault(name, {})[edge] = int(ts)
samples = []
for t, gpu, watts, temp, _ in csv.reader(open(os.path.join(HERE, "results", r, "power.csv"))):
    ts = datetime.datetime.strptime(t.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
    samples.append((ts, int(gpu), float(watts), int(temp)))
groups = {
    "Prefill sweeps (5×, warm)": (phases["prefill1"]["start"], phases["prefill5"]["end"]),
    "Decode, C1–C32": (phases["decode"]["start"], phases["decode"]["end"]),
    "Long-context decode": (phases["context"]["start"], phases["context"]["end"]),
}
print(f"\n### power ({label(r)})\n")
print(header(["Phase", "Avg / max power", "Peak temp"]))
for name, (a, b) in groups.items():
    w = [s[2] for s in samples if a <= s[0] <= b]
    t = max(s[3] for s in samples if a <= s[0] <= b)
    print(row([name, f"{statistics.mean(w):.0f} / {max(w):.0f} W", f"{t} °C"]))
