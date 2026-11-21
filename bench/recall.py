#!/usr/bin/env python3
"""Long-context recall check: a document of numbered log entries, then a
question asking for the readings of 10 random entries. Scores exact matches.

    python3 recall.py <label> [--url http://localhost:8000] [--sizes 2000,8000,16000] [--trials 3]

Sizes are numbers of entries (about 16 tokens each). Used to compare KV-cache
precisions: the same prompts, temperature 0."""
import argparse, json, os, random, re, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--model", default="inkling-2-mini")
ap.add_argument("--sizes", default="2000,8000,16000")
ap.add_argument("--trials", type=int, default=3)
a = ap.parse_args()
rows = []
for n in [int(x) for x in a.sizes.split(",")]:
    hit = total = 0
    ptok = 0
    t0 = time.time()
    for trial in range(a.trials):
        rng = random.Random(n * 1000 + trial)
        vals = [rng.randint(1000, 9999) for _ in range(n)]
        doc = "\n".join(f"Entry {i}: sensor {rng.randint(100, 999)} reported {vals[i]} units." for i in range(n))
        ask = sorted(rng.sample(range(n), 10))
        q = (doc + "\n\nFrom the log above, give the reported units for entries "
             + ", ".join(map(str, ask)) + ". Answer with one line per entry in the form 'Entry N: VALUE' and nothing else.")
        body = {"model": a.model, "messages": [{"role": "user", "content": q}], "max_tokens": 3000,
                "temperature": 0, "reasoning_effort": "low"}
        req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=3600))
        text = r["choices"][0]["message"].get("content") or ""
        got = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"Entry\s+(\d+)\s*:\s*(\d+)", text)}
        hit += sum(got.get(i) == vals[i] for i in ask)
        total += len(ask)
        ptok = r["usage"]["prompt_tokens"]
    rows.append({"entries": n, "prompt_tokens": ptok, "correct": hit, "asked": total})
    print(f"{a.label}: {ptok:>7,} prompt tokens: {hit}/{total} correct  ({time.time() - t0:.0f} s)", flush=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "recall")
os.makedirs(out, exist_ok=True)
json.dump(rows, open(os.path.join(out, a.label + ".json"), "w"), indent=1)
