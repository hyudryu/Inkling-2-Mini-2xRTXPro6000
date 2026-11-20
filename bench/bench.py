#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible vLLM endpoint: prefill sweep + decode matrix.

Stdlib only. Usage:
  python3 bench.py --url http://localhost:8000 --model inkling-2-mini --out results.json
  python3 bench.py --only prefill        # or: decode, context

Measurements
- prefill: one request at a time, prompt of N tokens (unique random words so the
  prefix cache never hits), max_tokens=1. Prefill tok/s = prompt_tokens / TTFT.
- decode: C concurrent requests (C in 1,2,4,8), short prompts for three workloads
  (prose, coding, counting), temperature 0, up to --max-tokens output tokens.
  Reports per-request decode tok/s (tokens after the first / time after the first
  token) and aggregate output tok/s over the steady window in which every request
  of the batch is decoding (prefills excluded). Output tokens include reasoning.
- context: same as decode but each request carries a long unique document
  (16K/64K/128K tokens) before the task; cells whose KV need exceeds the pool are
  skipped.
"""
import argparse, http.client, json, random, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

WORDS = ("amber falcon river copper lantern meadow quartz harbor willow ember orchid canyon "
         "glacier saffron tundra violet granite comet maple prairie sable cobalt juniper "
         "lagoon marble nectar obsidian pepper quill raven sierra thistle umber velvet "
         "walnut yarrow zephyr basalt cedar delta fjord grove heron indigo jasper kelp "
         "lotus mesa nimbus oak pebble quarry reef spruce tide").split()

TOPICS = ["a lighthouse keeper", "a retired cartographer", "a night-shift baker", "a beekeeper",
          "a ferry pilot", "a violin repairer", "a glacier researcher", "a train conductor"]
CODE = ["an LRU cache class", "a trie with insert/search/prefix methods", "a min-heap from scratch",
        "Dijkstra's shortest path", "a JSON tokenizer", "a token-bucket rate limiter",
        "a union-find with path compression", "a CSV parser handling quoted fields"]
WORKLOADS = {
    "prose": lambda i: f"Write a vivid, detailed short story (about 700 words) about {TOPICS[i % 8]}.",
    "coding": lambda i: f"Implement {CODE[i % 8]} in Python with type hints, docstrings, and unit tests.",
    "counting": lambda i: f"Count from 1 to {400 + i} in words, one number per line (e.g. 'one', 'two', ...). Output only the list.",
}


class Client:
    def __init__(self, url, model):
        u = urlparse(url)
        self.host, self.port, self.model = u.hostname, u.port or 80, model
        self.continuous = True

    def stream(self, prompt, max_tokens, temperature=0.0):
        """Return dict with ttft, e2e, prompt_tokens, completion_tokens, token_times."""
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True,
                                   "continuous_usage_stats": self.continuous}}
        conn = http.client.HTTPConnection(self.host, self.port, timeout=3600)
        t0 = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", json.dumps(body),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:400]!r}")
        ttft, ptok, ctok, times, cached = None, 0, 0, [], None  # times: (t, cumulative completion tokens)
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            now = time.perf_counter() - t0
            got_text = any((c.get("delta") or {}).get(k) for c in ev.get("choices") or []
                           for k in ("content", "reasoning_content", "reasoning"))
            if got_text and ttft is None:
                ttft = now
            if ev.get("usage"):
                ptok = ev["usage"].get("prompt_tokens", ptok)
                details = ev["usage"].get("prompt_tokens_details") or {}
                cached = details.get("cached_tokens", cached)
                new = ev["usage"].get("completion_tokens", ctok)
                if new != ctok:
                    ctok = new
                    times.append((now, ctok))
                    if ttft is None:  # first token produced no text (e.g. a hidden thinking marker)
                        ttft = now
        e2e = time.perf_counter() - t0
        conn.close()
        return {"ttft": ttft, "e2e": e2e, "t0": t0, "prompt_tokens": ptok, "cached_tokens": cached,
                "completion_tokens": ctok, "token_times": times}


def unique_doc(n_words, seed):
    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    # sprinkle numbers so every request is distinct from its first token on
    return f"[doc {seed}-{rng.randrange(10**9)}] " + " ".join(
        f"{w} {rng.randrange(1000)}" if i % 7 == 0 else w for i, w in enumerate(words))


def calibrate(cl):
    """Tokens per filler word, measured on the server's tokenizer."""
    r = cl.stream(unique_doc(4000, 1) + "\nReply OK.", 1)
    return r["prompt_tokens"] / 4000.0


def doc_for_tokens(target, tpw, seed):
    return unique_doc(max(1, int(target / tpw)), seed)


def run_prefill(cl, tpw, sizes, seed_offset=0):
    rows = []
    for n in sizes:
        prompt = doc_for_tokens(n - 32, tpw, seed=10_000 + n + seed_offset * 10_000_000) + "\nReply with the single word OK."
        r = cl.stream(prompt, 1)
        row = {"target_tokens": n, "prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"], "ttft_s": r["ttft"],
               "prefill_tps": r["prompt_tokens"] / r["ttft"] if r["ttft"] else None}
        rows.append(row)
        print(json.dumps({"prefill": row}), flush=True)
    return rows


def summarize_batch(results, wall):
    first = max(r["ttft"] + (r["t0"]) for r in results)            # all requests decoding
    last = min(r["e2e"] + r["t0"] for r in results)                  # first finisher
    toks_in_window, per_req = 0, []
    for r in results:
        tt = [(t + r["t0"], c) for t, c in r["token_times"]]
        if r["ttft"] is not None and r["completion_tokens"] > 1 and r["e2e"] > r["ttft"]:
            ft = next((c for t, c in r["token_times"] if t >= r["ttft"]), 1)
            per_req.append((r["completion_tokens"] - ft) / (r["e2e"] - r["ttft"]))
        if last > first and tt:
            before = max((c for t, c in tt if t <= first), default=0)
            upto = max((c for t, c in tt if t <= last), default=0)
            toks_in_window += upto - before
    steady = toks_in_window / (last - first) if last > first and toks_in_window else None
    total = sum(r["completion_tokens"] for r in results)
    return {"per_req_decode_tps_median": statistics.median(per_req) if per_req else None,
            "steady_aggregate_tps": steady,
            "end_to_end_aggregate_tps": total / wall,
            "ttft_median_s": statistics.median(r["ttft"] for r in results if r["ttft"]),
            "avg_output_tokens": total / len(results)}


def run_batch(cl, prompts, max_tokens):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(len(prompts)) as ex:
        res = list(ex.map(lambda p: cl.stream(p, max_tokens), prompts))
    return res, time.perf_counter() - t0


def run_decode(cl, levels, max_tokens, rounds):
    rows = []
    for name, fn in WORKLOADS.items():
        for c in levels:
            agg = []
            for rd in range(rounds):
                res, wall = run_batch(cl, [fn(rd * c + i) for i in range(c)], max_tokens)
                agg.append(summarize_batch(res, wall))
            row = {"workload": name, "concurrency": c, **{k: statistics.median(
                [a[k] for a in agg if a[k] is not None]) if any(a[k] is not None for a in agg) else None
                for k in agg[0]}}
            rows.append(row)
            print(json.dumps({"decode": row}), flush=True)
    return rows


def run_context(cl, tpw, levels, contexts, max_tokens, pool_tokens, seed_offset=0):
    rows = []
    for ctx in contexts:
        for c in levels:
            need = c * (ctx + max_tokens + 256)
            if pool_tokens and need > pool_tokens:
                row = {"context": ctx, "concurrency": c, "skipped": f"needs ~{need} KV tokens > pool {pool_tokens}"}
            else:
                prompts = [doc_for_tokens(ctx, tpw, seed=ctx * 100 + c * 10 + i + seed_offset * 10_000_000) +
                           f"\n\nIgnore the document above. Write a detailed short story (about 600 words) about {TOPICS[i % 8]}."
                           for i in range(c)]
                res, wall = run_batch(cl, prompts, max_tokens)
                row = {"context": ctx, "concurrency": c,
                       "prompt_tokens": statistics.median(r["prompt_tokens"] for r in res),
                       **summarize_batch(res, wall)}
            rows.append(row)
            print(json.dumps({"context": row}), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--only", choices=["prefill", "decode", "context"], action="append")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--prefill-sizes", default="1024,4096,16384,32768,65536,131072,250000")
    ap.add_argument("--contexts", default="16384,65536,131072")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--context-max-tokens", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="vary prompts between runs so the prefix cache never serves a repeat")
    ap.add_argument("--pool-tokens", type=int, default=0, help="KV pool size from the server log")
    a = ap.parse_args()
    cl = Client(a.url, a.model)
    try:
        cl.stream("Say hi.", 4)
    except RuntimeError:
        cl.continuous = False  # frontend without continuous_usage_stats
        cl.stream("Say hi.", 4)
    parts = a.only or ["prefill", "decode", "context"]
    levels = [int(x) for x in a.levels.split(",")]
    tpw = calibrate(cl)
    out = {"meta": {"url": a.url, "model": a.model, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "tokens_per_filler_word": tpw, "continuous_usage_stats": cl.continuous,
                    "max_tokens": a.max_tokens, "context_max_tokens": a.context_max_tokens,
                    "rounds": a.rounds, "pool_tokens": a.pool_tokens, "seed_offset": a.seed_offset}}
    if "prefill" in parts:
        out["prefill"] = run_prefill(cl, tpw, [int(x) for x in a.prefill_sizes.split(",")], a.seed_offset)
    if "decode" in parts:
        out["decode"] = run_decode(cl, levels, a.max_tokens, a.rounds)
    if "context" in parts:
        out["context"] = run_context(cl, tpw, levels, [int(x) for x in a.contexts.split(",")],
                                     a.context_max_tokens, a.pool_tokens, a.seed_offset)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
