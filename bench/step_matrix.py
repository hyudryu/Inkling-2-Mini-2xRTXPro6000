"""step_matrix.py <label> [levels]: decode matrix reporting ms/step and tokens/step (from /metrics deltas) next to throughput. PORT env selects the server."""
import json, sys, time, re, http.client, statistics
from concurrent.futures import ThreadPoolExecutor
S=__import__("os").path.dirname(__import__("os").path.abspath(__file__))
PORT=int(__import__("os").environ.get("PORT","8000")); label=sys.argv[1]; levels=[int(x) for x in (sys.argv[2] if len(sys.argv)>2 else "1,8,32").split(",")]
TOPICS=["a lighthouse keeper","a retired cartographer","a night-shift baker","a beekeeper","a ferry pilot","a violin repairer","a glacier researcher","a train conductor"]
CODE=["an LRU cache class","a trie with insert/search/prefix methods","a min-heap from scratch","Dijkstra's shortest path","a JSON tokenizer","a token-bucket rate limiter","a union-find with path compression","a CSV parser handling quoted fields"]
WL={"prose":lambda i:f"Write a vivid, detailed short story (about 700 words) about {TOPICS[i%8]}.",
    "coding":lambda i:f"Implement {CODE[i%8]} in Python with type hints, docstrings, and unit tests.",
    "counting":lambda i:f"Count from 1 to {400+i} in words, one number per line (e.g. 'one', 'two', ...). Output only the list."}
def metrics():
    c=http.client.HTTPConnection("localhost",PORT,timeout=30); c.request("GET","/metrics"); t=c.getresponse().read().decode(); d={}
    for k in ("num_drafts_total","num_accepted_tokens_total","num_draft_tokens_total"):
        m=re.search(rf"^vllm:spec_decode_{k}\{{[^}}]*\}} ([0-9.e+]+)",t,re.M); d[k]=float(m.group(1)) if m else 0.0
    return d
def stream(prompt):
    body={"model":"inkling-2-mini","messages":[{"role":"user","content":prompt}],"max_tokens":1024,"temperature":0,"stream":True,"stream_options":{"include_usage":True,"continuous_usage_stats":True}}
    c=http.client.HTTPConnection("localhost",PORT,timeout=900); c.request("POST","/v1/chat/completions",json.dumps(body),{"Content-Type":"application/json"}); r=c.getresponse()
    times=[]; ctok=0
    for raw in r:
        line=raw.decode().strip()
        if not line.startswith("data:") or line[5:].strip()=="[DONE]": continue
        u=json.loads(line[5:]).get("usage")
        if u and u.get("completion_tokens",0)!=ctok: ctok=u["completion_tokens"]; times.append((time.perf_counter(),ctok))
    return times
def interp(times,t):
    prev=None
    for x in times:
        if x[0]>t: break
        prev=x
    return prev[1] if prev else 0
rows=[]
for w,fn in WL.items():
    for c in levels:
        m0=metrics(); 
        with ThreadPoolExecutor(c) as ex: res=list(ex.map(stream,[fn(i) for i in range(c)]))
        m1=metrics()
        a=max(r[0][0] for r in res); b=min(r[-1][0] for r in res)
        agg=sum(interp(r,b)-interp(r,a) for r in res)/(b-a)
        dr=m1["num_drafts_total"]-m0["num_drafts_total"]; acc=m1["num_accepted_tokens_total"]-m0["num_accepted_tokens_total"]; dt=m1["num_draft_tokens_total"]-m0["num_draft_tokens_total"]
        tps=1+acc/dr; per=agg/c; step=tps/per*1e3
        rows.append((w,c,agg,per,step,tps,acc/dt*100))
        print(f"{w+'-C'+str(c):13s} agg {agg:7.1f} tok/s | per-user {per:6.1f} | {step:5.1f} ms/step | {tps:.2f} tok/step | accept {acc/dt*100:2.0f}%",flush=True)
for c in levels:
    print(f"mean C{c}: agg {statistics.mean(r[2] for r in rows if r[1]==c):7.1f} | {statistics.mean(r[4] for r in rows if r[1]==c):5.2f} ms/step | {statistics.mean(r[5] for r in rows if r[1]==c):.2f} tok/step")
json.dump(rows,open(f"{S}/results/steps/{label}.json","w"))
