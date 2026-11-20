#!/usr/bin/env python3
"""Render README charts (light + dark SVG) from the benchmark JSON. Stdlib only.

  python3 make_charts.py            # writes ../docs/charts/*.svg
"""
import json, math, os
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "charts")
RESULTS = os.path.join(HERE, "results", os.environ.get("RESULTS", "baseline"))      # charted run
BASELINE = os.path.join(HERE, "results", "baseline")                          # dashed comparison
_run = os.path.basename(RESULTS)
MODEL_LABEL = "Inkling-2-Mini + DSpark, " + (_run.capitalize() if _run == "baseline" else _run.upper())
LABEL_NUDGE = {}  # per-series (dx, dy) for the context chart end labels; tune after a render

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]},
}
MARKERS = ["circle", "square", "triangle", "diamond", "tri_down"]  # secondary encoding: identity is never color-alone
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

W, H = 760, 430
ML, MR, MT, MB = 64, 150, 92, 56   # right margin holds direct labels


def nice_max(v, max_ticks=6):
    """Smallest round axis max >= v using a 1/2/2.5/5 x 10^k tick step."""
    base = 10 ** math.floor(math.log10(v / max_ticks))
    for m in (1, 2, 2.5, 5, 10):
        step = m * base
        if v / step <= max_ticks:
            return math.ceil(v / step) * step, step
    return math.ceil(v / (10 * base)) * 10 * base, 10 * base


def marker(kind, x, y, color, surface):
    ring = f'stroke="{surface}" stroke-width="2"'
    if kind == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" {ring}/>'
    if kind == "square":
        return f'<rect x="{x-4.6:.1f}" y="{y-4.6:.1f}" width="9.2" height="9.2" rx="1.5" fill="{color}" {ring}/>'
    if kind == "diamond":
        return (f'<path d="M{x:.1f},{y-6.2:.1f} L{x+6.2:.1f},{y:.1f} L{x:.1f},{y+6.2:.1f} L{x-6.2:.1f},{y:.1f} Z" '
                f'fill="{color}" {ring} stroke-linejoin="round"/>')
    if kind == "tri_down":
        return (f'<path d="M{x:.1f},{y+6:.1f} L{x+5.6:.1f},{y-4:.1f} L{x-5.6:.1f},{y-4:.1f} Z" '
                f'fill="{color}" {ring} stroke-linejoin="round"/>')
    return (f'<path d="M{x:.1f},{y-6:.1f} L{x+5.6:.1f},{y+4:.1f} L{x-5.6:.1f},{y+4:.1f} Z" '
            f'fill="{color}" {ring} stroke-linejoin="round"/>')


def line_chart(title, subtitle, xlabels, series, ylabel, theme, label_fmt="{:.0f}", label_nudge=None,
               ghosts=None, legend_note=None):
    """ghosts: optional list of (series index, [y...]) drawn dashed, unlabelled (a comparison run)."""
    t = THEMES[theme]
    pw, ph = W - ML - MR, H - MT - MB
    allv = [v for _, ys in series for v in ys if v is not None] + [v for _, ys in (ghosts or []) for v in ys if v is not None]
    ymax, ystep = nice_max(max(allv) * 1.08)
    xs = [ML + pw * (i + 0.5) / len(xlabels) for i in range(len(xlabels))]
    ypos = lambda v: MT + ph * (1 - v / ymax)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{FONT}" role="img" aria-label="{escape(title)}">',
         f'<rect width="{W}" height="{H}" rx="8" fill="{t["surface"]}"/>',
         f'<text x="{ML-40}" y="30" font-size="17" font-weight="600" fill="{t["text"]}">{escape(title)}</text>',
         f'<text x="{ML-40}" y="51" font-size="12.5" fill="{t["text2"]}">{escape(subtitle)}</text>']
    # legend row (always present for >= 2 series)
    lx = ML - 40
    for i, (name, _) in enumerate(series):
        o.append(marker(MARKERS[i], lx + 6, 72, t["series"][i], t["surface"]))
        o.append(f'<text x="{lx+17}" y="76" font-size="12.5" fill="{t["text2"]}">{escape(name)}</text>')
        lx += 30 + 7.2 * len(name)
    if legend_note:
        o.append(f'<line x1="{lx+6}" x2="{lx+30}" y1="72" y2="72" stroke="{t["muted"]}" stroke-width="2" stroke-dasharray="5 4"/>')
        o.append(f'<text x="{lx+37}" y="76" font-size="12.5" fill="{t["text2"]}">{escape(legend_note)}</text>')
    # grid + y ticks
    v = 0.0
    while v <= ymax + 1e-9:
        y = ypos(v)
        o.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{y:.1f}" y2="{y:.1f}" '
                 f'stroke="{t["axis"] if v == 0 else t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{ML-10}" y="{y+4:.1f}" font-size="11.5" text-anchor="end" fill="{t["muted"]}">{v:.0f}</text>')
        v += ystep
    o.append(f'<text x="18" y="{MT+ph/2:.1f}" font-size="11.5" fill="{t["muted"]}" text-anchor="middle" '
             f'transform="rotate(-90 18 {MT+ph/2:.1f})">{escape(ylabel)}</text>')
    for x, lab in zip(xs, xlabels):
        o.append(f'<text x="{x:.1f}" y="{MT+ph+22}" font-size="12" text-anchor="middle" fill="{t["text2"]}">{escape(lab)}</text>')
    o.append(f'<text x="{ML+pw/2:.1f}" y="{H-12}" font-size="11.5" text-anchor="middle" fill="{t["muted"]}">Concurrent requests</text>')
    for i, ys in (ghosts or []):
        pts = [(xs[j], ypos(v)) for j, v in enumerate(ys) if v is not None]
        o.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" '
                 f'stroke="{t["series"][i]}" stroke-width="2" stroke-dasharray="5 4" stroke-linecap="round" opacity="0.75"/>')
    # lines, markers, then direct end labels spread apart so they never overprint
    ends = []
    for i, (name, ys) in enumerate(series):
        c = t["series"][i]
        pts = [(xs[j], ypos(v)) for j, v in enumerate(ys) if v is not None]
        o.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" '
                 f'stroke="{c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            o.append(marker(MARKERS[i], x, y, c, t["surface"]))
        last = max(j for j, v in enumerate(ys) if v is not None)
        ends.append([xs[last], ypos(ys[last]), ypos(ys[last]), name, ys[last], i])
    gap = 16
    for x in sorted({e[0] for e in ends}):
        group = sorted((e for e in ends if e[0] == x), key=lambda e: e[1])
        for k in range(1, len(group)):                       # push down to keep a minimum gap
            group[k][2] = max(group[k][2], group[k - 1][2] + gap)
        overflow = group[-1][2] - (MT + ph)                  # keep inside the plot, then push up
        if overflow > 0:
            for e in group:
                e[2] -= overflow
            for k in range(len(group) - 2, -1, -1):
                group[k][2] = min(group[k][2], group[k + 1][2] - gap)
    for x, y, ly, name, v, i in ends:
        if label_nudge and name in label_nudge:
            dx, dy = label_nudge[name]
            x, ly = x + dx, ly + dy
        if abs(ly - y) > 3:                                  # leader line from point to moved label
            o.append(f'<path d="M{x+7:.1f},{y:.1f} L{x+16:.1f},{ly:.1f} L{x+20:.1f},{ly:.1f}" fill="none" '
                     f'stroke="{t["muted"]}" stroke-width="1"/>')
        o.append(f'<text x="{x+24 if abs(ly - y) > 3 else x+12:.1f}" y="{ly+4:.1f}" font-size="12.5" fill="{t["text"]}">'
                 f'<tspan font-weight="600">{label_fmt.format(v)}</tspan>'
                 f'<tspan fill="{t["text2"]}"> {escape(name)}</tspan></text>')
    o.append("</svg>")
    return "\n".join(o)


def main():
    os.makedirs(OUT, exist_ok=True)
    dec = json.load(open(os.path.join(RESULTS, "results-decode.json")))["decode"]
    ctx_doc = json.load(open(os.path.join(RESULTS, "results-context.json")))
    ctx, pool = ctx_doc["context"], ctx_doc["meta"]["pool_tokens"]
    levels = sorted({r["concurrency"] for r in dec})
    xl = [f"C{c}" for c in levels]
    get = lambda w, c, k: next(r[k] for r in dec if r["workload"] == w and r["concurrency"] == c)
    base = None
    if os.path.abspath(RESULTS) != os.path.abspath(BASELINE) and os.path.exists(os.path.join(BASELINE, "results-decode.json")):
        base = json.load(open(os.path.join(BASELINE, "results-decode.json")))["decode"]
    bget = lambda w, c, k: next(r[k] for r in base if r["workload"] == w and r["concurrency"] == c)
    ghost = lambda k: [(i, [bget(w, c, k) for c in levels]) for i, (w, _) in enumerate(wl)] if base else None
    note = "baseline (dashed)" if base else None
    wl = [("prose", "Prose"), ("coding", "Coding"), ("counting", "Counting")]
    charts = {
        "decode-aggregate": dict(
            title="Aggregate decode throughput vs concurrency",
            subtitle=f"{MODEL_LABEL} · 1× RTX PRO 6000 @ 450 W · short prompts, 1,024 output tokens",
            series=[(n, [get(w, c, "steady_aggregate_tps") for c in levels]) for w, n in wl],
            ylabel="Output tokens/s (all requests)", ghosts=ghost("steady_aggregate_tps"), legend_note=note),
        "decode-per-user": dict(
            title="Per-user decode speed vs concurrency",
            subtitle="Median request, tokens after first token ÷ time after first token · same runs as aggregate",
            series=[(n, [get(w, c, "per_req_decode_tps_median") for c in levels]) for w, n in wl],
            ylabel="Output tokens/s per request", ghosts=ghost("per_req_decode_tps_median"), legend_note=note),
    }
    def ctx_name(n):
        return f"{round(n / 1000)}K ctx" if n % 1024 else f"{n // 1024}K ctx"
    cs = [(n, ctx_name(n)) for n in sorted({r["context"] for r in ctx})]
    cs = [x for x in cs if x[0] != 16384][:5]  # palette has 5 series; 16K sits between 8K and 32K
    cget = lambda n, c: next((r.get("steady_aggregate_tps") for r in ctx
                              if r["context"] == n and r["concurrency"] == c and "skipped" not in r), None)
    charts["decode-context"] = dict(
        title="Aggregate decode throughput at long context",
        subtitle=f"Prose task after a unique N-token document · 512 output tokens · lines end where the {pool / 1e3:,.0f}K-token KV pool is full",
        series=[(name, [cget(n, c) for c in levels]) for n, name in cs],
        ylabel="Output tokens/s (all requests)",
        label_nudge=LABEL_NUDGE)
    for key, spec in charts.items():
        for theme in THEMES:
            path = os.path.join(OUT, f"{key}-{theme}.svg")
            with open(path, "w") as f:
                f.write(line_chart(spec["title"], spec["subtitle"], xl, spec["series"],
                                   spec["ylabel"], theme, label_nudge=spec.get("label_nudge"),
                                   ghosts=spec.get("ghosts"), legend_note=spec.get("legend_note")))
            print("wrote", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
