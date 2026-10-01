"""Shared statistics: t-CIs, bootstrap CIs, ratio CIs, kneedle, pareto, paired comparisons."""
import math
import random
import statistics

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def tcrit(df):
    keys = [k for k in T975 if k <= df]
    return T975[max(keys)] if keys else 1.96 if df > 30 else T975[1]


def mean_ci_t(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, None
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, None, None
    h = tcrit(len(xs) - 1) * statistics.stdev(xs) / math.sqrt(len(xs))
    return m, m - h, m + h


def boot_ci(xs, fn=statistics.mean, B=1000, seed=7):
    if len(xs) < 2:
        return None, None
    rng = random.Random(seed)
    n = len(xs)
    bs = sorted(fn([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def boot_ratio(a, b, B=1000, seed=7):
    if len(a) < 2 or len(b) < 2:
        return None, None, None
    rng = random.Random(seed)
    r0 = statistics.mean(a) / statistics.mean(b)
    bs = sorted(statistics.mean([a[rng.randrange(len(a))] for _ in a]) /
                statistics.mean([b[rng.randrange(len(b))] for _ in b]) for _ in range(B))
    return r0, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def kneedle(xs, ys, min_gain=0.05):
    """Knee of a concave-increasing curve: index with max height above the chord (normalized).
    Returns None when the curve is ~linear in the given x-space (no diminishing returns yet)."""
    if len(xs) < 3:
        return None
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    if x1 == x0 or y1 <= y0:
        return None
    d = [(y - y0) / (y1 - y0) - (x - x0) / (x1 - x0) for x, y in zip(xs, ys)]
    i = max(range(1, len(d) - 1), key=d.__getitem__)
    return i if d[i] >= min_gain else None


def throughput_knee(rows):
    """Knee of total tok/s vs log2(C): beyond it, raising C gains less than logarithmically."""
    rs = [r for r in rows if r.get("total")]
    if len(rs) < 3:
        return None
    peak = max(range(len(rs)), key=lambda i: rs[i]["total"])
    rs = rs[:peak + 1]
    if len(rs) < 3:
        return rs[-1] if len(rs) == 2 else None
    i = kneedle([math.log2(r["c"]) for r in rs], [r["total"] for r in rs])
    return rs[i] if i is not None else None


def pareto(points):
    """points: (x, y, obj) with both axes 'higher is better'; returns non-dominated points sorted by x."""
    front = [p for p in points if not any(q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1])
                                          for q in points)]
    return sorted(front, key=lambda p: p[0])


def pareto3(points):
    """points: (throughput, quality, cost, obj); throughput and quality higher-is-better, cost
    lower-is-better. Returns the objs not dominated in all three dimensions at once."""
    norm = [(t, q, -c, obj) for t, q, c, obj in points]
    return [obj for t, q, c, obj in norm
            if not any(t2 >= t and q2 >= q and c2 >= c and (t2 > t or q2 > q or c2 > c)
                       for t2, q2, c2, _ in norm)]


def recommend(by, best, q):
    """Quality first (statistically), then speed at the point of diminishing returns.
    1) eligible configs = best-quality config + any config NOT significantly worse (paired bootstrap)
    2) multi-user pick = eligible config with the highest throughput at its knee (or its plateau if no knee)
    3) single-user pick = eligible config with the highest per-request tok/s at C=1"""
    rec = {"eligible": [v for v in by if v in best], "quality_top": None, "vs_top": {}}
    if not rec["eligible"]:
        return rec
    if q:
        vq = [v for v in rec["eligible"] if v in q]
        if vq:
            top = max(vq, key=lambda v: q[v]["score"])
            rec["quality_top"] = top
            elig = [top]
            for v in vq:
                if v == top:
                    continue
                pr = paired(q, v, top)
                rec["vs_top"][v] = pr
                if pr and pr["hi"] >= 0:
                    elig.append(v)
            rec["eligible"] = elig
    multi = [v for v in rec["eligible"] if len(by[v]) > 1] or rec["eligible"]
    cands = [(v, best[v].get("knee") or best[v]["best"]) for v in multi]
    rec["multi"] = max(cands, key=lambda t: t[1]["total"])
    c1 = [(v, r) for v in rec["eligible"] for r in by[v] if r["c"] == 1 and r.get("req")]
    if c1:
        rec["single"] = max(c1, key=lambda t: t[1]["req"])
    return rec


def row_label(model, profile):
    return model if profile == "baseline" else f"{model}/{profile}"


def speed_summary(cells, meta):
    """cells: already filtered to one workload. Groups by (model, profile, c)."""
    from bench.workload import tps_floor
    mt, ratio = meta["args"]["max_tokens"], meta["args"]["ratio"]
    groups = {}
    for c in cells:
        groups.setdefault((c["model"], c["profile"], c["c"]), []).append(c)
    rows = {}
    for (model, profile, c), cs in groups.items():
        reqs = [r for x in cs for r in x["requests"]]
        ok = [r for r in reqs if r["ok"]]
        tps = [r["tps"] for r in ok if r["tokens"] >= tps_floor(mt)]
        m, lo, hi = mean_ci_t([x.get("total_tps") for x in cs])
        row = {"model": model, "profile": profile, "c": c, "trials": len(cs), "n": len(reqs),
               "errors": len(reqs) - len(ok),
               "total": m, "total_lo": lo, "total_hi": hi,
               "total_sd": statistics.stdev([x["total_tps"] for x in cs if x.get("total_tps")])
               if sum(1 for x in cs if x.get("total_tps")) >= 2 else None,
               "tokJ": statistics.mean([x["tok_per_j"] for x in cs if x.get("tok_per_j")])
               if any(x.get("tok_per_j") for x in cs) else None,
               "temp": statistics.mean([x["thermal"]["mean"] for x in cs if x.get("thermal")])
               if any(x.get("thermal") for x in cs) else None,
               "freq_mean": statistics.mean([x["gpu"]["freq_mean"] for x in cs
                                             if x.get("gpu") and x["gpu"].get("freq_mean")])
               if any(x.get("gpu") and x["gpu"].get("freq_mean") for x in cs) else None,
               "util_mean": statistics.mean([x["gpu"]["util_mean"] for x in cs
                                             if x.get("gpu") and x["gpu"].get("util_mean")])
               if any(x.get("gpu") and x["gpu"].get("util_mean") for x in cs) else None,
               "req_tps_list": tps}
        if tps:
            row["req"] = statistics.mean(tps)
            row["req_lo"], row["req_hi"] = boot_ci(tps)
            row["req_p5"] = pct(tps, 5)
        if ok:
            row["ttft"] = statistics.mean(r["ttft"] for r in ok)
            row["ttft_p95"] = pct([r["ttft"] for r in ok], 95)
            row["lat_p95"] = pct([r["latency"] for r in ok], 95)
            row["tokens_avg"] = statistics.mean(r["tokens"] for r in ok)
        if tps and ok:
            row["timeout"] = math.ceil(2 * (row["ttft_p95"] + mt / max(row["req_p5"], 1e-6)))
        rows[(model, profile, c)] = row
    by = {}
    for r in rows.values():
        by.setdefault((r["model"], r["profile"]), []).append(r)
    best = {}
    for key, rs in by.items():
        rs.sort(key=lambda r: r["c"])
        base = next((r.get("req") for r in rs if r["c"] == 1), None)
        elig = [r for r in rs if r["total"] and r["n"] and r["errors"] / r["n"] <= 0.01
                and (base is None or r.get("req", 0) >= ratio * base)]
        if not elig:
            continue
        for r in rs:
            r["eligible"] = r in elig
        top = max(elig, key=lambda r: r["total"])
        margin = meta["args"].get("tie_margin", 0.05)

        def tied_with_top(r, top=top, margin=margin):
            gap = (top["total"] - r["total"]) / top["total"]
            if gap <= margin:
                return True
            if gap > 2 * margin or r["total_sd"] is None or top["total_sd"] is None:
                return False
            se = math.sqrt(r["total_sd"] ** 2 / r["trials"] + top["total_sd"] ** 2 / top["trials"])
            df = max(1, min(r["trials"], top["trials"]) - 1)
            return (top["total"] - r["total"]) <= tcrit(df) * se

        tied = [r for r in elig if tied_with_top(r)]
        best[key] = {"best": min(tied, key=lambda r: r["c"]), "top": top, "knee": throughput_knee(elig)}
    return by, best


def pct(xs, p):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, math.ceil(p / 100 * len(xs)) - 1))] if xs else float("nan")


def _pct_ci(boots):
    bs = sorted(boots)
    return bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]


def quality_summary(responses, B=1000):
    """Per (model, profile): IFBench (prompt/instruction x loose/strict) and MMLU-Pro accuracy, each
    with a prompt-clustered bootstrap 95% CI, plus the fused quality "score": the equal-weight mean
    of the benchmarks that ran (50% IF loose + 50% MMLU-Pro when both did). The score CI resamples
    the two benchmarks' items independently and fuses each bootstrap draw."""
    by, by_mmlu = {}, {}
    for r in responses:
        if "score" not in r:
            continue
        if r.get("bench") == "mmlu":
            by_mmlu.setdefault((r["model"], r["profile"]), {}).setdefault(r["pid"], []).append(r)
        else:
            by.setdefault((r["model"], r["profile"]), {}).setdefault(r["pid"], []).append(r["score"])
    out, boots_by = {}, {}
    for v, pp in by.items():
        pids = sorted(pp)
        rate_l = {p: statistics.mean(s["loose"] for s in pp[p]) for p in pids}
        rate_s = {p: statistics.mean(s["strict"] for s in pp[p]) for p in pids}
        inst_l = {p: (sum(sum(s["inst_loose"]) for s in pp[p]), sum(len(s["inst_loose"]) for s in pp[p])) for p in pids}
        inst_s = {p: (sum(sum(s["inst_strict"]) for s in pp[p]), sum(len(s["inst_strict"]) for s in pp[p])) for p in pids}

        def ratio_stat(d, sel):
            n = sum(d[p][0] for p in sel)
            den = sum(d[p][1] for p in sel)
            return n / den if den else 0.0

        rng = random.Random(11)
        boots = {"pl": [], "ps": [], "il": [], "is": []}
        for _ in range(B):
            sel = [pids[rng.randrange(len(pids))] for _ in pids]
            boots["pl"].append(statistics.mean(rate_l[p] for p in sel))
            boots["ps"].append(statistics.mean(rate_s[p] for p in sel))
            boots["il"].append(ratio_stat(inst_l, sel))
            boots["is"].append(ratio_stat(inst_s, sel))
        ci = {k: (sorted(b)[int(0.025 * B)], sorted(b)[int(0.975 * B) - 1]) for k, b in boots.items()}
        out[v] = {"prompts": len(pids), "responses": sum(len(pp[p]) for p in pids),
                  "samples": round(statistics.mean(len(pp[p]) for p in pids), 1),
                  "prompt_loose": statistics.mean(rate_l.values()), "prompt_loose_ci": ci["pl"],
                  "prompt_strict": statistics.mean(rate_s.values()), "prompt_strict_ci": ci["ps"],
                  "inst_loose": ratio_stat(inst_l, pids), "inst_loose_ci": ci["il"],
                  "inst_strict": ratio_stat(inst_s, pids), "inst_strict_ci": ci["is"],
                  "rates": rate_l}
        boots_by.setdefault(v, []).append(boots["pl"])
    for v, pp in by_mmlu.items():
        qids = sorted(pp)
        rates = {p: sum(r["score"]["correct"] for r in pp[p]) / len(pp[p]) for p in qids}
        vals = [rates[p] for p in qids]
        rng = random.Random(13)
        boot = [sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals) for _ in range(B)]
        cats = {}
        for p in qids:
            cats.setdefault(pp[p][0].get("category", "?"), []).append(rates[p])
        out.setdefault(v, {}).update({
            "mmlu": statistics.mean(vals), "mmlu_ci": _pct_ci(boot), "mmlu_rates": rates,
            "mmlu_questions": len(qids), "mmlu_samples": round(statistics.mean(len(pp[p]) for p in qids), 1),
            "mmlu_cat": {c: statistics.mean(rs) for c, rs in sorted(cats.items())}})
        boots_by.setdefault(v, []).append(boot)
    for v, x in out.items():
        parts = [x[k] for k in ("prompt_loose", "mmlu") if k in x]
        x["score"] = sum(parts) / len(parts)
        x["score_ci"] = _pct_ci([sum(t) / len(t) for t in zip(*boots_by[v])])
    return out


def paired(q, a, b, B=2000):
    """Paired bootstrap of the quality-score difference a - b, over the items both configs answered.
    Each benchmark's per-item differences are resampled independently and fused with equal weights."""
    if a not in q or b not in q:
        return None
    diffs = []
    for k in ("rates", "mmlu_rates"):
        if k in q[a] and k in q[b]:
            common = sorted(set(q[a][k]) & set(q[b][k]))
            if len(common) >= 10:
                diffs.append([q[a][k][p] - q[b][k][p] for p in common])
    if not diffs:
        return None
    rng = random.Random(3)
    bs = sorted(sum(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for d in diffs) / len(diffs)
                for _ in range(B))
    lo, hi = bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
    return {"diff": statistics.mean(statistics.mean(d) for d in diffs), "lo": lo, "hi": hi,
            "n": sum(len(d) for d in diffs), "sig": lo > 0 or hi < 0}
