"""Console report rendering: speed tables, quality table, prefix table, trade-off table, verdict."""
import math
import os
import statistics

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bench import pricing, stats

_console = Console()


def get_console():
    return _console


def record_console():
    return Console(record=True, width=160, file=open(os.devnull, "w"))


def f(x, spec=".1f", unit=""):
    return f"{x:{spec}}{unit}" if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)) else "-"


def ci(m, lo, hi, spec=".1f"):
    if m is None:
        return "-"
    return f"{m:{spec}} [{lo:{spec}}–{hi:{spec}}]" if lo is not None else f"{m:{spec}}"


def qci(x, k):
    """Percent with CI for quality key k of a quality_summary row ('-' when that benchmark didn't run)."""
    if not x or k not in x:
        return "-"
    return ci(100 * x[k], 100 * x[k + "_ci"][0], 100 * x[k + "_ci"][1])


def _row_quality(q, model, profile, run_meta):
    """§8.2: a speed row takes the quality of (model, profile); if absent, of (model, baseline)
    only if the profile has no model_settings. Otherwise '-'."""
    key = (model, profile)
    if key in q:
        return q[key]
    prof = run_meta.get("profiles", {}).get(profile, {})
    if not (prof.get("model_settings") or {}):
        return q.get((model, "baseline"))
    return None


def _throttled(ifbench_rows):
    """§8.3: flag rows whose freq_mean is >5% below the median of others at same (c, workload)."""
    groups = {}
    for r in ifbench_rows:
        groups.setdefault(r["c"], []).append(r)
    flags = {}
    for rs in groups.values():
        freqs = [r["freq_mean"] for r in rs if r.get("freq_mean")]
        if len(freqs) < 2:
            continue
        med = statistics.median(freqs)
        for r in rs:
            if r.get("freq_mean") and r["freq_mean"] < 0.95 * med:
                flags[(r["model"], r["profile"], r["c"])] = True
    return flags


def render_report(cells, responses, meta, con):
    ifbench_cells = [c for c in cells if c.get("workload") == "ifbench" and not c.get("scored")]
    prefix_cells = [c for c in cells if c.get("workload") == "prefix"]

    price = meta.get("pricing", {})
    sym = pricing.currency_symbol(price.get("currency", "EUR"))
    cost_hdr = f"{sym}/Mtok" + (" (est.)" if price.get("estimate") else "")

    by, best = stats.speed_summary(ifbench_cells, meta)
    thr_flags = _throttled([r for rs in by.values() for r in rs])

    q = stats.quality_summary(responses) if responses else {}

    for (model, profile) in sorted(by):
        rs = by[(model, profile)]
        label = stats.row_label(model, profile)
        repo = meta.get("models", {}).get(model, {}).get("repo", "")
        t = Table(title=f"[bold]{label}[/]  [dim]{repo}[/]", box=box.SIMPLE_HEAVY,
                  title_justify="left", header_style="bold", pad_edge=False)
        for col in ("C", "Total tok/s [95% CI]", "Avg tok/s/req [95% CI]", "p5 tok/s/req", "Avg TTFT",
                    "p95 lat", "Avg tok", "tok/J", cost_hdr, "GHz", "Util", "GPU °C", "Thr", "Err", "n", "Timeout*"):
            t.add_column(col, justify="right")
        b = best.get((model, profile), {}).get("best")
        for r in rs:
            star = r is b
            cost = pricing.cost_per_mtok(price.get("price_kwh", 0), r.get("tokJ"))
            thr = "⚠" if thr_flags.get((model, profile, r["c"])) else ""
            t.add_row(str(r["c"]), ci(r["total"], r["total_lo"], r["total_hi"]),
                      ci(r.get("req"), r.get("req_lo"), r.get("req_hi")), f(r.get("req_p5")),
                      f(r.get("ttft"), ".2f", "s"), f(r.get("lat_p95"), ".1f", "s"), f(r.get("tokens_avg"), ".0f"),
                      f(r.get("tokJ"), ".2f"), f(cost, ".3f"), f(r.get("freq_mean"), ".2f"),
                      f(100 * r.get("util_mean"), ".0f", "%") if r.get("util_mean") is not None else "-",
                      f(r.get("temp"), ".0f"), thr,
                      f"[red]{r['errors']}[/]" if r["errors"] else "0", str(r["n"]),
                      f(r.get("timeout"), "d", "s"), style="bold green" if star else None)
        con.print(t)

    if q:
        t = Table(title="[bold]Quality[/]  [dim]score = 50% IF loose + 50% MMLU-Pro · item-clustered bootstrap 95% CI[/]",
                  box=box.SIMPLE_HEAVY, title_justify="left", header_style="bold", pad_edge=False)
        for col in ("Model/profile", "Score", "IF prompt loose", "IF prompt strict", "IF inst loose", "IF inst strict",
                    "MMLU-Pro", "IF prompts", "MMLU q", "Samples"):
            t.add_column(col, justify="right")
        for key in sorted(q):
            x = q[key]
            t.add_row(stats.row_label(*key), "[bold]" + qci(x, "score") + "[/]", qci(x, "prompt_loose"),
                      qci(x, "prompt_strict"), qci(x, "inst_loose"), qci(x, "inst_strict"), qci(x, "mmlu"),
                      str(x.get("prompts", "-")), str(x.get("mmlu_questions", "-")),
                      f(x.get("samples", x.get("mmlu_samples")), ".1f"))
        con.print(t)
        cats = sorted({c for x in q.values() for c in x.get("mmlu_cat", {})})
        if cats:
            t = Table(title="[bold]MMLU-Pro by category[/]  [dim]accuracy %[/]", box=box.SIMPLE_HEAVY,
                      title_justify="left", header_style="bold", pad_edge=False)
            for col in ("Model/profile", *cats):
                t.add_column(col, justify="right")
            for key in sorted(q):
                mc = q[key].get("mmlu_cat", {})
                t.add_row(stats.row_label(*key), *(f(100 * mc[c], ".1f") if c in mc else "-" for c in cats))
            con.print(t)

    prefix_rows = _prefix_summary(prefix_cells, meta) if prefix_cells else {}
    if prefix_rows:
        t = Table(title="[bold]Prefix cache[/]", box=box.SIMPLE_HEAVY, title_justify="left",
                  header_style="bold", pad_edge=False)
        for col in ("Config", "C", "Uncached TTFT", "Cached TTFT [CI]", "speedup [CI]", "tok/s", cost_hdr):
            t.add_column(col, justify="right")
        for key in sorted(prefix_rows):
            r = prefix_rows[key]
            t.add_row(stats.row_label(key[0], key[1]), str(key[2]), f(r.get("cold_ttft"), ".2f", "s"),
                      ci(r.get("warm_ttft"), r.get("warm_ttft_lo"), r.get("warm_ttft_hi"), ".2f"),
                      ci(r.get("speedup"), r.get("speedup_lo"), r.get("speedup_hi"), ".2f"),
                      f(r.get("tok_s")), f(pricing.cost_per_mtok(price.get("price_kwh", 0), r.get("tokJ")), ".3f"))
        con.print(t)

    rec = stats.recommend(by, best, q)
    pts = [(r["total"], 100 * q[(v[0], v[1])]["score"] if (v[0], v[1]) in q else 0.0, v + (r["c"],))
           for v, rs in by.items() for r in rs if r.get("total") and r.get("eligible")]
    front = {p[2] for p in stats.pareto(pts)}
    pts3 = [(r["total"], 100 * q[(v[0], v[1])]["score"] if (v[0], v[1]) in q else 0.0,
             pricing.cost_per_mtok(price.get("price_kwh", 0), r.get("tokJ")) or 0.0, v + (r["c"],))
            for v, rs in by.items() for r in rs if r.get("total") and r.get("eligible")]
    front3 = set(stats.pareto3(pts3))
    mv, mrow = rec.get("multi", (None, None))
    sv, srow = rec.get("single", (None, None))
    t = Table(title="[bold]Trade-off[/]  [dim]each row = one config[/]", box=box.SIMPLE_HEAVY, title_justify="left",
              header_style="bold", pad_edge=False)
    for col in ("Config", "Quality [CI]", "Total tok/s", "Per-req tok/s", "p95 lat", cost_hdr,
                "Pareto", "Pareto 3D", "Role"):
        t.add_column(col, justify="right")
    for v in sorted(by):
        for r in by[v]:
            if not r.get("total"):
                continue
            b = best.get(v, {})
            role = []
            if r is mrow:
                role.append("[bold green]★ multi-user pick[/]")
            if r is srow:
                role.append("[bold green]★ single-user pick[/]")
            if r is b.get("knee"):
                role.append("knee")
            if r is b.get("best"):
                role.append("plateau")
            if not r.get("eligible"):
                role.append("[dim]below per-req floor / errors[/]")
            rq = _row_quality(q, v[0], v[1], meta)
            cost = pricing.cost_per_mtok(price.get("price_kwh", 0), r.get("tokJ"))
            t.add_row(f"{stats.row_label(*v)} · C={r['c']}",
                      qci(rq, "score"),
                      f(r["total"]), f(r.get("req")), f(r.get("lat_p95"), ".1f", "s"), f(cost, ".3f"),
                      "●" if v + (r["c"],) in front else "", "●" if v + (r["c"],) in front3 else "",
                      ", ".join(role))
    con.print(t)

    lines = []
    if mrow:
        why = "throughput knee" if mrow is best[mv].get("knee") else "throughput plateau (no clear knee)"
        lines.append(f"[bold]Recommended multi-user[/]  {stats.row_label(*mv)} · --max-concurrent-requests "
                     f"[bold]{mrow['c']}[/] ({why}) · {ci(mrow['total'], mrow['total_lo'], mrow['total_hi'])} tok/s total · "
                     f"{f(mrow.get('req'))} tok/s/req · timeout {f(mrow.get('timeout'), 'd', 's')}")
    if srow:
        lines.append(f"[bold]Recommended single-user[/] {stats.row_label(*sv)} · C=1 · "
                     f"{ci(srow['req'], srow.get('req_lo'), srow.get('req_hi'))} tok/s · "
                     f"timeout {f(srow.get('timeout'), 'd', 's')}")
    qv = rec.get("quality_top")
    if qv and qv in q:
        qrow = q[qv]
        lines.append(f"[bold]Best quality[/]        {stats.row_label(*qv)} · "
                     f"{qci(qrow, 'score')}% quality score"
                     + "".join(f" · {qci(qrow, k)}% {lbl}" for k, lbl in (("prompt_loose", "IF loose"), ("mmlu", "MMLU-Pro"))
                               if k in qrow))
    if prefix_rows:
        best_prefix = None
        for key, r in prefix_rows.items():
            if r.get("speedup_lo") and r["speedup_lo"] > 1 \
                    and (best_prefix is None or r["speedup"] > prefix_rows[best_prefix]["speedup"]):
                best_prefix = key
        if best_prefix:
            lines.append(f"[bold]Best prefix profile[/]  {stats.row_label(best_prefix[0], best_prefix[1])} "
                         f"C={best_prefix[2]} · speedup {prefix_rows[best_prefix]['speedup']:.2f}x")
    eligible_costed = [(v, r) for v, rs in by.items() for r in rs if r.get("eligible") and r.get("tokJ")]
    if eligible_costed:
        cv, crow = min(eligible_costed, key=lambda t: pricing.cost_per_mtok(price.get("price_kwh", 0), t[1]["tokJ"]) or 1e9)
        lines.append(f"[bold]Cheapest {cost_hdr}[/]  {stats.row_label(*cv)} · C={crow['c']} · "
                     f"{f(pricing.cost_per_mtok(price.get('price_kwh', 0), crow['tokJ']), '.3f')}")
    if front3:
        picks = ", ".join(f"{stats.row_label(v[0], v[1])} · C={v[2]}" for v in sorted(front3))
        lines.append(f"[bold]3D Pareto (speed/quality/cost)[/]  {picks}")
    if meta.get("args", {}).get("trials", 3) < 3:
        lines.append("[yellow]Only <3 trials per config: throughput CIs are very wide; use --trials ≥ 3.[/]")
    gates = [c.get("gate", {}) for c in cells]
    out_band = sum(1 for g in gates if g.get("in_band") is False)
    if meta.get("band"):
        lines.append(f"[bold]Thermal[/]           band {meta['band'][0]:.0f}–{meta['band'][1]:.0f}°C "
                     f"(steady state {meta.get('steady_state', float('nan')):.1f}°C); "
                     f"{out_band} of {len(cells)} cells started outside band")
    lines.append(f"[dim]Plateau = lowest C tied with the best total tok/s. Knee = diminishing returns vs log2(C). "
                 f"Eligible: ≤1% errors, per-request ≥ {meta['args']['ratio']:.0%} of C=1. "
                 f"⚠ = GPU freq > 5% below median at same C (hint only; decode is bandwidth-bound).[/]")
    con.print(Panel("\n".join(lines), title="Verdict", border_style="cyan", expand=False))
    return by, best, q, rec, prefix_rows


def _prefix_summary(cells, meta):
    out = {}
    groups = {}
    for c in cells:
        groups.setdefault((c["model"], c["profile"], c["c"]), []).append(c)
    for key, cs in groups.items():
        reqs = [r for x in cs for r in x["requests"] if r.get("ok")]
        cold = [r["ttft"] for r in reqs if not r.get("warm")]
        warm = [r["ttft"] for r in reqs if r.get("warm")]
        row = {}
        if cold:
            row["cold_ttft"] = statistics.mean(cold)
        if warm:
            m, lo, hi = stats.mean_ci_t(warm)
            row["warm_ttft"], row["warm_ttft_lo"], row["warm_ttft_hi"] = m, lo, hi
        if cold and warm:
            r0, lo, hi = stats.boot_ratio(cold, warm)
            row["speedup"], row["speedup_lo"], row["speedup_hi"] = r0, lo, hi
        total_tokens = sum(r["tokens"] for r in reqs)
        wall = sum(x.get("wall_s") or 0 for x in cs)
        if wall:
            row["tok_s"] = total_tokens / wall
        tokj = [x["tok_per_j"] for x in cs if x.get("tok_per_j")]
        if tokj:
            row["tokJ"] = statistics.mean(tokj)
        out[key] = row
    return out
