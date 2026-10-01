"""graphs.png / tradeoff.png rendering."""

from bench import pricing, stats

QUALITY_LABEL = "quality score (%) = 50% IF loose + 50% MMLU-Pro"
# 20 visually distinct hues (tab10 + 10 hand-picked non-duplicates); black/grey reserved for annotations.
DEFAULT_CYCLE = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
                 "#BCBD22", "#17BECF", "#0B3C5D", "#F2B701", "#00A86B", "#FF5A8C", "#6A3D9A", "#B15928",
                 "#66C2A5", "#E7298A", "#8DA0CB", "#666600"]


def color_map(*key_sets):
    """One unique, deterministic colour per (model, profile) series; shared by every graph of a run."""
    keys = sorted({k for ks in key_sets for k in ks})
    return {k: DEFAULT_CYCLE[i % len(DEFAULT_CYCLE)] for i, k in enumerate(keys)}


def plot_tradeoff(by, best, q, rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_q = bool(q)
    fig, axs = plt.subplots(1, 2 if has_q else 1, figsize=(16 if has_q else 8.5, 6.8), constrained_layout=True)
    axs = list(axs) if has_q else [axs]
    mv, mrow = rec.get("multi", (None, None))
    sv, srow = rec.get("single", (None, None))

    def style(ax, title, xl, yl):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    def ring(ax, x, y, label, color="black"):
        ax.scatter([x], [y], s=420, facecolor="none", edgecolor=color, lw=2.4, zorder=6)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(12, -16), fontsize=9,
                    fontweight="bold", color=color)

    colors = color_map(by, q)

    def col_for(key):
        return colors[key]

    if has_q:
        ax = axs[0]
        pts = []
        for key, rs in sorted(by.items()):
            if key not in q:
                continue
            rs = [r for r in rs if r.get("total")]
            y = 100 * q[key]["score"]
            lo, hi = (100 * z for z in q[key]["score_ci"])
            xs = [r["total"] for r in rs]
            col = col_for(key)
            ax.axhspan(lo, hi, color=col, alpha=0.08, zorder=1)
            if len(xs) > 1:
                ax.plot(xs, [y] * len(xs), "-o", color=col, lw=2, ms=7, label=f"{stats.row_label(*key)}", zorder=3)
            else:
                ax.plot(xs, [y] * len(xs), marker="o", markerfacecolor="none", markeredgewidth=2, ms=10,
                        color=col, linestyle="", label=f"{stats.row_label(*key)}", zorder=3)
            inel = [r for r in rs if not r.get("eligible")]
            if inel:
                ax.scatter([r["total"] for r in inel], [y] * len(inel), s=55, facecolor="white", edgecolor=col,
                           lw=1.5, zorder=4)
            for r in rs:
                ax.annotate(f"C={r['c']}", (r["total"], y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=col)
                if r.get("eligible"):
                    pts.append((r["total"], y, key + (r["c"],)))
        fr = stats.pareto(pts)
        if len(fr) > 1:
            ax.plot([p[0] for p in fr], [p[1] for p in fr], "--", color="black", lw=1, alpha=0.6,
                    label="Pareto frontier", zorder=2)
        if mrow and mv in q:
            ring(ax, mrow["total"], 100 * q[mv]["score"], "multi-user pick")
        if srow and sv in q and srow is not mrow:
            ring(ax, srow["total"], 100 * q[sv]["score"], "single-user pick", "#2E7D32")
        ax.scatter([], [], s=55, facecolor="white", edgecolor="grey", lw=1.5, label="below per-req floor")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        style(ax, "Quality vs throughput (point = model/profile x concurrency)", "total throughput (tok/s)",
              QUALITY_LABEL)

    ax = axs[-1]
    for key, rs in sorted(by.items()):
        rs = [r for r in rs if r.get("total") and r.get("req")]
        if not rs:
            continue
        col = col_for(key)
        if len(rs) > 1:
            ax.plot([r["req"] for r in rs], [r["total"] for r in rs], "-o", color=col, lw=2, ms=7,
                    label=stats.row_label(*key), zorder=3)
        else:
            ax.plot([rs[0]["req"]], [rs[0]["total"]], marker="o", markerfacecolor="none", markeredgewidth=2,
                    ms=10, color=col, linestyle="", label=stats.row_label(*key), zorder=3)
        inel = [r for r in rs if not r.get("eligible")]
        if inel:
            ax.scatter([r["req"] for r in inel], [r["total"] for r in inel], s=55, facecolor="white",
                       edgecolor=col, lw=1.5, zorder=4)
        for r in rs:
            ax.annotate(f"C={r['c']}", (r["req"], r["total"]), textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color=col)
        k = best.get(key, {}).get("knee")
        if k and k.get("req"):
            ax.scatter([k["req"]], [k["total"]], s=200, marker="D", facecolor="none", edgecolor=col, lw=1.8, zorder=5)
    if mrow and mrow.get("req"):
        ring(ax, mrow["req"], mrow["total"], "multi-user pick")
    if srow and srow.get("req") and srow is not mrow:
        ring(ax, srow["req"], srow["total"], "single-user pick", "#2E7D32")
    ax.scatter([], [], s=120, marker="D", facecolor="none", edgecolor="grey", label="throughput knee")
    ax.scatter([], [], s=55, facecolor="white", edgecolor="grey", lw=1.5, label="below per-req floor")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    style(ax, "Serving trade-off: more concurrency → throughput ↑, per-request speed ↓",
          "per-request speed (tok/s) - single-user feel", "total throughput (tok/s)")
    fig.suptitle("oMLX benchmark - speed/quality trade-off", fontweight="bold", fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot(cells, meta, by, best, q, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    price = meta.get("pricing", {})
    price_kwh = price.get("price_kwh", 0)
    levels = sorted({c["c"] for c in cells})
    fig, axs = plt.subplots(2, 4, figsize=(22, 9.5), constrained_layout=True)

    colors = color_map(by, q)  # same key sets as plot_tradeoff so both images agree

    def col_for(key):
        return colors.get(key, "#7F7F7F")

    def style(ax, title, ylab, xlab="max concurrent requests", xt=True):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)
        if xt:
            ax.set_xticks(levels)
        ax.grid(alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def series(ax, key_fn, lo=None, hi=None):
        for key, rs in sorted(by.items()):
            rs = [r for r in rs if key_fn(r) is not None]
            if not rs:
                continue
            xs = [r["c"] for r in rs]
            ys = [key_fn(r) for r in rs]
            err = None
            if lo and all(r.get(lo) is not None for r in rs):
                err = [[y - r[lo] for y, r in zip(ys, rs)], [r[hi] - y for y, r in zip(ys, rs)]]
            col = col_for(key)
            if len(rs) == 1:
                ax.errorbar(xs, ys, yerr=err, fmt="o", markerfacecolor="none", markeredgewidth=2, ms=10,
                            color=col, capsize=4, label=stats.row_label(*key), zorder=4)
            else:
                ax.errorbar(xs, ys, yerr=err, fmt="-o", lw=2, ms=5, color=col, capsize=4,
                            label=stats.row_label(*key), zorder=3)
        ax.set_ylim(bottom=0)

    series(axs[0, 0], lambda r: r.get("total"), "total_lo", "total_hi")
    for key, b in best.items():
        r = b["best"]
        if len(by[key]) > 1:
            axs[0, 0].scatter([r["c"]], [r["total"]], s=280, facecolor="none", edgecolor=col_for(key), lw=2, zorder=5)
            axs[0, 0].annotate(f"best C={r['c']}", (r["c"], r["total"]), textcoords="offset points",
                               xytext=(0, 14), ha="center", fontsize=9, color=col_for(key))
    style(axs[0, 0], "Total throughput (95% CI)", "tok/s")
    axs[0, 0].legend(frameon=False, loc="lower right")
    series(axs[0, 1], lambda r: r.get("req"), "req_lo", "req_hi")
    style(axs[0, 1], "Average speed per request (95% CI)", "tok/s")
    series(axs[0, 2], lambda r: r.get("lat_p95"))
    style(axs[0, 2], "p95 request latency", "seconds")
    series(axs[0, 3], lambda r: pricing.cost_per_mtok(price_kwh, r.get("tokJ")))
    sym = pricing.currency_symbol(price.get("currency", "EUR"))
    style(axs[0, 3], f"Cost ({sym}/Mtok)", f"{sym}/Mtok")

    if any(r.get("tokJ") for rs in by.values() for r in rs):
        series(axs[1, 0], lambda r: r.get("tokJ"))
        style(axs[1, 0], "Energy efficiency (SoC power)", "tokens / joule")
    else:
        series(axs[1, 0], lambda r: r.get("ttft"))
        style(axs[1, 0], "Average time to first token", "seconds")

    ax = axs[1, 1]
    t0 = min((s[0] for c in cells for s in (c.get("thermal") or {}).get("samples", [])), default=None)
    if t0 is not None:
        for c in cells:
            s = (c.get("thermal") or {}).get("samples", [])
            if s:
                col = col_for((c["model"], c["profile"]))
                ax2 = ax.twinx()
                ax.plot([(x[0] - t0) / 60 for x in s], [x[1] for x in s], color=col, lw=1.2,
                        marker="." if len(s) < 5 else None)
                freqs = [x[2] for x in s if len(x) > 2 and x[2] is not None]
                if freqs:
                    ax2.plot([(x[0] - t0) / 60 for x in s if len(x) > 2 and x[2] is not None],
                             freqs, color=col, lw=0.8, linestyle=":")
                    ax2.set_ylabel("GPU freq (MHz)")
        if meta.get("band"):
            ax.axhspan(*meta["band"], color="grey", alpha=0.15, label="test band")
            ax.legend(frameon=False, loc="lower right")
        style(ax, "GPU temperature during measurements", "°C", "minutes since start", xt=False)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "thermal data unavailable", ha="center", va="center", color="grey")

    ax = axs[1, 2]
    if q:
        vs = sorted(q)
        bars = [(k, lab, a) for k, lab, a in (("score", "score", 1.0), ("prompt_loose", "IF loose", 0.6),
                                              ("mmlu", "MMLU-Pro", 0.35)) if any(k in q[v] for v in vs)]
        w = 0.8 / len(bars)
        for j, (k, lab, alpha) in enumerate(bars):
            vk = [(i, v) for i, v in enumerate(vs) if k in q[v]]
            ys = [100 * q[v][k] for _, v in vk]
            err = [[y - 100 * q[v][k + "_ci"][0] for y, (_, v) in zip(ys, vk)],
                   [100 * q[v][k + "_ci"][1] - y for y, (_, v) in zip(ys, vk)]]
            xs = [i + (j - (len(bars) - 1) / 2) * w for i, _ in vk]
            ax.bar(xs, ys, w, yerr=err, capsize=3, color=[col_for(v) for _, v in vk], alpha=alpha, label=lab,
                   hatch="//" if k == "mmlu" else None)
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels([stats.row_label(*v) for v in vs], rotation=20, ha="right")
        top = max(100 * q[v][k + "_ci"][1] for v in vs for k, _, _ in bars if k in q[v])
        ax.set_ylim(0, min(100, top + 15))
        ax.legend(frameon=False, loc="upper center", ncol=len(bars))
        style(ax, "Quality (95% CI): score = 50% IF loose + 50% MMLU-Pro", "%", "", xt=False)
    else:
        series(ax, lambda r: r.get("ttft"))
        style(ax, "Average time to first token", "seconds")

    axs[1, 3].axis("off")
    fig.suptitle("oMLX benchmark" + (" — IFBench workload" if meta.get("args", {}).get("workload") == "ifbench" else ""),
                 fontweight="bold", fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)
