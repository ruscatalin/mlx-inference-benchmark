"""Static dashboard: aggregates every run under `root` into runs/index.html (Plotly)."""
import json
from pathlib import Path

from bench import pricing, report, runstore, stats

TEMPLATE_PATH = Path(__file__).parent / "dashboard_template.html"


QKEYS = ("score", "prompt_loose", "mmlu")


def _qfields(x):
    """Fused quality score plus its two components, each with its CI (absent when that benchmark didn't run)."""
    return {kk: x[kk] for k in QKEYS if k in x for kk in (k, k + "_ci")}


def _run_payload(run):
    meta = run.meta
    ifbench_cells = [c for c in run.cells if c.get("workload") == "ifbench" and not c.get("scored")]
    prefix_cells = [c for c in run.cells if c.get("workload") == "prefix"]
    by, best = stats.speed_summary(ifbench_cells, meta)
    q = stats.quality_summary(run.responses) if run.responses else {}
    prefix_rows = report._prefix_summary(prefix_cells, meta) if prefix_cells else {}
    rec = stats.recommend(by, best, q)

    rows = []
    for key, rs in by.items():
        for r in rs:
            rows.append({"model": key[0], "profile": key[1], "c": r["c"], "total": r.get("total"),
                        "total_lo": r.get("total_lo"), "total_hi": r.get("total_hi"), "req": r.get("req"),
                        "req_lo": r.get("req_lo"), "req_hi": r.get("req_hi"), "tokJ": r.get("tokJ"),
                        "eligible": r.get("eligible", False)})
    quality_rows = [{"model": k[0], "profile": k[1], **_qfields(v)} for k, v in q.items()]
    prefix_out = [{"model": k[0], "profile": k[1], "c": k[2], **v} for k, v in prefix_rows.items()]
    mv, mrow = rec.get("multi", (None, None))
    sv, srow = rec.get("single", (None, None))
    qv = rec.get("quality_top")
    price = meta.get("pricing", {})
    eligible_costed = [(v, r) for v, rs in by.items() for r in rs if r.get("eligible") and r.get("tokJ")]
    cheapest = None
    if eligible_costed:
        cv, crow = min(eligible_costed, key=lambda t: pricing.cost_per_mtok(price.get("price_kwh", 0), t[1]["tokJ"]) or 1e9)
        cheapest = {"model": cv[0], "profile": cv[1], "c": crow["c"]}
    pts3 = [(r["total"], 100 * q[(v[0], v[1])]["score"] if (v[0], v[1]) in q else 0.0,
             pricing.cost_per_mtok(price.get("price_kwh", 0), r.get("tokJ")) or 0.0, v + (r["c"],))
            for v, rs in by.items() for r in rs if r.get("total") and r.get("eligible")]
    pareto3 = [{"model": v[0], "profile": v[1], "c": v[2]} for v in stats.pareto3(pts3)]
    quality_pick = None
    if qv and qv in q:
        quality_pick = {"model": qv[0], "profile": qv[1], **_qfields(q[qv])}
    recommendations = {
        "multi": ({"model": mv[0], "profile": mv[1], "c": mrow["c"]} if mrow else None),
        "single": ({"model": sv[0], "profile": sv[1], "c": srow["c"]} if srow else None),
        "cheapest": cheapest, "pareto3": pareto3, "quality": quality_pick,
    }
    return {"run_id": meta["run_id"], "started": meta.get("started"), "host": meta.get("host", {}),
            "omlx_version": meta.get("omlx_version"), "pricing": meta.get("pricing", {}),
            "models": meta.get("models", {}), "profiles": meta.get("profiles", {}),
            "rows": rows, "quality": quality_rows, "prefix": prefix_out,
            "recommendations": recommendations}


def build(root="runs") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload: dict = {"runs": []}
    for d in runstore.list_runs(root):
        try:
            run = runstore.load_run(d)
        except ValueError:
            continue
        payload["runs"].append(_run_payload(run))
    data_json = json.dumps(payload).replace("</", "<\\/")
    template = TEMPLATE_PATH.read_text()
    html = template.replace("/*__DATA__*/", data_json)
    out = root / "index.html"
    out.write_text(html)
    return out
