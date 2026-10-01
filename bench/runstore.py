"""Run directory persistence: run.json, cells/*.json, responses.jsonl, finish()/rescore()."""
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Run:
    path: Path
    meta: dict
    cells: list
    responses: list


def save_run_json(out: Path, run_meta: dict):
    tmp = out / "run.json.tmp"
    tmp.write_text(json.dumps(run_meta))
    os.replace(tmp, out / "run.json")


def load_run(path) -> Run:
    path = Path(path)
    rp = path / "run.json"
    if not rp.exists():
        raise ValueError(f"not a schema-2 run: {path}")
    meta = json.loads(rp.read_text())
    if meta.get("schema") != 2:
        raise ValueError(f"not a schema-2 run: {path}")
    cells = [json.loads(p.read_text()) for p in sorted((path / "cells").glob("*.json"))]
    done = {c["id"] for c in cells}
    responses = []
    rp2 = path / "responses.jsonl"
    if rp2.exists():
        for line in rp2.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["cell"] in done:
                    responses.append(r)
    return Run(path=path, meta=meta, cells=cells, responses=responses)


def list_runs(root="runs"):
    root = Path(root)
    if not root.exists():
        return []
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "run.json").exists()]
    return sorted(dirs, key=lambda d: (json.loads((d / "run.json").read_text()).get("started", ""), d.name))


def rescore(out: Path):
    from bench import quality
    run = load_run(out)
    meta = run.meta
    ifbench_dir = meta["args"]["ifbench_dir"]
    data, _ = quality.ifbench_data(ifbench_dir)
    scorer = quality.IFScorer(ifbench_dir, data)
    rp = out / "responses.jsonl"
    rows = [json.loads(x) for x in rp.read_text().splitlines() if x.strip()] if rp.exists() else []
    mmlu = quality.mmlu_data(quality.MMLU_ALL_CATEGORIES) if any(r.get("bench") == "mmlu" for r in rows) else {}
    for r in rows:
        r.pop("score_error", None)
        try:
            if r.get("bench") == "mmlu":
                r["score"] = quality.mmlu_score(mmlu[r["pid"]], r["response"])
            else:
                r["score"] = scorer.score(r["pid"], r["response"])
        except Exception as e:  # noqa: BLE001
            r.pop("score", None)
            r["score_error"] = str(e)[:200]
    rp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"✓ re-scored {len(rows)} responses")


def finish(out: Path, open_graph, overrides=None):
    from bench import dashboard, plots, report
    run = load_run(out)
    if not run.cells:
        sys.exit("No completed cells.")
    meta = dict(run.meta)
    if overrides:
        meta.update(overrides)
    con = report.get_console()
    by, best, q, rec, _prefix_rows = report.render_report(run.cells, run.responses, meta, con)
    rec_console = report.record_console()
    report.render_report(run.cells, run.responses, meta, rec_console)
    rec_console.save_text(str(out / "summary.txt"))
    graph = out / "graphs.png"
    plots.plot(run.cells, meta, by, best, q, graph)
    trade = out / "tradeoff.png"
    plots.plot_tradeoff(by, best, q, rec, trade)
    con.print(f"\n✓ Graphs  {graph}\n✓ Trade   {trade}\n✓ Summary {out / 'summary.txt'}\n"
              f"✓ Raw     {out / 'cells'}/, {out / 'run.json'}"
              + (f", {out / 'responses.jsonl'}" if run.responses else ""))
    index = dashboard.build(str(out.parent))
    if open_graph and sys.platform == "darwin":
        from bench import omlx
        omlx.quiet(["open", str(graph)])
        omlx.quiet(["open", str(trade)])
        omlx.quiet(["open", str(index)])
