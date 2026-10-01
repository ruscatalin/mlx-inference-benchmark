"""Main execution loop: stage materialization, server lifecycle, cell running, thermal gating."""
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column
from rich.text import Text

from bench import omlx, plan, quality, runstore
from bench import thermal as thermal_mod
from bench.workload import (
    PREFIX_QUESTIONS,
    SYNTH_PROMPT,
    build_prefix_document,
    clean_response,
    stream_one,
)

console = Console()


def _hms(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class _Plain:
    """Mixin: rows created with plain=True (the Thermal line) show only their text, no bar or timers."""

    def render(self, task):
        return Text("") if task.fields.get("plain") else super().render(task)


class PlainPercent(_Plain, TaskProgressColumn):
    pass


class PlainElapsed(_Plain, TimeElapsedColumn):
    pass


class SmoothBar(_Plain, BarColumn):
    """Bar drawn with eighth-cell blocks: 8 steps per character (rich's own bar has 2), so the
    bar moves four times as smoothly at the same width. Indeterminate tasks keep rich's pulse."""
    PARTIAL = ("", "▏", "▎", "▍", "▌", "▋", "▊", "▉")

    def render(self, task):
        if task.fields.get("plain") or task.total is None or not task.started:
            return super().render(task)
        width = self.bar_width or 40
        steps = round(width * 8 * min(max(task.percentage / 100, 0.0), 1.0))
        full, part = divmod(steps, 8)
        color = "rgb(114,156,31)" if task.finished else "rgb(249,38,114)"  # rich's bar.finished / bar.complete
        bar = Text("█" * full, style=color)
        if part:
            bar.append(self.PARTIAL[part], style=f"{color} on grey23")
        bar.append(" " * (width - full - (1 if part else 0)), style="on grey23")
        return bar


class EtaColumn(ProgressColumn):
    """Overall row: remaining planned seconds x (actual / planned seconds for the work done so far
    this session), so server starts and thermal gates are absorbed; before a minute of planned work
    is done it shows the plan estimate as is. Other rows: rich's recent-speed estimate."""

    def render(self, task):
        if task.total is None or task.fields.get("plain"):
            return Text("")
        if "t0" in task.fields:
            left = max(task.total - task.completed, 0)
            t0, done = task.fields["t0"], task.completed - task.fields["base"]
            if t0 is not None and done > 60:
                left *= (time.monotonic() - t0) / done
        else:
            left = task.time_remaining
        return Text(f"ETA {_hms(left) if left is not None else '-:--:--'}", style="progress.remaining")


class Ctx:
    """Shared state passed to thermal.calibrate/gate."""

    def __init__(self, args, out, run_meta, th, band):
        self.args = args
        self.out = out
        self.run_meta = run_meta
        self.th = th
        self.band = band
        self.server = omlx.Server()
        self.admin = omlx.Admin(args.api_key)
        self.thermal_state = ""
        self.plan_cells = list(run_meta["cells"].values())
        self.resp_lock = threading.Lock()

    def start_server(self, cell, c, log_name):
        model_name = cell["model"]
        profile_name = cell["profile"]
        repo = self.run_meta["models"][model_name]["repo"]
        model_dir_name = repo.split("/")[-1]
        model_dir = Path(self.run_meta["args"]["model_root"]).expanduser() / model_name / model_dir_name
        profile = self.run_meta["profiles"][profile_name]
        model_settings = {**(self.run_meta["models"][model_name].get("settings") or {}),
                          **(profile.get("model_settings") or {})}
        base = omlx.prepare_base(self.run_meta["run_id"], cell["id"], profile, model_dir_name, model_settings)
        cache_dir = self.out / "cache" / cell["id"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.server.start(model_dir, c, profile, base, cache_dir, self.out / "logs" / log_name,
                          api_key=self.args.api_key)
        if not self.server.wait_ready(self.args.ready_timeout, self.args.api_key):
            raise RuntimeError(f"server failed to start (see logs/{log_name})")
        settings_check = omlx.check_settings(self.admin, model_dir_name, model_settings)
        model = omlx.pick_model(model_dir_name, self.args.api_key)
        stream_one(model, "Say OK.", 16, self.args)
        self._last_base = base
        self._last_cache = cache_dir
        self._last_settings_check = settings_check
        return model

    def cleanup_cell_dirs(self):
        import shutil
        for p in (getattr(self, "_last_base", None), getattr(self, "_last_cache", None)):
            if p:
                shutil.rmtree(p, ignore_errors=True)


def run_cell(ctx, cell, model, prompts, progress, task, overall):
    args = ctx.args
    jobs, mt = cell["jobs"], args.max_tokens
    workload = cell["workload"]
    n = len(jobs) if workload != "prefix" else len(jobs) * 6
    lock = threading.Lock()
    live = {"tok": 0, "done": 0, "pass": 0, "scored": 0}
    bench_s = " · MMLU-Pro" if workload == "mmlu" else " · IFBench" if cell["scored"] else ""
    progress.reset(task, total=n * mt, description=f"{cell['stage']}{bench_s} · {cell['model']}/{cell['profile']} · "
                   f"C={cell['c']} · t{cell['trial']}", info="")
    fh = open(ctx.out / "responses.jsonl", "a")  # noqa: SIM115 (closed after the retry pass)
    scorer = getattr(ctx, "scorer", None)

    def on_token():
        with lock:
            live["tok"] += 1
        progress.advance(task, 1)

    abort = threading.Event()
    requests_out = []
    req_lock = threading.Lock()

    def record_request(rec, pid, extra=None):
        full = dict(rec)
        full["pid"] = pid
        if extra:
            full.update(extra)
        with req_lock:
            requests_out.append(full)
        return full

    def maybe_write_response(r, cell, pid, sample, group=None):
        if not (cell["scored"] or cell["workload"] == "ifbench"):
            return None
        if not r["ok"]:
            return None
        resp = clean_response(r.get("text", ""))
        full = {"cell": cell["id"], "model": cell["model"], "profile": cell["profile"], "c": cell["c"],
                "trial": cell["trial"], "pid": pid, "sample": sample, "tokens": r["tokens"], "response": resp}
        if workload == "mmlu":
            full.update(bench="mmlu", category=ctx.mmlu[pid]["category"])
        if cell["scored"] and (scorer or workload == "mmlu"):
            try:
                if workload == "mmlu":
                    sc = quality.mmlu_score(ctx.mmlu[pid], resp)
                    passed = sc["correct"]
                else:
                    sc = scorer.score(pid, resp)
                    passed = sc["loose"]
                full["score"] = sc
                with lock:
                    live["scored"] += 1
                    live["pass"] += passed
            except Exception as e:  # noqa: BLE001
                full["score_error"] = str(e)[:200]
        if cell["scored"]:
            with ctx.resp_lock:
                fh.write(json.dumps(full) + "\n")
                fh.flush()
        return resp

    failed = []   # (job, its failed request record), for the retry pass of scored cells

    def ifbench_job(j, retry=False):
        if abort.is_set():
            return {"ok": False, "error": "aborted", "tokens": 0, "latency": 0, "ttft": 0, "tps": 0, "pid": j[0]}
        pid, s = j
        if workload == "mmlu":
            prompt = quality.mmlu_prompt(ctx.mmlu[pid])
        else:
            prompt = SYNTH_PROMPT if pid == "syn" else prompts.get(pid, "")
        r = stream_one(model, prompt, mt, args, None if retry else on_token, abort=abort)
        if not retry and r.get("chunks", 0) < mt:
            progress.advance(task, mt - r.get("chunks", 0))
        maybe_write_response(r, cell, pid, s)
        if not retry:
            with lock:
                live["done"] += 1
        rec = record_request({k: r.get(k) for k in ("ok", "tokens", "latency", "ttft", "tps", "error")}, pid,
                             extra={"retry": True} if retry else None)
        if not r["ok"] and not retry and cell["scored"]:
            with lock:
                failed.append((j, rec))
        return rec

    def prefix_group(j):
        g = int(j[0][1:])
        doc = build_prefix_document(prompts, g)
        history = []
        for k, q in enumerate(PREFIX_QUESTIONS):
            if abort.is_set():
                break
            messages = [{"role": "system", "content": doc}]
            for jj, past_q in enumerate(PREFIX_QUESTIONS[:k]):
                messages.append({"role": "user", "content": past_q})
                messages.append({"role": "assistant", "content": history[jj]})
            messages.append({"role": "user", "content": q})
            r = stream_one(model, "", mt, args, on_token, abort=abort, messages=messages)
            if r.get("chunks", 0) < mt:
                progress.advance(task, mt - r.get("chunks", 0))
            history.append(clean_response(r.get("text", "")) if r.get("ok") else "")
            maybe_write_response(r, cell, f"g{g}t{k}", 0, group=g, turn=k)
            with lock:
                live["done"] += 1
            record_request({k2: r.get(k2) for k2 in ("ok", "tokens", "latency", "ttft", "tps", "error")},
                           f"g{g}t{k}", extra={"group": g, "turn": k, "warm": k > 0})

    job_fn = prefix_group if workload == "prefix" else ifbench_job
    score_label = "MMLU-Pro" if workload == "mmlu" else "IF loose"
    cell_est = plan.estimate_seconds([cell], ctx.run_meta["args"])
    cell_task = next(t for t in progress.tasks if t.id == task)
    t_start, t0 = time.time(), time.perf_counter()
    ex = ThreadPoolExecutor(cell["c"])
    try:
        futs = [ex.submit(job_fn, j) for j in jobs]
        pending = set(futs)
        while pending:
            _, pending = wait(pending, timeout=0.25)
            el = max(time.perf_counter() - t0, 1e-6)
            q_ = f" · {score_label} {100 * live['pass'] / live['scored']:.0f}%" if live["scored"] else ""
            progress.update(task, info=f"req {live['done']}/{n} · {live['tok'] / el:6.1f} tok/s{q_}")
            progress.update(overall, completed=ctx.overall_done + cell_est * cell_task.percentage / 100)
        for fut in futs:
            fut.result()
    except BaseException:
        abort.set()
        ex.shutdown(wait=True, cancel_futures=True)
        fh.close()
        raise
    finally:
        ex.shutdown(wait=True)
    wall, t_end = time.perf_counter() - t0, time.time()
    total_tokens = sum(r["tokens"] for r in requests_out if r["ok"])   # speed figures exclude the retry pass

    # Quality cells: requests the server rejected (e.g. its memory guard at high concurrency) are not wrong
    # answers, so they get one more try, one at a time.
    retried = len(failed)
    if failed:
        for j, old in failed:
            progress.update(task, info=f"retrying {len(failed)} failed request(s) at C=1")
            ifbench_job(j, retry=True)
            requests_out.remove(old)
    fh.close()

    ok = [r for r in requests_out if r["ok"]]
    res = dict(cell)
    res.update({"requests": requests_out, "n": len(requests_out), "errors": len(requests_out) - len(ok),
               "retried": retried,
               "wall_s": wall, "total_tokens": total_tokens, "total_tps": total_tokens / wall if ok else None,
               "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
               "settings_check": getattr(ctx, "_last_settings_check", None)})
    if ctx.th and ctx.th.ok:
        w = ctx.th.window(t_start, t_end)
        if w:
            pw = [s["power"] for s in w if isinstance(s["power"], (int, float))]
            freqs = [s["gpu_freq"] for s in w if s.get("gpu_freq") is not None]
            utils = [s["gpu_util"] for s in w if s.get("gpu_util") is not None]
            res["thermal"] = {"mean": statistics.mean(s["gpu"] for s in w), "max": max(s["gpu"] for s in w),
                              "power_w": statistics.mean(pw) if pw else None,
                              "samples": [[round(s["t"], 1), round(s["gpu"], 1), s.get("gpu_freq")]
                                          for s in w[::max(1, len(w) // 300)]]}
            res["gpu"] = {"freq_mean": statistics.mean(freqs) if freqs else None,
                         "freq_min": min(freqs) if freqs else None,
                         "util_mean": statistics.mean(utils) if utils else None}
            if pw and total_tokens:
                res["tok_per_j"] = total_tokens / (statistics.mean(pw) * wall)
    return res, live


def print_cell(r, live):
    if not r.get("total_tps"):
        console.print(f"[red]✗ {r['id']}: all requests failed[/]")
        return
    th = r.get("thermal") or {}
    g = r.get("gate") or {}
    parts = [f"[green]✓[/] [bold]{r['model']}/{r['profile']:<9}[/] C={r['c']:<2} t{r['trial']}"
             + (" MMLU-Pro" if r.get("workload") == "mmlu" else " IFBench" if r.get("scored") else ""),
             f"total [bold]{r['total_tps']:6.1f}[/] tok/s", f"err {r['errors']}" + (f" (retried {r['retried']})" if r.get("retried") else "")]
    if th:
        parts.append(f"{g.get('start_temp', float('nan')):.0f}→{th['max']:.0f}°C")
    if r.get("tok_per_j"):
        parts.append(f"{r['tok_per_j']:.2f} tok/J")
    if live["scored"]:
        label = "MMLU-Pro" if r.get("workload") == "mmlu" else "IF loose"
        parts.append(f"{label} {100 * live['pass'] / live['scored']:.1f}%")
    console.print(" │ ".join(parts))


def save_plan(out, run_meta):
    runstore.save_run_json(out, run_meta)


def planned_cells(run_meta, prompts, seed, mmlu=None):
    """Materialized cells plus the ones the not-yet-materialized stages will get (materialized
    into a scratch copy, so the real plan is untouched)."""
    scratch = {k: run_meta[k] for k in ("models", "profiles", "args")}
    scratch["cells"] = dict(run_meta["cells"])
    cells = list(run_meta["cells"].values())
    for stage in run_meta["stages"]:
        if not stage.get("cells"):
            cells += plan.materialize(dict(stage), scratch, prompts, seed, mmlu)
    return cells


def run(args, out, run_meta, cfg, prompts, scorer, thermal, band, mmlu=None):
    ctx = Ctx(args, out, run_meta, thermal, band)
    ctx.scorer = scorer
    ctx.mmlu = mmlu or {}

    stages = run_meta["stages"]
    done_ids = {p.stem for p in (out / "cells").glob("*.json")}
    rp = out / "responses.jsonl"
    if rp.exists():
        keep = [x for x in rp.read_text().splitlines() if x.strip() and json.loads(x)["cell"] in done_ids]
        rp.write_text("".join(k + "\n" for k in keep))

    progress = Progress(SpinnerColumn(), TextColumn("[bold]{task.description:<30}", table_column=Column(no_wrap=True, overflow="crop")), SmoothBar(bar_width=28),
                        PlainPercent(), TextColumn("{task.fields[info]}", table_column=Column(no_wrap=True, overflow="crop")), PlainElapsed(),
                        EtaColumn(), console=console, refresh_per_second=12)
    stop_status = threading.Event()

    def status_text():
        if not (ctx.th and ctx.th.ok):
            return " · ".join(x for x in ("thermal: off", ctx.thermal_state) if x)
        g = ctx.th.gpu()
        band_s = f"band {ctx.band[0]:.0f}–{ctx.band[1]:.0f}°C" if ctx.band else "band: pending"
        return " · ".join(x for x in (f"GPU {g:.1f}°C" if g is not None else "", band_s, ctx.thermal_state) if x)

    def plan_total():
        cells = planned_cells(run_meta, prompts, args.seed, ctx.mmlu)
        progress.update(overall, total=sum(plan.estimate_seconds([c], run_meta["args"]) for c in cells))
        return len(cells)

    with progress:
        ctx.overall_done = sum(plan.estimate_seconds([c], run_meta["args"])
                               for cid, c in run_meta["cells"].items() if cid in done_ids)
        overall = progress.add_task("Overall", total=None, completed=ctx.overall_done, info="",
                                    t0=None, base=ctx.overall_done)
        therm = progress.add_task("Thermal", total=None, info="", plain=True)
        cell_t = progress.add_task("", total=None, info="")

        def status_loop():
            while not stop_status.is_set():
                progress.update(therm, info=status_text())
                time.sleep(1)
        threading.Thread(target=status_loop, daemon=True).start()

        for stage in stages:
            if not stage.get("cells"):
                plan.materialize(stage, run_meta, prompts, args.seed, ctx.mmlu)
                save_plan(out, run_meta)
                break
        n_cells = plan_total()
        n_seen, t0_unset = len(done_ids), True

        if ctx.th and ctx.th.ok and ctx.band is None:
            ctx.plan_cells = list(run_meta["cells"].values())
            progress.update(overall, info="calibrating thermal band")
            progress.update(cell_t, description="calibration", info="finding steady-state temperature")
            lo, hi, tss = thermal_mod.calibrate(ctx, progress, cell_t)
            ctx.band = (lo, hi)
            run_meta["band"] = [lo, hi]
            run_meta["steady_state"] = tss
            save_plan(out, run_meta)
            progress.console.print(f"[green]✓[/] steady-state GPU {tss:.1f}°C → test band {lo:.0f}–{hi:.0f}°C")

        for stage in stages:
            if not stage.get("cells"):
                plan.materialize(stage, run_meta, prompts, args.seed, ctx.mmlu)
                save_plan(out, run_meta)
                n_cells = plan_total()
            todo_ids = [cid for cid in stage["cells"] if cid not in done_ids]
            for cid in todo_ids:
                cell = run_meta["cells"][cid]
                bench_s = " MMLU-Pro" if cell["workload"] == "mmlu" else " IFBench" if cell["scored"] else ""
                label = f"{cell['stage']}{bench_s} {cell['model']}/{cell['profile']} C={cell['c']} t{cell['trial']}"
                n_seen += 1
                progress.update(overall, info=f"cell {n_seen}/{n_cells} · stage {stage['name']}")
                if t0_unset:
                    progress.update(overall, t0=time.monotonic(), base=ctx.overall_done)
                    t0_unset = False
                progress.reset(cell_t, total=None, description=label, info="starting server ...")
                try:
                    model = ctx.start_server(cell, cell["c"], f"server_{cid}.log")
                    progress.update(cell_t, info="thermal gate ...")
                    g = thermal_mod.gate(ctx, model, cell["c"])
                    res, live = run_cell(ctx, cell, model, prompts, progress, cell_t, overall)
                    res["gate"] = g
                    (out / "cells" / f"{cid}.json").write_text(json.dumps(res))
                    done_ids.add(cid)
                    print_cell(res, live)
                except Exception as e:  # noqa: BLE001
                    progress.console.print(f"[red]✗ {label}: {e}[/]")
                finally:
                    ctx.overall_done += plan.estimate_seconds([cell], run_meta["args"])
                    progress.update(overall, completed=ctx.overall_done)
                    ctx.server.stop()
                    ctx.cleanup_cell_dirs()
                    time.sleep(1)
            stage["status"] = "done"
            save_plan(out, run_meta)
        stop_status.set()
