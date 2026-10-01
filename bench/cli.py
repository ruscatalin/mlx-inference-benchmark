"""CLI entry point: run/resume/report/rescore/dashboard subcommands."""
import argparse
import contextlib
import os
import re
import signal
import sys
from datetime import datetime
from pathlib import Path

from bench import bootstrap

bootstrap.ensure_deps()

from bench import config, omlx, plan, quality, runstore, sweep
from bench import thermal as thermal_mod
from bench.workload import stream_one

REPO_ROOT = Path(__file__).resolve().parent.parent


def _api_key(args):
    return args.api_key or os.environ.get("OMLX_API_KEY") or config.omlx_settings_key()


def build_run_parser(sub):
    p = sub.add_parser("run", help="run a new benchmark")
    p.add_argument("--preset", choices=["default", "baseline_quality", "baseline_speed", "exhaustive"], default="default")
    p.add_argument("--models", default=None, help="comma-separated model names (default: all in models.yaml)")
    p.add_argument("--profiles", default="baseline")
    p.add_argument("--levels", default="1,2,3,4,6,8")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--quality-c", type=int, default=None)
    p.add_argument("--quality-prompts", type=int, default=None,
                    help="subsample this many IFBench prompts for the quality stage (default: all)")
    p.add_argument("--no-quality", action="store_true")
    p.add_argument("--mmlu-questions", type=int, default=300,
                   help="MMLU-Pro questions sampled for the quality stage, from --mmlu-categories (default: 300)")
    p.add_argument("--mmlu-categories", default=",".join(quality.MMLU_CATEGORIES),
                   help="comma-separated MMLU-Pro categories (default: the math/logic-heavy ones)")
    p.add_argument("--no-mmlu", action="store_true", help="IFBench only in the quality stage")
    p.add_argument("--workload", choices=["ifbench", "prefix"], default="ifbench")
    p.add_argument("--prefix-max-tokens", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--min-requests", type=int, default=10)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temp-band", default="auto")
    p.add_argument("--no-thermal", action="store_true")
    p.add_argument("--calib-max", type=int, default=420)
    p.add_argument("--preheat-floor", type=float, default=50.0)
    p.add_argument("--heat-max", type=int, default=600)
    p.add_argument("--cool-max", type=int, default=900)
    p.add_argument("--cooldown", type=int, default=30)
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--tie-margin", type=float, default=0.05)
    p.add_argument("--price", type=float, default=None)
    p.add_argument("--currency", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--price-source", choices=["static", "web"], default="static")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=None)
    p.add_argument("--ready-timeout", type=int, default=240)
    p.add_argument("--models-file", default="models.yaml")
    p.add_argument("--pricing-file", default="pricing.yaml")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--name", default=None, help="optional label added to the run folder name")
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--ifbench-dir", default=str(Path.home() / ".omlx-bench" / "IFBench"))
    p.add_argument("--model-root", default=None)


def build_parser():
    p = argparse.ArgumentParser(prog="bench", description="oMLX benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)
    build_run_parser(sub)

    r = sub.add_parser("resume", help="continue an interrupted run")
    r.add_argument("dir")
    r.add_argument("--api-key", default=None)
    r.add_argument("--port", type=int, default=None)
    r.add_argument("--no-open", action="store_true")

    rp = sub.add_parser("report", help="re-render report + graphs")
    rp.add_argument("dir")
    rp.add_argument("--price", type=float, default=None)
    rp.add_argument("--currency", default=None)
    rp.add_argument("--region", default=None)
    rp.add_argument("--ratio", type=float, default=None)
    rp.add_argument("--tie-margin", type=float, default=None)
    rp.add_argument("--no-open", action="store_true")

    rs = sub.add_parser("rescore", help="re-score saved IFBench responses")
    rs.add_argument("dir")

    db = sub.add_parser("dashboard", help="rebuild the runs index dashboard")
    db.add_argument("--root", default="runs")
    db.add_argument("--no-open", action="store_true")

    sm = sub.add_parser("smoke", help="start each (model, profile) briefly and verify the server accepts it")
    sm.add_argument("--models", default=None, help="comma-separated model names (default: all in models.yaml)")
    sm.add_argument("--profiles", default=None, help="comma-separated profile names (default: all in models.yaml)")
    sm.add_argument("--models-file", default="models.yaml")
    sm.add_argument("--port", type=int, default=8000)
    sm.add_argument("--api-key", default=None)
    sm.add_argument("--ready-timeout", type=int, default=240)
    sm.add_argument("--model-root", default=None)

    return p


def _load_mmlu(args, run_meta):
    """MMLU-Pro items for the quality stage, or {} when disabled / no quality stage / an old run."""
    if getattr(args, "no_mmlu", True) or not any(s["name"] == "quality" for s in run_meta["stages"]):
        return {}
    return quality.mmlu_data(_mmlu_categories(args))


def _mmlu_categories(args):
    cats = [c.strip() for c in args.mmlu_categories.split(",") if c.strip()]
    unknown = set(cats) - set(quality.MMLU_ALL_CATEGORIES)
    if unknown:
        sys.exit(f"unknown MMLU-Pro categories: {', '.join(sorted(unknown))} "
                 f"(choose from: {', '.join(quality.MMLU_ALL_CATEGORIES)})")
    return cats


def _estimate_and_print(run_meta):
    total_s = 0.0
    lines = [f"Run {run_meta['run_id']}"]
    for stage in run_meta["stages"]:
        cells = [run_meta["cells"][c] for c in stage.get("cells", [])]
        n_req = sum(len(c["jobs"]) for c in cells)
        est = plan.estimate_seconds(cells, run_meta["args"]) if cells else 0.0
        total_s += est
        lines.append(f"  stage {stage['name']:<8} cells={len(cells):<4} requests={n_req:<6} "
                     f"est={est / 3600:.1f}h")
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    lines.append(f"Total estimate: ~{h}h {m:02d}m")
    for line in lines:
        print(line)
    return total_s


def _new_run_id(name=None):
    """DD-MM-YYYY_HHMM, plus `_<name>` if given; `-2`, `-3`, ... if the folder already exists."""
    rid = f"{datetime.now().astimezone():%d-%m-%Y_%H%M}"
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", name or "").strip("-.")
    if label:
        rid += f"_{label}"
    taken, n = rid, 1
    while (REPO_ROOT / "runs" / taken).exists():
        n += 1
        taken = f"{rid}-{n}"
    return taken


def _build_run_meta(args, cfg, run_id):
    models_sel = [m.strip() for m in args.models.split(",")] if args.models else sorted(cfg.models)
    profiles_sel = [p.strip() for p in args.profiles.split(",")] if isinstance(args.profiles, str) else args.profiles
    for m in models_sel:
        if m not in cfg.models:
            sys.exit(f"unknown model: {m}")
    for pname in profiles_sel:
        if pname not in cfg.profiles:
            sys.exit(f"unknown profile: {pname}")
    args.profiles = profiles_sel
    args.levels = [int(x) if x != "knee" else x for x in str(args.levels).split(",") if str(x).strip()]

    stages = plan.resolve_stages(args, cfg)

    args_dict = vars(args).copy()
    args_dict.pop("api_key", None)
    args_dict["models"] = models_sel

    stage_profiles = sorted({p for s in stages for p in s["profiles"]})
    for pname in stage_profiles:
        if pname not in cfg.profiles:
            sys.exit(f"unknown profile: {pname}")

    run_meta = {
        "schema": 2, "run_id": run_id, "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": {}, "omlx_version": None, "args": args_dict,
        "models": {m: cfg.models[m] for m in models_sel},
        "profiles": {p: cfg.profiles[p] for p in stage_profiles},
        "omlx_settings": {}, "band": None, "steady_state": None,
        "stages": [{"name": s["name"], "status": "pending", "cells": [], "profiles": s["profiles"],
                   "workload": s.get("workload"), "levels": s.get("levels")} for s in stages],
        "cells": {},
    }
    return run_meta


def cmd_run(args):
    cfg = config.load_models(args.models_file)
    pricing_yaml = config.load_pricing(args.pricing_file)
    if args.model_root:
        cfg.model_root = args.model_root
    else:
        args.model_root = cfg.model_root
    args.api_key = _api_key(args)
    if args.quality_c is None:  # explicit flag > preset's `quality_c` > 1
        args.quality_c = cfg.presets.get(args.preset, {}).get("quality_c", 1)

    run_id = _new_run_id(args.name)
    run_meta = _build_run_meta(args, cfg, run_id)

    from bench import pricing as pricing_mod
    run_meta["pricing"] = pricing_mod.resolve(args, pricing_yaml)

    # Materialize stages against a placeholder prompt set for counting/estimation, before any
    # network access (§9: dry-run must exit before preflight/IFBench clone/model download/server start).
    if not args.no_mmlu:
        _mmlu_categories(args)
    fake_prompts = plan.fake_prompts()
    fake_mmlu = {} if args.no_mmlu else plan.fake_mmlu(args.mmlu_questions)
    for stage in run_meta["stages"]:
        plan.materialize(stage, run_meta, fake_prompts, args.seed, fake_mmlu)

    _estimate_and_print(run_meta)
    if args.dry_run:
        return 0
    if not args.yes:
        resp = input("Proceed? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            return 1

    out = Path(args.out_dir).expanduser() if args.out_dir else REPO_ROOT / "runs" / run_id
    (out / "cells").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    run_meta["cells"] = {}
    for stage in run_meta["stages"]:
        stage["cells"] = []
    runstore.save_run_json(out, run_meta)

    omlx.preflight(args.port, args.api_key)

    prompts = {}
    scorer = None
    needs_ifbench = (not args.no_quality) or any(s.get("workload") in ("ifbench", "prefix")
                                                 for s in run_meta["stages"])
    if needs_ifbench:
        data, prompts = quality.ifbench_data(args.ifbench_dir)
        try:
            scorer = quality.IFScorer(args.ifbench_dir, data)
        except Exception as e:  # noqa: BLE001
            print(f"! IFBench scorer unavailable ({e}); responses saved, score later with `bench rescore`.")
    mmlu = _load_mmlu(args, run_meta)

    root = Path(args.model_root).expanduser()
    for m in run_meta["models"]:
        repo = run_meta["models"][m]["repo"]
        omlx.ensure_model(repo, root / m / repo.split("/")[-1])

    thermal = None
    if not args.no_thermal:
        thermal = thermal_mod.Thermal()
        if thermal.start():
            print(f"  ✓ macmon thermal readings (GPU {thermal.gpu():.1f}°C)")
        else:
            print(f"  ! no temperature readings (macmon missing or unsupported; install with `brew install macmon`) → fixed {args.cooldown}s cooldown")

    run_meta["cells"] = {}
    for stage in run_meta["stages"]:
        stage["cells"] = []
    band = tuple(run_meta["band"]) if run_meta.get("band") else None
    try:
        sweep.run(args, out, run_meta, cfg, prompts, scorer, thermal, band, mmlu)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Resume with: python -m bench resume {out}")
    finally:
        if thermal:
            thermal.stop()

    runstore.finish(out, not args.no_open)
    return 0


def cmd_resume(args):
    out = Path(args.dir).expanduser()
    run = runstore.load_run(out)
    run_meta = run.meta
    saved_args = argparse.Namespace(**run_meta["args"])
    saved_args.no_open = args.no_open
    if args.port is not None:
        saved_args.port = args.port
    saved_args.api_key = args.api_key or os.environ.get("OMLX_API_KEY") or config.omlx_settings_key()

    cfg = config.load_models(saved_args.models_file)
    omlx.preflight(saved_args.port, saved_args.api_key)

    prompts, scorer = {}, None
    needs_ifbench = any(c.get("workload") in ("ifbench", "prefix") for c in run_meta["cells"].values()) \
        or not saved_args.no_quality
    if needs_ifbench:
        data, prompts = quality.ifbench_data(saved_args.ifbench_dir)
        with contextlib.suppress(Exception):
            scorer = quality.IFScorer(saved_args.ifbench_dir, data)
    mmlu = _load_mmlu(saved_args, run_meta)

    thermal = None
    if not saved_args.no_thermal:
        thermal = thermal_mod.Thermal()
        thermal.start()
    band = tuple(run_meta["band"]) if run_meta.get("band") else None
    try:
        sweep.run(saved_args, out, run_meta, cfg, prompts, scorer, thermal, band, mmlu)
    finally:
        if thermal:
            thermal.stop()
    runstore.finish(out, not args.no_open)
    return 0


def cmd_report(args):
    out = Path(args.dir).expanduser()
    overrides = {}
    if args.price is not None or args.region is not None:
        run = runstore.load_run(out)
        from bench import pricing as pricing_mod
        pricing_yaml = config.load_pricing(run.meta["args"].get("pricing_file", "pricing.yaml"))
        ns = argparse.Namespace(price=args.price, currency=args.currency, region=args.region,
                                price_source="static")
        overrides["pricing"] = pricing_mod.resolve(ns, pricing_yaml)
    if args.ratio is not None or args.tie_margin is not None:
        run = runstore.load_run(out)
        a = dict(run.meta["args"])
        if args.ratio is not None:
            a["ratio"] = args.ratio
        if args.tie_margin is not None:
            a["tie_margin"] = args.tie_margin
        overrides["args"] = a
    runstore.finish(out, not args.no_open, overrides=overrides)
    return 0


def cmd_rescore(args):
    out = Path(args.dir).expanduser()
    runstore.rescore(out)
    return 0


def cmd_smoke(args):
    """Start each (model, profile) pair for a moment and confirm the server accepted the profile's
    settings, without running any benchmark cell. Skips pairs with settings already proven identical
    to one already checked for the same repo (same dedup rule as the quality stage, §5.3)."""
    cfg = config.load_models(args.models_file)
    if args.model_root:
        cfg.model_root = args.model_root
    else:
        args.model_root = cfg.model_root
    args.api_key = _api_key(args)
    args.temperature, args.top_p = 0.6, 0.95

    models_sel = [m.strip() for m in args.models.split(",")] if args.models else sorted(cfg.models)
    profiles_sel = [p.strip() for p in args.profiles.split(",")] if args.profiles else sorted(cfg.profiles)
    for m in models_sel:
        if m not in cfg.models:
            sys.exit(f"unknown model: {m}")
    for pname in profiles_sel:
        if pname not in cfg.profiles:
            sys.exit(f"unknown profile: {pname}")

    omlx.preflight(args.port, args.api_key)
    root = Path(args.model_root).expanduser()

    seen, results = {}, []
    for model in models_sel:
        repo = cfg.models[model]["repo"]
        model_dir_name = repo.split("/")[-1]
        model_dir = root / model / model_dir_name
        omlx.ensure_model(repo, model_dir)
        for profile_name in profiles_sel:
            profile = cfg.profiles[profile_name]
            model_settings = config.effective_model_settings(cfg, model, profile_name)
            key = (repo, tuple(sorted(model_settings.items())), tuple(sorted(profile["server"].items())))
            if key in seen:
                results.append((model, profile_name, "skip", f"same effective config as {seen[key]}"))
                continue
            seen[key] = f"{model}/{profile_name}"
            server = omlx.Server()
            admin = omlx.Admin(args.api_key)
            base = cache_dir = None
            try:
                base = omlx.prepare_base(f"smoke-{os.getpid()}", f"{model}_{profile_name}", profile,
                                         model_dir_name, model_settings)
                cache_dir = Path.home() / ".omlx-bench" / "base" / f"smoke-{os.getpid()}" / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                server.start(model_dir, 1, profile, base, cache_dir,
                            Path.home() / ".omlx-bench" / "smoke.log", api_key=args.api_key)
                if not server.wait_ready(args.ready_timeout, args.api_key):
                    results.append((model, profile_name, "fail", "server did not become ready"))
                    continue
                check = omlx.check_settings(admin, model_dir_name, model_settings)
                served = omlx.pick_model(model_dir_name, args.api_key)
                r = stream_one(served, "Say OK.", 16, args)
                if not r["ok"]:
                    results.append((model, profile_name, "fail", f"chat request failed: {r['error']}"))
                elif check["verified"] is False:
                    results.append((model, profile_name, "fail", f"settings not applied: {check['mismatch']}"))
                else:
                    note = "verified" if check["verified"] else f"unverified ({check.get('reason', '?')})"
                    results.append((model, profile_name, "ok", note))
            except Exception as e:  # noqa: BLE001
                results.append((model, profile_name, "fail", str(e)[:200]))
            finally:
                server.stop()
                for p in (base, cache_dir):
                    if p:
                        import shutil
                        shutil.rmtree(p, ignore_errors=True)

    for model, profile_name, status, note in results:
        mark = {"ok": "[green]✓[/]", "fail": "[red]✗[/]", "skip": "[dim]-[/]"}[status]
        from bench.report import get_console
        get_console().print(f"{mark} {model}/{profile_name}: {note}")
    return 1 if any(s == "fail" for _, _, s, _ in results) else 0


def cmd_dashboard(args):
    from bench import dashboard
    idx = dashboard.build(args.root)
    print(f"✓ dashboard: {idx}")
    if not args.no_open and sys.platform == "darwin":
        omlx.quiet(["open", str(idx)])
    return 0


def main(argv=None):
    signal.signal(signal.SIGINT, signal.default_int_handler)

    def _term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    p = build_parser()
    args = p.parse_args(argv)
    fn = {"run": cmd_run, "resume": cmd_resume, "report": cmd_report,
          "rescore": cmd_rescore, "dashboard": cmd_dashboard, "smoke": cmd_smoke}[args.cmd]
    return fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
