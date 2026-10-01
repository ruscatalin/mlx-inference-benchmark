"""Plan construction: stages -> cells, per §5.3/§9. Cell ids are stable once materialized."""
import random

DEFAULT_IFBENCH_PROMPT_COUNT = 300


def fake_prompts(n=DEFAULT_IFBENCH_PROMPT_COUNT):
    """Placeholder pid->prompt map used for --dry-run counting, before IFBench is cloned."""
    return {f"p{i:03d}": "" for i in range(n)}


def fake_mmlu(n):
    """Placeholder MMLU-Pro qid->item map for --dry-run counting, before the dataset is downloaded."""
    return {f"mmlu/{i}": {} for i in range(n or 0)}


def resolve_stages(args, cfg):
    """Returns the list of stage skeletons {"name", "profiles", "workload", "levels"} for this run,
    from --preset. `default` is built in (quality + speed, profiles from --profiles); `baseline_quality`,
    `baseline_speed` and `exhaustive` list their stages in models.yaml."""
    if args.preset != "default":
        preset = cfg.presets.get(args.preset, {})
        stages = []
        for s in preset.get("stages", []):
            stages.append({"name": s["name"], "profiles": list(s["profiles"]),
                           "workload": s.get("workload", "ifbench"), "levels": s.get("levels")})
        return stages
    stages = []
    if not args.no_quality:
        stages.append({"name": "quality", "profiles": list(args.profiles), "workload": "ifbench", "levels": None})
    stages.append({"name": "speed", "profiles": list(args.profiles), "workload": args.workload,
                   "levels": list(args.levels)})
    return stages


def _cell_id(i, stage, model, profile, c, trial):
    return f"{i:03d}_{stage}_{model}_{profile}_c{c}_t{trial}"


def materialize(stage, run, prompts, seed, mmlu=None):
    """Fills stage['cells'] (list of ids) and run['cells'][id] = spec. Returns the new cell specs.
    The quality stage gets one IFBench cell and, when `mmlu` items are given, one MMLU-Pro cell per
    (model, profile)."""
    rng = random.Random(seed)
    models = list(run["models"])
    profiles = stage["profiles"]
    args_meta = run["args"]
    new_cells = []

    if stage["name"] == "quality":
        seen_settings = {}
        pairs = []
        for model in models:
            for profile in profiles:
                ms = _effective_settings(run, model, profile)
                repo = run["models"][model]["repo"]
                key = (repo, tuple(sorted(ms.items())))
                if key in seen_settings:
                    continue
                seen_settings[key] = True
                pairs.append((model, profile))
        for model, profile in pairs:
            pids = sorted(prompts)
            n_prompts = args_meta.get("quality_prompts")
            if n_prompts and n_prompts < len(pids):
                pids = rng.sample(pids, n_prompts)
            jobs = [[pid, s] for pid in pids for s in range(args_meta["samples"])]
            rng.shuffle(jobs)
            new_cells.append({"stage": "quality", "model": model, "profile": profile, "workload": "ifbench",
                              "c": args_meta["quality_c"], "trial": 0, "scored": True, "jobs": jobs})
            if mmlu:
                qids = sorted(mmlu, key=lambda k: int(k.split("/")[1]))
                n_q = args_meta.get("mmlu_questions")
                if n_q and n_q < len(qids):
                    qids = rng.sample(qids, n_q)
                jobs = [[qid, s] for qid in qids for s in range(args_meta["samples"])]
                rng.shuffle(jobs)
                new_cells.append({"stage": "quality", "model": model, "profile": profile, "workload": "mmlu",
                                  "c": args_meta["quality_c"], "trial": 0, "scored": True, "jobs": jobs})
    elif stage.get("workload") == "prefix":
        levels = _resolve_levels(stage["levels"], run)
        pids = sorted(prompts)
        for model in models:
            for profile in profiles:
                for c in levels:
                    for t in range(args_meta["trials"]):
                        groups = max(c, 2)
                        jobs = [[f"g{g}", 0] for g in range(groups)]
                        new_cells.append({"stage": stage["name"], "model": model, "profile": profile,
                                          "workload": "prefix", "c": c, "trial": t, "scored": False, "jobs": jobs})
    else:
        levels = stage["levels"] or []
        pids = sorted(prompts)
        for model in models:
            for profile in profiles:
                for c in levels:
                    for t in range(args_meta["trials"]):
                        n = max(args_meta["min_requests"], c * args_meta["rounds"])
                        pool = pids if pids else ["syn"]
                        n_eff = min(n, len(pool)) if pids else n
                        chosen = rng.sample(pool, n_eff) if pids else pool * n
                        jobs = [[pid, 0] for pid in (chosen if pids else [pool[0]] * n)]
                        new_cells.append({"stage": stage["name"], "model": model, "profile": profile,
                                          "workload": "ifbench", "c": c, "trial": t, "scored": False, "jobs": jobs})

    rng.shuffle(new_cells)
    start_i = len(run["cells"])
    ids = []
    for i, cell in enumerate(new_cells, start_i):
        stage_label = cell["stage"] + ("-mmlu" if cell["workload"] == "mmlu" else "")
        cid = _cell_id(i, stage_label, cell["model"], cell["profile"], cell["c"], cell["trial"])
        cell["id"] = cid
        run["cells"][cid] = cell
        ids.append(cid)
    stage["cells"] = ids
    return [run["cells"][cid] for cid in ids]


def _effective_settings(run, model, profile):
    m = run["models"].get(model, {})
    p = run["profiles"].get(profile, {})
    return {**(m.get("settings") or {}), **(p.get("model_settings") or {})}


def _resolve_levels(levels, run):
    """Resolve the literal 'knee' level using speed cells of (model, 'baseline'); default 2."""
    out = []
    for lv in levels or []:
        if lv == "knee":
            from bench import stats
            speed_cells = [c for c in run["cells"].values()
                           if c["stage"] == "speed" and c["profile"] == "baseline"]
            rows = []
            for c in speed_cells:
                total = c.get("total_tps")
                if total:
                    rows.append({"c": c["c"], "total": total})
            k = stats.throughput_knee(rows) if rows else None
            out.append(k["c"] if k else 2)
        else:
            out.append(int(lv))
    seen, dedup = set(), []
    for lv in out:
        if lv not in seen:
            seen.add(lv)
            dedup.append(lv)
    return dedup


def estimate_seconds(stage_cells, args_meta):
    total = 0.0
    for cell in stage_cells:
        n = len(cell["jobs"])
        c = cell["c"]
        if cell["workload"] == "prefix":
            groups = max(c, 2)
            total += 30 + 60 + groups / c * 6 * (args_meta["prefix_max_tokens"] / 40 + 2)
        else:
            total += 30 + 60 + n / c * args_meta["max_tokens"] / 40
    return total
