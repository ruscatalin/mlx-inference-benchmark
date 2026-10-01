"""Thermal control (macmon, no sudo): GPU temp/freq/util reading, heat-soak calibration, gating."""
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time

import mlx.core as mx


class Thermal:
    """Reads `macmon pipe` JSON lines (no sudo): GPU/CPU temperature, SoC power, GPU frequency/util."""

    def __init__(self):
        self.ok = False
        self.proc = None
        self.hist = []
        self.lock = threading.Lock()

    def start(self):
        exe = shutil.which("macmon")
        if not exe:
            return False
        self.proc = subprocess.Popen([exe, "pipe", "-i", "1000"], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()
        for _ in range(20):
            if self.gpu() is not None:
                break
            time.sleep(0.5)
        self.ok = self.gpu() is not None
        return self.ok

    def _read(self):
        for line in self.proc.stdout:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = d.get("temp") or {}
            gpu = t.get("gpu_temp_avg")
            u = d.get("gpu_usage")
            gpu_freq = u[0] if isinstance(u, list) and len(u) >= 2 else None
            gpu_util = u[1] if isinstance(u, list) and len(u) >= 2 else None
            s = {"t": time.time(), "gpu": gpu if isinstance(gpu, (int, float)) and gpu > 0 else None,
                 "cpu": t.get("cpu_temp_avg"), "power": d.get("all_power"),
                 "gpu_freq": gpu_freq, "gpu_util": gpu_util}
            with self.lock:
                self.hist.append(s)

    def gpu(self):
        with self.lock:
            return self.hist[-1]["gpu"] if self.hist else None

    def window(self, t0, t1):
        with self.lock:
            return [s for s in self.hist if t0 <= s["t"] <= t1 and s["gpu"] is not None]

    def slope(self, secs=60):
        now = time.time()
        pts = [(s["t"] - now, s["gpu"]) for s in self.window(now - secs, now)]
        if len(pts) < 10:
            return None
        mx = statistics.mean(p[0] for p in pts)
        my = statistics.mean(p[1] for p in pts)
        den = sum((x - mx) ** 2 for x, _ in pts)
        return 60 * sum((x - mx) * (y - my) for x, y in pts) / den if den else None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class Heater:
    """Raw GPU load (large MLX matmuls), independent of any inference server or its concurrency
    limit — so it can heat a c=1 cell just as fast as a c=8 one, and stops instantly (no server
    requests to drain/abort)."""

    SIZE = 4096  # matmul dim; big enough to saturate the GPU compute units
    N = 8        # parallel streams, fixed regardless of cell concurrency
    DEPTH = 4    # matmuls queued per submit, so the GPU never idles between syncs
    DUTY = 1.0   # target GPU busy fraction (Metal can't pin cores; duty-cycle is the proxy)
    PERIOD = 1.0 # s; all threads share one on/off schedule so the idle windows line up
    CPU = 8      # busy-loop processes (not threads: GIL) adding CPU heat to the shared die

    def __init__(self, model, args, n=N):
        n = max(n, self.N)
        self.stop_ev = threading.Event()
        self.threads = [threading.Thread(target=self._loop, daemon=True) for _ in range(n)]

    def _loop(self):
        s = mx.new_stream(mx.gpu)  # own command queue; the default stream serializes all threads
        with mx.stream(s):
            a = mx.random.normal((self.SIZE, self.SIZE))
            b = mx.random.normal((self.SIZE, self.SIZE))
            mx.eval(a, b)
            prev = None
            while not self.stop_ev.is_set():
                off = self.PERIOD - time.time() % self.PERIOD
                if off <= self.PERIOD * (1 - self.DUTY):  # idle window: drain, then sleep it out
                    if prev is not None:
                        mx.eval(prev)
                        prev = None
                    self.stop_ev.wait(self.PERIOD - time.time() % self.PERIOD)
                    continue
                cur = [mx.matmul(a, b) for _ in range(self.DEPTH)]
                mx.async_eval(cur)
                if prev is not None:
                    mx.eval(prev)  # keep one batch in flight while waiting on the previous
                prev = cur
            if prev is not None:
                mx.eval(prev)

    def start(self):
        self.procs = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(self.CPU)]
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_ev.set()
        for p in self.procs:
            p.kill()
            p.wait()
        for t in self.threads:
            t.join(timeout=60)


def calibrate(ctx, progress, cell_t):
    """ctx needs: args, server, th, start_server(cell_like, c, log_name)->model."""
    first = ctx.plan_cells[0]
    c = min(p["c"] for p in ctx.plan_cells)
    progress.update(cell_t, description="calibration", info="starting server")
    model = ctx.start_server(first, c, "calibration.log")
    heater = Heater(model, ctx.args)
    heater.start()
    t0 = time.time()
    deadline = t0 + ctx.args.calib_max
    try:
        floor = ctx.args.preheat_floor
        if floor > 0:
            while True:
                el = time.time() - t0
                g = ctx.th.gpu()
                ctx.thermal_state = "preheating"
                progress.update(cell_t, description="calibration: preheat",
                                info=f"{el:4.0f}s · target ≥{floor:.0f}°C")
                if (g is not None and g >= floor) or time.time() >= deadline:
                    break
                time.sleep(2)
        t1 = time.time()
        min_t = min(120, max(0, deadline - t1))
        while True:
            el = time.time() - t1
            sl = ctx.th.slope(60)
            info = f"{el:4.0f}s · slope {sl:+.2f}°C/min" if sl is not None else f"{el:4.0f}s"
            ctx.thermal_state = "heat-soaking"
            progress.update(cell_t, description="calibration: heat-soak", info=info)
            if (el >= min_t and sl is not None and abs(sl) < 0.3) or time.time() >= deadline:
                break
            time.sleep(2)
    finally:
        ctx.thermal_state = ""
        heater.stop()
        ctx.server.stop()
    ss = [s["gpu"] for s in ctx.th.window(time.time() - 25, time.time())]
    tss = statistics.mean(ss) if ss else (ctx.th.gpu() or 60.0)
    return round(tss - 5, 1), round(tss + 1, 1), round(tss, 1)


def gate(ctx, model, c):
    if not (ctx.th and ctx.th.ok and ctx.band):
        ctx.thermal_state = f"cooldown {ctx.args.cooldown}s"
        time.sleep(ctx.args.cooldown)
        ctx.thermal_state = ""
        return {"action": "fixed-cooldown"}
    lo, hi = ctx.band
    cur = lambda: ctx.th.gpu() if ctx.th.gpu() is not None else (lo + hi) / 2
    t0 = time.time()
    actions = []
    tol = 1.5
    for _ in range(3):
        if cur() > hi:
            actions.append("cool")
            t1 = time.time()
            while cur() > hi and time.time() - t1 < ctx.args.cool_max:
                ctx.thermal_state = f"cooling to ≤{hi:.0f}°C"
                time.sleep(2)
        elif cur() < lo:
            actions.append("heat")
            target = (lo + hi) / 2
            heater = Heater(model, ctx.args)
            heater.start()
            t1 = time.time()
            while cur() < target and time.time() - t1 < ctx.args.heat_max:
                ctx.thermal_state = f"heating to ~{target:.0f}°C"
                time.sleep(0.5)
            heater.stop()
            time.sleep(1)
            if cur() >= lo - tol:
                break
        else:
            break
    ctx.thermal_state = ""
    action = "+".join(actions) or "in-band"
    g = cur()
    return {"action": action, "gate_s": round(time.time() - t0, 1), "start_temp": g,
            "in_band": lo - tol <= g <= hi + tol}
