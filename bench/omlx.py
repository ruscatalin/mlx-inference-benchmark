"""oMLX server control: preflight, server start/stop, isolated base-path settings, admin checks."""
import contextlib
import http.cookiejar
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from bench.workload import headers, http_json

PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"


def configure(port):
    global PORT, BASE
    PORT = port
    BASE = f"http://127.0.0.1:{PORT}"


def quiet(cmd, timeout=60):
    with contextlib.suppress(Exception):
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, check=False)


def ensure_omlx():
    if shutil.which("omlx"):
        return
    sys.exit("oMLX is not installed (the `omlx` command was not found). Install it with:\n"
             "  brew tap jundot/omlx https://github.com/jundot/omlx && brew install omlx")


def ensure_model(repo, dest):
    if (dest / "config.json").exists():
        print(f"  ✓ {repo} (cached)")
        return
    from huggingface_hub import snapshot_download
    print(f"  ↓ {repo}")
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, local_dir=str(dest))


def free_port():
    quiet(["omlx", "stop"])
    quiet(["brew", "services", "stop", "omlx"])
    time.sleep(2)
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            sys.exit(f"Port {PORT} still in use - quit the oMLX menu-bar app and rerun.")


def preflight(port, api_key):
    """§4.1: run once before any server start."""
    ensure_omlx()
    configure(port)
    free_port()
    other_port = None
    try:
        settings = json.loads((Path.home() / ".omlx" / "settings.json").read_text())
        other_port = ((settings.get("server") or {}).get("port"))
    except Exception:  # noqa: BLE001
        other_port = None
    if other_port:
        try:
            http_json(f"http://127.0.0.1:{other_port}/v1/models", api_key, timeout=2)
            sys.exit(f"oMLX app server is running on port {other_port}; quit oMLX.app before benchmarking")
        except urllib.error.HTTPError:
            sys.exit(f"oMLX app server is running on port {other_port}; quit oMLX.app before benchmarking")
        except Exception:  # noqa: BLE001
            other_port = None  # the app's server isn't answering: nothing to conflict with


def pick_model(dir_name, key):
    ids = [m["id"] for m in http_json(BASE + "/v1/models", key, timeout=30)["data"]]
    for want in (dir_name.lower(), "gemma"):
        for i in ids:
            if want in i.lower():
                return i
    raise RuntimeError(f"model not found; server lists {ids}")


class Admin:
    def __init__(self, key):
        self.key = key
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def _req(self, method, path, body):
        req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method=method,
                                     headers=headers(self.key))
        try:
            with self.op.open(req, timeout=30) as r:
                return r.status, r.read().decode(errors="ignore")[:300]
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="ignore")[:300]
        except Exception as e:  # noqa: BLE001
            return 0, str(e)[:300]

    def login(self):
        if self.key:
            self._req("POST", "/admin/api/login", {"api_key": self.key, "remember": False})

    def get_settings(self, model_dir_name):
        code, msg = self._req("GET", f"/admin/api/models/{urllib.parse.quote(model_dir_name)}/settings", None)
        return code, msg


class Server:
    def __init__(self):
        self.proc = self.log = None

    def start(self, model_dir, concurrency, profile, base_path, cache_dir, log_path, api_key=None):
        srv = (profile or {}).get("server", {})
        cmd = ["omlx", "serve", "--model-dir", str(model_dir), "--port", str(PORT),
               "--base-path", str(base_path), "--max-concurrent-requests", str(concurrency)]
        if "hot_cache_max_size" in srv:
            cmd += ["--hot-cache-max-size", str(srv["hot_cache_max_size"])]
        no_cache = bool(srv.get("no_cache", False))
        if not no_cache:
            cmd += ["--paged-ssd-cache-dir", str(cache_dir)]
            if "paged_ssd_cache_max_size" in srv:
                cmd += ["--paged-ssd-cache-max-size", str(srv["paged_ssd_cache_max_size"])]
        if srv.get("hot_cache_write_through"):
            cmd += ["--hot-cache-write-through"]
        if "initial_cache_blocks" in srv:
            cmd += ["--initial-cache-blocks", str(srv["initial_cache_blocks"])]
        if no_cache:
            cmd += ["--no-cache"]
        if api_key:
            cmd += ["--api-key", str(api_key)]
        self.log = open(log_path, "w")  # noqa: SIM115 (closed in stop())
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_ready(self, timeout, key):
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                return False
            try:
                http_json(BASE + "/v1/models", key, timeout=3)
                return True
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise RuntimeError("server requires an API key: pass --api-key or set OMLX_API_KEY") from e
                time.sleep(1)
            except Exception:  # noqa: BLE001
                time.sleep(1)
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    os.killpg(self.proc.pid, signal.SIGKILL)
        if self.log:
            self.log.close()
        self.proc = self.log = None


# ---- §4.3 isolated base path -----------------------------------------------------------------
def prepare_base(run_id, cell_id, profile, model_dir_name, model_settings):
    base = Path.home() / ".omlx-bench" / "base" / run_id / cell_id
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    try:
        settings = json.loads((Path.home() / ".omlx" / "settings.json").read_text())
    except Exception:  # noqa: BLE001
        settings = {}
    settings.setdefault("cache", {})["hot_cache_only"] = bool((profile or {}).get("server", {}).get("hot_cache_only", False))
    (base / "settings.json").write_text(json.dumps(settings))
    try:
        ms = json.loads((Path.home() / ".omlx" / "model_settings.json").read_text())
    except Exception:  # noqa: BLE001
        ms = {"version": 1, "models": {}}
    ms.setdefault("models", {})
    ms["models"][model_dir_name] = {**ms["models"].get(model_dir_name, {}), **(model_settings or {})}
    (base / "model_settings.json").write_text(json.dumps(ms))
    return base


def check_settings(admin, model_dir_name, model_settings):
    """§4.4"""
    admin.login()
    code, msg = admin.get_settings(model_dir_name)
    if 200 <= code < 300:
        try:
            got = json.loads(msg) if isinstance(msg, str) else msg
        except Exception:  # noqa: BLE001
            got = None
        if isinstance(got, dict):
            mismatch = {}
            for k, wanted in (model_settings or {}).items():
                actual = got.get(k)
                if actual != wanted:
                    mismatch[k] = [wanted, actual]
            return {"verified": len(mismatch) == 0, "mismatch": mismatch}
    return {"verified": None, "reason": f"HTTP {code}"}

