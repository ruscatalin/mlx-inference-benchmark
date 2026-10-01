"""Request generation helpers and workload job/definitions (ifbench, synthetic, prefix)."""
import json
import math
import re
import time
import urllib.error
import urllib.request

from bench import omlx

REQUEST_TIMEOUT = 1800
MIN_TOKENS_FOR_TPS = 64          # per-request tok/s only from responses with >= this many tokens


def tps_floor(max_tokens):
    return min(MIN_TOKENS_FOR_TPS, max(1, max_tokens // 2))


SYNTH_PROMPT = ("Write an extremely detailed technical guide (at least 3000 words) on designing a "
                "distributed key-value store: replication, consensus, partitioning, failure handling, "
                "and operations. Use many sections and examples. Do not stop early.")
THINK_RE = re.compile(r"<think>.*?</think>|\[THINK\].*?\[/THINK\]", re.DOTALL)


def headers(key):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def http_json(url, key, timeout=10):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers(key)), timeout=timeout) as r:
        return json.load(r)


def pct(xs, p):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, math.ceil(p / 100 * len(xs)) - 1))] if xs else float("nan")


def clean_response(text):
    text = THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def stream_one(model, prompt, max_tokens, args, on_token=None, abort=None, messages=None):
    msgs = messages if messages is not None else [{"role": "user", "content": prompt}]
    body = json.dumps({"model": model, "messages": msgs,
                       "max_tokens": max_tokens, "temperature": args.temperature, "top_p": args.top_p,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(omlx.BASE + "/v1/chat/completions", data=body, headers=headers(args.api_key))
    t0 = time.perf_counter()
    ttft, chunks_n, usage, text = None, 0, None, []
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
            for raw in r:
                if abort is not None and abort.is_set():
                    break
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {})
                    piece = d.get("content")
                    if piece:
                        text.append(piece)
                    if piece or d.get("reasoning_content") or d.get("reasoning"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chunks_n += 1
                        if on_token:
                            on_token()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:200], "chunks": chunks_n, "latency": time.perf_counter() - t0}
    lat = time.perf_counter() - t0
    toks = (usage or {}).get("completion_tokens") or chunks_n
    gen = lat - (ttft or 0)
    return {"ok": True, "tokens": toks, "chunks": chunks_n, "latency": lat, "ttft": ttft or 0.0,
            "tps": toks / gen if gen > 0 else 0.0, "text": "".join(text)}


# ---- prefix workload (§7.3) ----------------------------------------------------------------
PREFIX_QUESTIONS = [
    "Summarize section 3 in two sentences.",
    "What is the main argument of the document?",
    "List two potential weaknesses mentioned or implied.",
    "Propose one follow-up question a reviewer might ask.",
    "Restate the conclusion in your own words.",
    "Name one thing that could be clarified further.",
]


def build_prefix_document(prompts, group, min_chars=16000):
    """Deterministic document for a prefix-workload group: concatenated IFBench prompts in
    sorted-pid order, starting at an offset of group*50 pids so groups don't share a prefix."""
    pids = sorted(prompts)
    if not pids:
        return ""
    offset = (group * 50) % len(pids)
    ordered = pids[offset:] + pids[:offset]
    parts, total = [], 0
    i = 0
    while total < min_chars:
        pid = ordered[i % len(ordered)]
        parts.append(prompts[pid])
        total += len(prompts[pid]) + 2
        i += 1
        if i > 10 * len(ordered):
            break
    return "\n\n".join(parts)
