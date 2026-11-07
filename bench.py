#!/usr/bin/env python3
"""Ollama benchmark for macOS.

Talks only to the local Ollama server. Writes only inside its own run-* folder.
Settings (environment variables, all optional):
  BENCH_REPS      repeats per prompt (default 2)
  BENCH_CTX       context window in tokens (default 8192)
  BENCH_COOLDOWN  seconds to rest between models (default 30)
  BENCH_MODELS    comma list of name substrings to test (default all local models)
  BENCH_THINK     set to 1 to leave thinking mode on (default off)
  BENCH_HOST      Ollama URL (default http://127.0.0.1:11434)
"""
import ast
import collections
import csv
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime

HOST = os.environ.get("BENCH_HOST", "http://127.0.0.1:11434")
REPS = int(os.environ.get("BENCH_REPS", "2"))
NUM_CTX = int(os.environ.get("BENCH_CTX", "8192"))
COOLDOWN = int(os.environ.get("BENCH_COOLDOWN", "30"))
ONLY = [m.strip() for m in os.environ.get("BENCH_MODELS", "").split(",") if m.strip()]
USE_THINK = os.environ.get("BENCH_THINK") == "1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))


# ---------- prompts and checks ----------

def access_code(i):
    letters = "".join(chr(65 + (i * 7 + j * 11) % 26) for j in range(4))
    return letters + str((i * 37) % 100).zfill(2)


def haystack(n=140):
    rows = []
    for i in range(1, n + 1):
        rows.append("Entry %d: Station %d logged %d units on day %d. Access code: %s."
                    % (i, i % 17 + 1, 100 + (i * 13) % 90, i % 28 + 1, access_code(i)))
    return "\n".join(rows)


SAFE = {"len": len, "range": range, "set": set, "list": list, "dict": dict, "tuple": tuple,
        "str": str, "enumerate": enumerate, "zip": zip, "min": min, "max": max, "sum": sum,
        "any": any, "all": all, "reversed": reversed, "sorted": sorted,
        "isinstance": isinstance, "bool": bool, "int": int, "ord": ord}
BANNED = {"exec", "eval", "open", "compile", "__import__", "os", "sys", "subprocess",
          "globals", "locals", "getattr", "setattr", "input"}
TESTS = [("()", True), ("([]{})", True), ("(]", False), ("((", False), ("", True),
         ("{[()]}", True), ("{[(])}", False), (")(", False)]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def check_code(text):
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    src = m.group(1) if m else text
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "is_balanced"), None)
    if fn is None:
        return False
    keep = []
    for n in tree.body:
        if n is fn:
            keep.append(n)
        elif isinstance(n, ast.Assign):
            try:
                ast.literal_eval(n.value)
                keep.append(n)
            except Exception:
                pass
    mod = ast.Module(keep, [])
    for node in ast.walk(mod):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            return False
        if isinstance(node, ast.Name) and (node.id in BANNED or node.id.startswith("__")):
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False
    ns = {"__builtins__": SAFE, "deque": collections.deque}
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(3)
    try:
        exec(compile(ast.fix_missing_locations(mod), "<model>", "exec"), ns)
        f = ns["is_balanced"]
        return all(f(s) == expected for s, expected in TESTS)
    except Exception:
        return False
    finally:
        signal.alarm(0)


def check_reasoning(text):
    return bool(re.search(r"ANSWER:\s*\**\s*(0?6:00\s*p\.?m\.?|18:00|6\s*p\.?m\.?)", text, re.I))


def check_bullets(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return len(lines) == 5 and all(
        l.startswith("- ") and "," not in l and len(l[2:].split()) < 12 for l in lines)


def check_needle(text):
    return access_code(87).lower() in text.lower()


PROMPTS = [
    ("short_qa",
     "Explain in two sentences what unified memory is on Apple Silicon.",
     120, None),
    ("reasoning",
     "Train A leaves a station at 2:15 pm going 72 km/h. Train B leaves the same station on the "
     "same track at 3:00 pm going 90 km/h. At what time does B catch A? Show the steps briefly. "
     "End with a final line in the form ANSWER: h:mm am/pm.",
     400, check_reasoning),
    ("code",
     "Write a Python function is_balanced(s) that returns True when every (), [] and {} in the "
     "string s is closed in the correct order, else False. Return only one python code block.",
     400, check_code),
    ("instructions",
     "Write exactly 5 bullet points about why heat sinks matter in computers. Each line must "
     "start with '- ', contain fewer than 12 words, and contain no commas. Output only the bullets.",
     150, check_bullets),
    ("long_context",
     haystack() + "\n\nWhat is the Access code in Entry 87? Reply with the code only.",
     40, check_needle),
    ("long_generation",
     "Write a 600 word essay explaining how a heat sink and fan move heat away from a processor. "
     "Use plain language.",
     800, None),
]


# ---------- helpers ----------

def http(path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(HOST + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 2) if xs else None


def generate(model, prompt, num_predict):
    body = {"model": model, "prompt": prompt, "stream": True, "keep_alive": "10m",
            "options": {"temperature": 0, "seed": 42, "num_predict": num_predict,
                        "num_ctx": NUM_CTX}}
    if not USE_THINK:
        body["think"] = False
    for attempt in (0, 1):
        t0 = time.perf_counter()
        first = None
        text = []
        final = {}
        try:
            req = urllib.request.Request(HOST + "/api/generate", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                for line in r:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if (d.get("response") or d.get("thinking")) and first is None:
                        first = time.perf_counter() - t0
                    text.append(d.get("response", ""))
                    if d.get("done"):
                        final = d
            break
        except urllib.error.HTTPError as e:
            if attempt == 0 and e.code == 400 and "think" in body:
                del body["think"]
                continue
            raise
    wall = time.perf_counter() - t0
    ns = 1e9
    pe, ped = final.get("prompt_eval_count", 0), final.get("prompt_eval_duration", 0)
    ec, ed = final.get("eval_count", 0), final.get("eval_duration", 0)
    return {"text": "".join(text), "ttft_s": first, "wall_s": wall,
            "load_s": final.get("load_duration", 0) / ns,
            "prompt_tokens": pe, "prefill_tps": (pe / (ped / ns)) if ped else None,
            "gen_tokens": ec, "decode_tps": (ec / (ed / ns)) if ed else None,
            "total_s": final.get("total_duration", 0) / ns,
            "done_reason": final.get("done_reason")}


def unload(name):
    try:
        http("/api/generate", {"model": name, "keep_alive": 0}, timeout=120)
    except Exception:
        pass


def unload_all():
    try:
        for m in http("/api/ps").get("models", []):
            unload(m["name"])
    except Exception:
        pass
    time.sleep(2)


def thermal_state():
    out = sh(["pmset", "-g", "therm"])
    d = {}
    for k in ("CPU_Speed_Limit", "CPU_Scheduler_Limit", "CPU_Available_CPUs"):
        m = re.search(k + r"\s*=\s*(\d+)", out)
        if m:
            d[k] = int(m.group(1))
    d["raw"] = " | ".join(l.strip() for l in out.splitlines() if l.strip())[:300]
    return d


def mem_state():
    mp = sh(["memory_pressure"], 30)
    m = re.search(r"free percentage:\s*(\d+)%", mp)
    return {"swap": sh(["sysctl", "-n", "vm.swapusage"]).strip(),
            "free_pct": int(m.group(1)) if m else None}


def first_match(pattern, text):
    m = re.search(pattern, text)
    return m.group(1) if m else None


def hardware():
    batt = sh(["pmset", "-g", "batt"]).splitlines()
    mem = sh(["sysctl", "-n", "hw.memsize"]).strip()
    return {"chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]).strip(),
            "model_id": sh(["sysctl", "-n", "hw.model"]).strip(),
            "memory_gb": round(int(mem) / 2 ** 30, 1) if mem.isdigit() else None,
            "macos": sh(["sw_vers", "-productVersion"]).strip(),
            "gpu_cores": first_match(r"Total Number of Cores:\s*(\d+)",
                                     sh(["system_profiler", "SPDisplaysDataType"], 40)),
            "power": batt[0] if batt else "",
            "python": sys.version.split()[0]}


def flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict):
            for item in flat(v, p + k + "."):
                yield item
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield p + k, v


class Sampler:
    """Samples macmon in the background when installed. Needs no sudo."""

    def __init__(self):
        self.path = shutil.which("macmon")
        self.proc = None
        self.rows = []

    def start(self):
        if not self.path:
            return
        self.rows = []
        try:
            self.proc = subprocess.Popen([self.path, "pipe", "-i", "1000"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         universal_newlines=True)
        except Exception:
            self.proc = None
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.rows.append(json.loads(line))
            except Exception:
                pass

    def stop(self):
        if not self.proc:
            return None
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        cols = {}
        for r in list(self.rows):
            for k, v in flat(r):
                if "temp" in k or "power" in k:
                    cols.setdefault(k, []).append(v)
        self.proc = None
        return {k: {"mean": round(statistics.mean(v), 2), "max": round(max(v), 2)}
                for k, v in cols.items() if v}


def save(state):
    tmp = os.path.join(OUT, "results.json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, os.path.join(OUT, "results.json"))


def write_csv(rows):
    cols = ["model", "prompt", "rep", "correct", "ttft_s", "wall_s", "load_s", "prompt_tokens",
            "prefill_tps", "gen_tokens", "decode_tps", "total_s", "done_reason",
            "cpu_speed_limit_after", "error"]
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])


def summary(rows):
    names = [p[0] for p in PROMPTS]
    seen = []
    for r in rows:
        if r["model"] not in seen:
            seen.append(r["model"])
    print("\nDecode tokens per second by prompt")
    print("model".ljust(32) + "".join(n[:14].ljust(16) for n in names))
    for m in seen:
        line = m[:31].ljust(32)
        for n in names:
            v = mean([r.get("decode_tps") for r in rows
                      if r["model"] == m and r["prompt"] == n and not r.get("error")])
            line += str(v).ljust(16)
        print(line)


# ---------- main ----------

def main():
    try:
        ver = http("/api/version")
    except Exception as e:
        sys.exit("Ollama is not reachable at %s (%s). Start Ollama and run again." % (HOST, e))
    os.makedirs(os.path.join(OUT, "outputs"))
    try:
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    except Exception:
        pass

    models = []
    for t in http("/api/tags").get("models", []):
        name = t["name"]
        if t.get("remote_host") or t.get("remote_model") or name.endswith("cloud"):
            print("skip (cloud model):", name)
            continue
        if ONLY and not any(s in name for s in ONLY):
            continue
        try:
            caps = http("/api/show", {"model": name}).get("capabilities")
        except Exception:
            caps = None
        if caps is not None and "completion" not in caps:
            print("skip (not a text model):", name)
            continue
        d = t.get("details", {})
        models.append({"name": name, "size_gb": round(t.get("size", 0) / 1e9, 2),
                       "params": d.get("parameter_size"), "quant": d.get("quantization_level"),
                       "family": d.get("family")})
    models.sort(key=lambda m: m["size_gb"])
    if not models:
        sys.exit("No local text models found.")

    sampler = Sampler()
    if not sampler.path:
        print("macmon not found: no die temperatures or power draw. "
              "Install with: brew install macmon, then run again.")
    state = {"started": datetime.now().isoformat(timespec="seconds"),
             "ollama_version": ver.get("version"), "hardware": hardware(),
             "settings": {"reps": REPS, "num_ctx": NUM_CTX, "temperature": 0, "seed": 42,
                          "thinking": USE_THINK, "cooldown_s": COOLDOWN,
                          "temperature_source": "macmon" if sampler.path else None},
             "prompts": [{"name": p[0], "num_predict": p[2]} for p in PROMPTS],
             "models": [], "rows": []}
    total = len(models) * sum(1 if p[0] == "long_generation" else REPS for p in PROMPTS)
    done = 0
    print("Models (smallest first):", ", ".join(m["name"] for m in models))
    print("Runs planned:", total)

    for mi, m in enumerate(models):
        name = m["name"]
        unload_all()
        m["mem_before"] = mem_state()
        try:
            warm = generate(name, "Say hi.", 1)
            m["cold_load_s"] = round(warm["load_s"], 2)
            for e in http("/api/ps").get("models", []):
                if e.get("name") == name or e.get("model") == name:
                    m["loaded_size_gb"] = round(e.get("size", 0) / 1e9, 2)
                    sv = e.get("size_vram", 0)
                    m["gpu_share_pct"] = round(100 * sv / e["size"]) if e.get("size") else None
        except Exception as e:
            m["error"] = "load failed: %s" % e
            state["models"].append(m)
            save(state)
            print("FAILED to load", name, e)
            continue
        state["models"].append(m)
        for pname, prompt, npred, check in PROMPTS:
            reps = 1 if pname == "long_generation" else REPS
            for rep in range(1, reps + 1):
                done += 1
                row = {"model": name, "prompt": pname, "rep": rep}
                sampler.start()
                row["thermal_before"] = thermal_state()
                try:
                    res = generate(name, "[run %s]\n%s" % (uuid.uuid4().hex[:8], prompt), npred)
                except Exception as e:
                    res = None
                    row["error"] = str(e)
                row["thermal_after"] = thermal_state()
                row["cpu_speed_limit_after"] = row["thermal_after"].get("CPU_Speed_Limit")
                row["sensors"] = sampler.stop()
                if res:
                    text = res.pop("text")
                    row.update(res)
                    row["correct"] = check(text) if check else None
                    fn = "%s__%s__r%d.txt" % (name.replace("/", "_").replace(":", "_"), pname, rep)
                    with open(os.path.join(OUT, "outputs", fn), "w") as f:
                        f.write(text)
                row["mem_after"] = mem_state()
                state["rows"].append(row)
                state["updated"] = datetime.now().isoformat(timespec="seconds")
                state["progress"] = "%d/%d" % (done, total)
                save(state)
                print("[%d/%d] %s %s r%d: decode %s tok/s, prefill %s tok/s, ttft %s s, correct=%s %s"
                      % (done, total, name, pname, rep,
                         mean([row.get("decode_tps")]), mean([row.get("prefill_tps")]),
                         mean([row.get("ttft_s")]), row.get("correct"), row.get("error", "")))
        unload(name)
        if mi < len(models) - 1 and COOLDOWN:
            print("Cooling down %d s" % COOLDOWN)
            time.sleep(COOLDOWN)

    state["finished"] = datetime.now().isoformat(timespec="seconds")
    state["progress"] = "done"
    save(state)
    write_csv(state["rows"])
    summary(state["rows"])
    print("\nDone. Results are in:", OUT)


if __name__ == "__main__":
    main()
