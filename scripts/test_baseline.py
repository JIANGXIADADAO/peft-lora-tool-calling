"""Baseline: Qwen2.5-0.5B before LoRA — tool-calling capability."""

import os, json, re, torch

# ── CPU 安全：限制线程数 ──
N_THREADS = 2
torch.set_num_threads(N_THREADS)
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "../models/Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading model (CPU, {N_THREADS} threads)...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

TOOLS = """read(path), write(path,content), edit(path,old_string,new_string), grep(pattern,path), glob(pattern), bash(cmd), fetch(url)"""

SYSTEM = f"""You are a tool-calling assistant. Available tools: {TOOLS}.
Output ONLY a JSON object: {{"tool": "<name>", "params": {{...}}}}. No explanation."""

TESTS = [
    ("读取 /home/project/README.md", "read", {"path": "/home/project/README.md"}),
    ("创建 /tmp/hello.py，内容 print('hello')", "write", {"path": "/tmp/hello.py", "content": "print('hello')"}),
    ("把 /app/config.py 的 DEBUG=False 改成 DEBUG=True", "edit", {"path": "/app/config.py", "old_string": "DEBUG=False", "new_string": "DEBUG=True"}),
    ("搜索所有 .py 文件里的 TODO", "grep", {"pattern": "TODO", "path": "."}),
    ("列出 src/ 下所有 .py 文件", "glob", {"pattern": "src/**/*.py"}),
    ("运行 pip install requests", "bash", {"cmd": "pip install requests"}),
    ("抓取 https://api.github.com 的内容", "fetch", {"url": "https://api.github.com"}),
    ("打开 /etc/hosts 看看", "read", {"path": "/etc/hosts"}),
    ("在 /tmp/test.py 写入 import sys", "write", {"path": "/tmp/test.py", "content": "import sys"}),
    ("执行 ls -la /var/log", "bash", {"cmd": "ls -la /var/log"}),
    ("搜 /src 下所有包含 import os 的文件", "grep", {"pattern": "import os", "path": "/src"}),
    ("把 /app/main.py 中的 port=8080 替换为 port=3000", "edit", {"path": "/app/main.py", "old_string": "port=8080", "new_string": "port=3000"}),
    ("找一下项目里所有的 .json 文件", "glob", {"pattern": "**/*.json"}),
    ("请求 http://example.com/api/data", "fetch", {"url": "http://example.com/api/data"}),
    ("读一下 ~/.bashrc", "read", {"path": "~/.bashrc"}),
]


def predict(instruction):
    msg = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction},
    ]
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, temperature=1.0, pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    return raw


def evaluate(raw, exp_tool, exp_params):
    r = {"valid_json": False, "tool_ok": False, "params_ok": False}
    clean = raw.strip()
    # strip markdown fences
    if clean.startswith("```"):
        parts = clean.split("```")
        clean = parts[1] if len(parts) > 1 else clean
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    # try full JSON parse
    try:
        parsed = json.loads(clean)
        r["valid_json"] = True
    except:
        # regex fallback: find {...}
        m = re.search(r'\{[^{}]*\}', raw)
        if m:
            try:
                parsed = json.loads(m.group())
                r["valid_json"] = True
            except:
                return r
        else:
            return r

    r["tool_ok"] = parsed.get("tool") == exp_tool
    params = parsed.get("params", {})
    if isinstance(params, dict):
        ok = all(str(params.get(k, "")).strip() == str(v).strip() for k, v in exp_params.items())
        r["params_ok"] = ok
    return r


# ── Run ──
print(f"\n{'#':<4} {'Tool':<6} {'JSON':<6} {'ToolOK':<8} {'Params':<8} {'Raw'}")
print("-" * 80)

counts = {"json": 0, "tool": 0, "params": 0}
for i, (inst, exp_tool, exp_params) in enumerate(TESTS):
    raw = predict(inst)
    ev = evaluate(raw, exp_tool, exp_params)
    counts["json"] += ev["valid_json"]
    counts["tool"] += ev["tool_ok"]
    counts["params"] += ev["params_ok"]
    j, t, p = ("Y" if ev[k] else "N" for k in ["valid_json", "tool_ok", "params_ok"])
    print(f"{i+1:<4} {exp_tool:<6} {j:<6} {t:<8} {p:<8} {raw.replace(chr(10), '↵')[:50]}")

n = len(TESTS)
print(f"\n===== BEFORE LORA =====")
print(f"  JSON valid:   {counts['json']}/{n} ({100*counts['json']/n:.0f}%)")
print(f"  Tool correct: {counts['tool']}/{n} ({100*counts['tool']/n:.0f}%)")
print(f"  Params OK:    {counts['params']}/{n} ({100*counts['params']/n:.0f}%)")

with open("../data/baseline.json", "w") as f:
    json.dump(counts, f)
print("  → data/baseline.json")
