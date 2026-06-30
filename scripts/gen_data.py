"""Generate ~300 tool-calling training examples covering 7 tools."""

import json, random

random.seed(42)

PATHS = [
    "/home/project/README.md", "/app/config.py", "/tmp/hello.py", "~/.bashrc",
    "/etc/hosts", "/var/log/syslog", "/src/main.py", "/app/utils/helpers.py",
    "/home/user/notes.txt", "/data/input.csv", "/tmp/result.json", "/etc/nginx/nginx.conf",
    "/project/src/index.js", "/app/templates/base.html", "/home/project/Makefile",
    "/opt/app/settings.yaml", "/usr/local/bin/run.sh", "/docs/api.md",
    "/tests/test_app.py", "/config/database.yml",
]

CONTENTS = [
    'print("hello world")', 'import os\nimport sys', 'DEBUG = True',
    'port = 8080', 'LOG_LEVEL = "info"', 'ALLOWED_HOSTS = ["*"]',
    'DATABASE_URL = "postgresql://..."', 'SECRET_KEY = "changeme"',
    'export PATH=$PATH:/usr/local/bin', 'VERSION = "1.0.0"',
    'def main():\n    pass', 'class AppConfig:\n    pass',
    '#!/bin/bash\nset -e', '<html><body></body></html>',
    'FROM python:3.12', 'pip install flask',
]

SEARCH_TERMS = ["TODO", "FIXME", "import os", "import re", "DEBUG", "localhost",
                "API_KEY", "password", "print(", "def ", "class ", "TODO: refactor"]

URLS = [
    "https://api.github.com", "https://httpbin.org/json",
    "http://example.com/api/data", "https://jsonplaceholder.typicode.com/todos/1",
    "https://api.openai.com/v1/models", "https://r.jina.ai/http://example.com",
]

CMDS = [
    "pip install requests", "ls -la /var/log", "git status", "python -m pytest",
    "docker compose up -d", "curl -s http://localhost:8000/health",
    "cat /etc/os-release", "df -h", "ps aux | grep python",
    "npm install", "mkdir -p /tmp/build", "tar -xzf archive.tar.gz",
]


def gen_read():
    path = random.choice(PATHS)
    prompts = [
        f"读取 {path} 文件内容", f"打开 {path} 看看", f"帮我读一下 {path}",
        f"查看 {path} 里面有什么", f"展示 {path} 的内容", f"阅读 {path}",
        f"给我看看 {path}", f"读取文件 {path}", f"显示 {path} 的全文",
        f"翻看 {path}", f"读 {path} 这个文件", f"read {path}",
        f"请把 {path} 的内容读出来", f"看看 {path} 写了什么",
    ]
    return random.choice(prompts), {"tool": "read", "params": {"path": path}}


def gen_write():
    path = random.choice(PATHS)
    content = random.choice(CONTENTS)
    prompts = [
        f"创建 {path}，内容为 {content}", f"把 {content} 写入 {path}",
        f"在 {path} 里写入：{content}", f"新建文件 {path} 并写入 {content}",
        f"向 {path} 写入内容 {content}", f"写一个文件 {path}：{content}",
        f"生成 {path}，内容是 {content}", f"覆盖 {path} 为 {content}",
        f"保存以下内容到 {path}：{content}", f"write {path} with {content}",
        f"将 {content} 保存为 {path}", f"创建并写入 {path}，内容 {content}",
    ]
    return random.choice(prompts), {"tool": "write", "params": {"path": path, "content": content}}


def gen_edit():
    path = random.choice(PATHS)
    old = random.choice(["DEBUG = False", "port = 8080", "import os", "VERSION = \"1.0.0\"",
                         "DEBUG=False", "LOG_LEVEL = \"info\"", "localhost:8000"])
    new = random.choice(["DEBUG = True", "port = 3000", "import os, sys", "VERSION = \"2.0.0\"",
                         "DEBUG=True", "LOG_LEVEL = \"debug\"", "localhost:3000"])
    prompts = [
        f"把 {path} 里的 {old} 替换为 {new}",
        f"在 {path} 中，将 {old} 改为 {new}",
        f"修改 {path}：{old} → {new}",
        f"编辑 {path}，把 {old} 改成 {new}",
        f"替换 {path} 中的 {old} 为 {new}",
        f"update {path}: change {old} to {new}",
    ]
    return random.choice(prompts), {"tool": "edit", "params": {"path": path, "old_string": old, "new_string": new}}


def gen_grep():
    term = random.choice(SEARCH_TERMS)
    path = random.choice(PATHS + [".", "/src", "/home/project", "/app"])
    prompts = [
        f"在 {path} 下搜索 {term}", f"搜索 {path} 目录下的 {term}",
        f"查找 {path} 中所有包含 {term} 的文件", f"grep {term} in {path}",
        f"在所有文件里找 {term}", f"找一下 {path} 里的 {term}",
        f"搜索关键词 {term}，路径 {path}", f"搜 {path} 下的 {term}",
        f"看看哪些文件里出现了 {term}", f"查找 {term} 这个词",
    ]
    return random.choice(prompts), {"tool": "grep", "params": {"pattern": term, "path": path}}


def gen_glob():
    patterns = ["src/**/*.py", "**/*.json", "tests/**/*.py", "**/*.txt",
                "app/**/*.html", "**/*.yml", "*.md", "**/*.csv"]
    pattern = random.choice(patterns)
    prompts = [
        f"列出所有 {pattern.split('/')[-1]} 文件", f"找 {pattern} 匹配的文件",
        f"列出 {pattern}", f"有哪些 {pattern}？", f"glob {pattern}",
        f"搜索文件：{pattern}", f"显示 {pattern} 文件列表",
        f"查找 {pattern} 模式的文件",
    ]
    return random.choice(prompts), {"tool": "glob", "params": {"pattern": pattern}}


def gen_bash():
    cmd = random.choice(CMDS)
    prompts = [
        f"执行 {cmd}", f"运行命令 {cmd}", f"在终端执行：{cmd}",
        f"跑一下 {cmd}", f"bash: {cmd}", f"运行 {cmd}",
        f"执行命令 {cmd}", f"帮我跑 {cmd}", f"run {cmd}",
        f"敲命令 {cmd}", f"执行：{cmd}",
    ]
    return random.choice(prompts), {"tool": "bash", "params": {"cmd": cmd}}


def gen_fetch():
    url = random.choice(URLS)
    prompts = [
        f"抓取 {url} 的内容", f"访问 {url}", f"获取 {url} 的数据",
        f"请求 {url}", f"fetch {url}", f"下载 {url} 的内容",
        f"帮我获取 {url}", f"从 {url} 拉取数据", f"GET {url}",
        f"去 {url} 拿数据", f"爬一下 {url}",
    ]
    return random.choice(prompts), {"tool": "fetch", "params": {"url": url}}


# ── Generate ──
GENERATORS = [gen_read, gen_write, gen_edit, gen_grep, gen_glob, gen_bash, gen_fetch]
PER_TOOL = 43

data = []
for gen_func in GENERATORS:
    for _ in range(PER_TOOL):
        prompt, output = gen_func()
        data.append({"instruction": prompt, "output": json.dumps(output, ensure_ascii=False)})

random.shuffle(data)

# Split: 250 train / 50 eval
train = data[:250]
eval_data = data[250:]

for name, subset in [("train", train), ("eval", eval_data)]:
    with open(f"../data/data_{name}.json", "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    print(f"  {name}: {len(subset)} samples → data/data_{name}.json")

# Quick stats
from collections import Counter
tc = Counter(json.loads(d["output"])["tool"] for d in data)
print(f"\n  Tool distribution: {dict(tc)}")
print(f"  Total: {len(data)} samples")
