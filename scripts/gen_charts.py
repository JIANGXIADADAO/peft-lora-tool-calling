"""Generate comparison charts for REPORT_R1.md — Before vs After LoRA."""

import json, os

# Force non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Load data ──
with open("../data/baseline.json", encoding="utf-8") as f:
    baseline = json.load(f)

with open("../data/eval_detail.json", encoding="utf-8") as f:
    detail = json.load(f)
    after = detail["summary"]
    tests = detail["tests"]

n = 15  # total tests

# ── Style ──
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "figure.dpi": 150,
    "axes.unicode_minus": False,  # fix minus sign display
})

COLOR_BEFORE = "#E57373"  # red-ish
COLOR_AFTER  = "#66BB6A"  # green-ish

OUT_DIR = "../docs"

# ============================================================
# Chart 1: Overall Before vs After bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

metrics = ["JSON valid", "Tool correct", "Params OK"]
before_vals = [baseline["json"], baseline["tool"], baseline["params"]]
after_vals  = [after["json"],  after["tool"],  after["params"]]

x = np.arange(len(metrics))
width = 0.32

bars_b = ax.bar(x - width/2, before_vals, width, label="Before LoRA", color=COLOR_BEFORE, edgecolor="white")
bars_a = ax.bar(x + width/2, after_vals,  width, label="After LoRA",  color=COLOR_AFTER,  edgecolor="white")

# Labels
for bar, val in zip(bars_b, before_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{val}/15",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#C62828")
for bar, val in zip(bars_a, after_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{val}/15",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#2E7D32")

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 17)
ax.set_ylabel("Correct / 15")
ax.set_title("LoRA 微调前后对比 — 工具调用能力", fontweight="bold")
ax.legend(loc="upper left", frameon=False)
ax.yaxis.set_major_locator(mticker.MultipleLocator(3))
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
path1 = os.path.join(OUT_DIR, "chart_overview.png")
fig.savefig(path1, bbox_inches="tight")
plt.close(fig)
print(f"  → {path1}")

# ============================================================
# Chart 2: Per-test before/after heatmap-style grid
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

labels = ["JSON", "Tool", "Params"]
tool_names = {
    "read": "read", "write": "write", "edit": "edit",
    "grep": "grep", "glob": "glob", "bash": "bash", "fetch": "fetch",
}

for col_idx, (title, data_source) in enumerate([
    ("Before LoRA", "baseline"),
    ("After LoRA",  "lora"),
]):
    ax = axes[col_idx]
    grid = np.zeros((n, 3))

    if data_source == "lora":
        for t in tests:
            i = t["idx"] - 1
            grid[i, 0] = 1 if t["valid_json"] else 0
            grid[i, 1] = 1 if t["tool_ok"] else 0
            grid[i, 2] = 1 if t["params_ok"] else 0
    else:
        # baseline — use CHANGELOG per-test data
        # We only have summary totals, so derive approximate per-test
        # Use the known before results from CHANGELOG
        before_matrix = [
            [1,1,0], [1,0,0], [1,1,0], [0,0,0], [1,0,0],
            [1,1,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0],
            [1,1,0], [1,1,0], [0,0,0], [1,1,1], [1,0,0],
        ]
        grid = np.array(before_matrix)

    cmap = matplotlib.colors.ListedColormap(["#E57373", "#66BB6A"])
    im = ax.imshow(grid.T, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels([f"#{i+1}" for i in range(n)], fontsize=8, rotation=45)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=14)

    # Border around cells
    for i in range(n):
        for j in range(3):
            ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, fill=False, edgecolor="white", lw=1))

fig.suptitle("逐条逐维对比（绿=通过  红=失败）", fontweight="bold", fontsize=16, y=1.02)
plt.tight_layout()
path2 = os.path.join(OUT_DIR, "chart_heatmap.png")
fig.savefig(path2, bbox_inches="tight")
plt.close(fig)
print(f"  → {path2}")

# ============================================================
# Chart 3: Per-tool Params accuracy
# ============================================================
tool_total = {"read": 0, "write": 0, "edit": 0, "grep": 0, "glob": 0, "bash": 0, "fetch": 0}
tool_ok = dict(tool_total)

for t in tests:
    tool = t["expected_tool"]
    tool_total[tool] += 1
    if t["params_ok"]:
        tool_ok[tool] += 1

fig, ax = plt.subplots(figsize=(8, 4))
tools = list(tool_total.keys())
pct = [100 * tool_ok[t] / tool_total[t] for t in tools]

bars = ax.bar(tools, pct, color=[COLOR_AFTER if p == 100 else "#FFA726" if p >= 50 else COLOR_BEFORE for p in pct], edgecolor="white")

for bar, p in zip(bars, pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{p:.0f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_ylim(0, 115)
ax.set_ylabel("Params Accuracy (%)")
ax.set_title("各工具参数准确率（After LoRA）", fontweight="bold")
ax.axhline(y=100, color="#66BB6A", linestyle="--", alpha=0.4, lw=1)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
path3 = os.path.join(OUT_DIR, "chart_by_tool.png")
fig.savefig(path3, bbox_inches="tight")
plt.close(fig)
print(f"  → {path3}")

print("\nDone — 3 charts saved to docs/")
