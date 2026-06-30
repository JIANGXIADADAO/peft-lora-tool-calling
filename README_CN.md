# PEFT — LoRA 微调实验

> Qwen2.5-0.5B + LoRA → 工具调用。CPU only，全程在自己电脑上跑。

[**English**](README.md) | [**📊 第一轮实验报告（R1）**](docs/REPORT_R1.md) | [**📝 学习日志**](docs/CHANGELOG.md)

---

## 这是什么？

一个 PEFT（Parameter-Efficient Fine-Tuning）从零到一的学习实验。用 LoRA 微调 0.5B 参数的小模型，让它学会根据中文指令输出正确的工具调用 JSON。

**核心问题**：大模型全量微调需要几十 GB 显存，个人开发者玩不起。LoRA 能不能用几百条数据、在 CPU 上就把模型从"不行"拉到"能行"？

**答案**：能。250 条数据 + 20 分钟训练 + 33.6MB 适配器 = 工具选择准确率从 40% → 100%。

---

## 项目结构

```
peft/
├── README.md              ← 英文入口
├── README_CN.md           ← 你在这里
├── scripts/               ← 所有脚本
│   ├── gen_data.py        ← 生成训练数据
│   ├── train.py           ← LoRA 微调训练
│   ├── test_baseline.py   ← 训练前基线测试
│   ├── eval_lora.py       ← 训练后评估对比
│   └── gen_charts.py      ← 生成可视化图表
├── data/                  ← 数据文件
│   ├── data_train.json    ← 训练集（250条）
│   ├── data_eval.json     ← 验证集（51条）
│   ├── baseline.json      ← 训练前基线结果
│   └── eval_detail.json   ← 训练后逐条评估
├── docs/                  ← 文档
│   ├── REPORT_R1.md       ← 第一轮实验报告 ★
│   ├── CHANGELOG.md       ← 完整学习日志
│   └── chart_*.png        ← 可视化图表
├── output/                ← 训练产物（gitignore）
│   └── lora_tool_calling/ ← LoRA 适配器权重（33.6MB）
├── models/                ← 基座模型（gitignore，需自行下载）
└── venv/                  ← Python 虚拟环境（gitignore）
```

---

## 快速开始

```bash
# 1. 激活环境
source venv/Scripts/activate

# 2. 下载模型（从 ModelScope）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='./models')"

# 3. 跑基线测试
cd scripts && python test_baseline.py

# 4. 训练
python train.py

# 5. 评估
python eval_lora.py
```

---

## 实验摘要（R1）

| 维度 | Before LoRA | After LoRA | 提升 |
|------|------------|------------|------|
| JSON 合法 | 87% | **100%** | +2 |
| 工具正确 | 40% | **100%** | +9 |
| 参数全对 | 7% | **73%** | +10 |

[→ 完整报告](docs/REPORT_R1.md)

---

## 技术栈

- **基座模型**: Qwen2.5-0.5B-Instruct（954MB, 国产最小 instruct 模型）
- **微调方法**: LoRA (r=16, alpha=32)
- **框架**: PyTorch 2.12 + transformers 5.12 + PEFT 0.19
- **硬件**: CPU only, 2 线程限制
