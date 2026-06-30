# PEFT — LoRA Fine-Tuning Experiment

> Qwen2.5-0.5B + LoRA → Tool-calling. CPU only, runs on a laptop.

[**中文版本**](README_CN.md) | [**📊 R1 Report (Chinese)**](docs/REPORT_R1.md) | [**📝 Learning Journal (Chinese)**](docs/CHANGELOG.md)

---

## What is this?

A from-scratch PEFT learning experiment. Fine-tune a 0.5B-parameter model with LoRA to output correct tool-calling JSON from Chinese instructions.

**Core question**: Full fine-tuning needs dozens of GB of VRAM. Can LoRA, with a few hundred examples on CPU, lift a model from "broken" to "working"?

**Answer**: Yes. 250 samples + 20 min training + 33.6MB adapter = tool selection accuracy 40% → 100%.

---

## Project Structure

```
peft/
├── README.md
├── README_CN.md
├── scripts/          ← All scripts
├── data/             ← Training & eval data
├── docs/             ← Reports & charts
├── output/           ← Trained LoRA weights (33.6MB, gitignored)
├── models/           ← Base model (gitignored)
└── venv/             ← Python virtualenv (gitignored)
```

---

## Quick Start

```bash
source venv/Scripts/activate
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='./models')"
cd scripts
python test_baseline.py   # Before
python train.py           # Train (~20 min, CPU)
python eval_lora.py       # After
```

---

## R1 Results at a Glance

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| JSON valid | 87% | **100%** | +2 |
| Tool correct | 40% | **100%** | +9 |
| Params OK | 7% | **73%** | +10 |

[→ Full Report (Chinese)](docs/REPORT_R1.md)

---

## Tech Stack

- **Base model**: Qwen2.5-0.5B-Instruct (954MB)
- **Method**: LoRA (r=16, alpha=32)
- **Frameworks**: PyTorch 2.12 + transformers 5.12 + PEFT 0.19
- **Hardware**: CPU only, 2-thread limited
