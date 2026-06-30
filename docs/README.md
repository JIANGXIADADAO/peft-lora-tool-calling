# PEFT（参数高效微调）

> 目标：用 GLM-4-9B + QLoRA 在 RTX 4060 (8GB) 上完成一次完整微调实验。

## 硬件

- GPU：NVIDIA GeForce RTX 4060 Laptop (8GB)
- 方案：QLoRA（4-bit 量化 + LoRA），别无选择

## 模型

- GLM-4-9B（MIT 协议，智谱同源）
- 8GB 显存刚好跑 QLoRA

## 待定

- 微调任务（Agent 工具调用？代码生成？指令跟随？）
- 数据集（自建 or 开源）
- 评估方式
