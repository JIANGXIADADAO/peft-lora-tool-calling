# PEFT 学习日志

> 边做边学。记录困惑、解答、关键决策。

---

## 2026-07-01 — 概念奠基

### Q1：PEFT 是什么？

大模型全量微调要更新全部参数（90 亿个），显存需求几十 GB，个人做不到。

PEFT（Parameter-Efficient Fine-Tuning）只改极少量参数——在原模型旁边插入小矩阵，冻结原模型，只训小矩阵。最终产物是一个几 MB 的权重文件。

**类比**：不改整本书，只贴几张便签纸。

### Q2：LoRA 和 QLoRA 的区别？

**LoRA**：把微调更新量 ΔW 分解为 A×B 两个窄矩阵（r=16），只训 A 和 B。原模型保持 16-bit。

**QLoRA**：LoRA + 把原模型从 16-bit 压到 4-bit。A 和 B 仍然是 16-bit 高精度训练。

区别只有一个参数：原模型用什么精度存。数学上完全一样。

| | LoRA | QLoRA |
|---|---|---|
| 原模型精度 | 16-bit | 4-bit |
| 训练精度 | 16-bit | 16-bit |
| 显存 (9B) | ~16GB | **~8GB** |
| 效果 | 略好 | 几乎持平 |

RTX 4060 8GB 只能用 QLoRA。

### Q3：为什么 ΔW = A×B？为什么是两个矩阵？

微调不需要更新全部 1600 万个独立方向。r=16 个旋钮就够覆盖大部分行为变化。

两个矩阵才能"借"尺寸：A(4096×16) 把 4096 维压到 16 维，B(16×4096) 再解压回 4096 维。一个矩阵做不到——要么只能压，要么只能解，不能同时。

**类比**：调一台通用机床去专做螺丝，你只需要调几个旋钮，不需要重造机床。A×B 就是那几个旋钮的参数化。

### Q4：r 的大小意味着什么？

r = 瓶颈维度 = 你能调的自由度。

| r | 可训参数 | 适用场景 |
|---|---------|---------|
| 8 | 6.5 万 | 简单任务 |
| 16 | 13 万 | 大多数任务（默认） |
| 64 | 52 万 | 复杂多任务 |
| 4096（全量） | 1600 万 | 没钱别想 |

r 太小 → 表达能力不够。r 太大 → 失去了 PEFT 的意义。实践中 r=16 基本够用。

### Q5：QLoRA 训练精度差，是不是需要更小的 r？

不用。QLoRA 压的是冻结的原模型 W（4-bit），你训练的 A×B 仍是 16-bit 高精度。论文对比显示同 r 下 QLoRA 和 LoRA 效果几乎持平。

### Q6：训练产物是什么？

一个几 MB 的 LoRA 权重文件（A 和 B 的值）。基座模型不变，用时加载不同的 LoRA 权重实现不同能力。10 个任务 = 10 个几 MB 的文件，不是 10 个 18GB 的模型。

---

## 2026-07-01 — 环境搭建

### 硬件

- GPU：NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
- 方案选择：CPU 训练 + 小模型（0.5B），本地跑通流程
- 放弃原因：CUDA 版 PyTorch 下载屡次 hash 校验失败，清华镜像只有 CPU 版

### 模型

- Qwen2.5-0.5B-Instruct（~1GB）
- 下载渠道：ModelScope（HuggingFace 被墙，hf-mirror 不稳定）
- 本地路径：`./models/Qwen/Qwen2.5-0.5B-Instruct`

### 关键决策：为什么用小模型而不是大模型？

大模型（9B+）本身能力太强，LoRA 微调效果会被淹没在基线能力里。
0.5B 模型在工具调用上本身很弱，LoRA 能把它从"不行"拉到"能行"，
before/after 对比才真正有说服力。

### 包依赖

```
torch==2.12.1+cpu
transformers==5.12.1
peft==0.19.1
datasets==5.0.0
accelerate==1.14.0
safetensors==0.8.0
modelscope==1.38.0
```

---

## 2026-07-01 — Baseline 测试

### 测试脚本：test_baseline.py

**脚本逻辑**：
1. 加载 Qwen2.5-0.5B-Instruct（CPU，torch_dtype=auto）
2. System prompt：列出 7 个工具及参数，要求只输出 JSON，不解释
3. 15 条中文测试指令，每条期望一个工具调用 JSON
4. 推理参数：max_new_tokens=100, do_sample=False（贪心解码，确保可复现）
5. 评分三步：JSON 合法性 → 工具名匹配 → 参数键值逐一比对
6. JSON 解析容错：去除 markdown 代码块标记、正则兜底提取 `{...}`

**评价三维度**：
- JSON 合法：输出能否被 json.loads 解析（含正则兜底）
- 工具正确：`parsed["tool"]` == 期望工具名
- 参数全对：所有期望参数的键和值完全匹配

### 实验设计

- 任务：Agent 工具调用格式生成（7 工具：read/write/edit/grep/glob/bash/fetch）
- 测试集：15 条中文指令 → 期望 JSON 输出，每个工具 2-3 条

### Before LoRA 结果

```
#    Tool   JSON   ToolOK   Params   Raw
--------------------------------------------------------------------------------
1    read   ✓      ✓        ✗        {"tool":"read","params":["/home/project/README.md"
2    write  ✓      ✗        ✗        {"tool":"edit","path":"/tmp/hello.py","content":"p
3    edit   ✓      ✓        ✗        {"tool":"edit","params":{"path":"/app/config.py","
4    grep   ✗      ✗        ✗        {"tool":"grep","params":["TODO.*\.py"],"url":"http
5    glob   ✓      ✗        ✗        {"tool": "grep", "params": ["-r", "*.py", "src"]}
6    bash   ✓      ✓        ✗        {"tool": "bash", "cmd": "pip install requests"}
7    fetch  ✓      ✗        ✗        {"tool":"grep","params":["github.*?content"],"url"
8    read   ✓      ✗        ✗        {"tool":"grep","params":["/etc/hosts"],"cmd":"cat"
9    write  ✓      ✗        ✗        {"tool": "bash", "params": ["cat /tmp/test.py | py"
10   bash   ✓      ✗        ✗        {"tool":"grep","params":["/var/log","ls -la"]}
11   grep   ✓      ✓        ✗        {"tool":"grep","params":["/src/import os"],"file":
12   edit   ✓      ✓        ✗        {"tool":"edit","params":{"old_string":"/app/main.p
13   glob   ✗      ✗        ✗        {"tool":"grep","params":[".*\.json$"],"url":"https
14   fetch  ✓      ✓        ✓        {"tool":"fetch","params":{"url":"http://example.co
15   read   ✓      ✗        ✗        {"tool":"grep","params":["/home/user/.bashrc"],"co

===== BEFORE LORA =====
  JSON valid:   13/15 (87%)
  Tool correct: 6/15 (40%)
  Params OK:    1/15 (7%)
  → baseline.json
```

### 逐条分析

| # | 指令 | 期望工具 | 实际输出 | JSON | 工具 | 参数 | 问题 |
|---|------|---------|---------|------|------|------|------|
| 1 | 读取 README.md | read | `{"tool":"read","params":["/home/..."]}` | ✓ | ✓ | ✗ | params 是数组不是对象 |
| 2 | 创建 hello.py | write | `{"tool":"edit","path":"/tmp/hello.py"...}` | ✓ | ✗ | ✗ | 选了 edit 而非 write，params 缺 content 键 |
| 3 | 改 DEBUG=False→True | edit | `{"tool":"edit","params":{"path":"/app/config.py"...}` | ✓ | ✓ | ✗ | 工具对，但 old_string 被截断 |
| 4 | 搜 .py 里的 TODO | grep | `{"tool":"grep","params":["TODO.*\.py"],"url":"http..."}` | ✗ | ✗ | ✗ | params 数组 + 多了 url 字段，JSON 未闭合 |
| 5 | 列 src/ 下 .py | glob | `{"tool":"grep","params":["-r","*.py","src"]}` | ✓ | ✗ | ✗ | 用 grep -r 代替 glob |
| 6 | pip install requests | bash | `{"tool":"bash","cmd":"pip install requests"}` | ✓ | ✓ | ✗ | 工具对，但 params 是 `cmd` 而非 `{"cmd":"..."}` 对象 |
| 7 | 抓取 github API | fetch | `{"tool":"grep","params":["github.*?content"],"url":"..."}` | ✓ | ✗ | ✗ | 用 grep 正则代替 fetch |
| 8 | 打开 /etc/hosts | read | `{"tool":"grep","params":["/etc/hosts"],"cmd":"cat"}` | ✓ | ✗ | ✗ | grep + cat 代替 read |
| 9 | 写入 /tmp/test.py | write | `{"tool":"bash","params":["cat /tmp/test.py \| py..."]}` | ✓ | ✗ | ✗ | 用 bash 管道代替 write |
| 10 | ls -la /var/log | bash | `{"tool":"grep","params":["/var/log","ls -la"]}` | ✓ | ✗ | ✗ | grep 代替 bash |
| 11 | 搜 /src 下 import os | grep | `{"tool":"grep","params":["/src/import os"],"file":"..."}` | ✓ | ✓ | ✗ | 工具对，但 pattern 和 path 拼在一起 |
| 12 | port=8080→3000 | edit | `{"tool":"edit","params":{"old_string":"/app/main.py"...}}` | ✓ | ✓ | ✗ | path 写进了 old_string |
| 13 | 找所有 .json | glob | `{"tool":"grep","params":[".*\.json$"],"url":"https..."}` | ✗ | ✗ | ✗ | grep 正则 + url 字段 |
| 14 | 请求 example.com/api | fetch | `{"tool":"fetch","params":{"url":"http://example.co..."}}` | ✓ | ✓ | ✓ | **唯一全对** |
| 15 | 读 ~/.bashrc | read | `{"tool":"grep","params":["/home/user/.bashrc"],"co..."}` | ✓ | ✗ | ✗ | grep + co 字段代替 read |

### 模式总结

- **15 条中 10 条选错工具**，核心模式是"一切皆 grep"——模型把任何检索/读取类操作都当成 grep
- **就算工具选对，params 格式也大概率错**——数组 vs 对象混淆、缺少必需字段、添加不存在的字段
- **14/15 条失败，1/15 全对（fetch 简单 URL）**

核心问题：0.5B 模型学了 JSON 外壳，但对"什么场景用哪个工具、参数该怎么填"完全没概念。LoRA 要做的就是把这个语义映射教会。

### 下一步

造训练数据：7 个工具 × ~45 条 = ~300 条 (instruction → JSON) 对。

---

## 2026-07-01 — 训练数据生成

### 数据设计

- 脚本：`gen_data.py`
- 7 个工具，每个 ~43 条，总计 301 条
- 切分：train 250 / eval 51
- 输出格式：`{"instruction": "中文指令", "output": "{\"tool\":\"...\",\"params\":{...}}"}`

### 工具分布

| 工具 | 数量 | 参数 |
|------|------|------|
| read | 43 | path |
| write | 43 | path, content |
| edit | 43 | path, old_string, new_string |
| grep | 43 | pattern, path |
| glob | 43 | pattern |
| bash | 43 | cmd |
| fetch | 43 | url |

### 数据多样性

- 路径池：20 个不同路径
- 内容池：16 种不同写入内容
- 搜索词池：12 个不同 pattern
- URL 池：6 个不同地址
- 命令池：12 条不同 shell 命令
- 每条指令有 6-14 种不同中文表述变体

### 文件

- `data_train.json` — 250 条训练集
- `data_eval.json` — 51 条验证集

---

## 2026-07-01 — CPU 安全防护与模型决策

### 问题：训练导致电脑卡死

运行 train.py 后整个系统失去响应。排查后发现根因不是内存不足（0.5B 模型仅 ~1GB），而是 **CPU 16 线程全被训练吃满**。

### 解决方案

1. **`torch.set_num_threads(2)`** — 训练只占 2 个核心，系统保持响应
2. **`OMP_NUM_THREADS=2` + `MKL_NUM_THREADS=2`** — 限制底层 BLAS 线程
3. **`max_length: 512 → 256`** — 减半序列长度，降低计算量
4. **`batch_size: 2 → 1`** — 减小批次
5. **`epochs: 5 → 3`** — 减少轮数
6. **`max_steps: 200`** — 硬上限，防止无限训练
7. **`torch_dtype: bfloat16 → float32`** — CPU 不支持 bfloat16 原生运算，用 float32 反而更快

### 模型选型结论

国产开源模型中，**0.5B 已经是最小的指令微调规格**（Qwen 系列、腾讯混元系列均以此起步）。不存在更小的国产 instruct 模型。唯一更小的选择是 SmolLM2-135M（英文），但下载不稳定且中文能力存疑。

最终方案：**Qwen2.5-0.5B-Instruct + CPU 安全防护 = 不卡死 + 流程跑通**。

### Baseline 复测（CPU 安全模式）

重新下载模型后跑 baseline，结果与原记录一致，确认环境正常：

```
JSON valid:   13/15 (87%)
Tool correct: 6/15 (40%)
Params OK:    1/15 (7%)
```

模型加载正常，2 线程限制生效，系统响应流畅。

---

## 2026-07-01 — 训练运行

### 训练配置（CPU 安全模式）

| 参数 | 值 |
|------|-----|
| 模型 | Qwen2.5-0.5B-Instruct |
| LoRA r | 16 |
| LoRA alpha | 32 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| max_length | 256 |
| batch_size | 1 |
| gradient_accumulation | 4（等效 batch=4） |
| epochs | 3 |
| max_steps | 200 |
| lr | 2e-4 |
| CPU 线程 | 2（torch + OMP + MKL 全部限制） |
| dtype | float32 |

### 训练结果

```
train_loss:          0.3603
eval_loss:           0.168
耗时:                20 分 37 秒
步数:                200/200（撞到 max_steps 上限）
实际 epoch:          3.176
可训参数:            8,798,208 / 502,830,976 (1.75%)
train samples/s:     0.647
```

### 产物

```
lora_tool_calling/
├── adapter_model.safetensors   33.6 MB   ← LoRA 权重 A×B
├── adapter_config.json          1.1 KB   ← r=16, alpha=32 配置
├── checkpoint-150/                        ← 中间检查点
├── checkpoint-200/                        ← 最终检查点
├── tokenizer.json              10.9 MB
└── tokenizer_config.json        723 B
```

### 观察

- **eval_loss 持续下降**：0.168 很低，模型对训练集格式已充分拟合
- **训练稳定**：无 spike、无 NaN，loss 曲线平滑
- **系统流畅**：2 线程限制生效，20 分钟训练期间电脑正常使用
- **max_steps 命中**：200 步提前停止，实际跑了 3.176 个 epoch（数据过了 3 遍多）
- **LoRA 权重仅 33.6MB**：对比基座模型 954MB，只多出 3.5%

### 深入理解：这 33.6MB 本质上是什么？

不是语言知识，不是词汇语法——那些全在冻结的基座模型里。这 33.6MB 存的是 **一个高维空间里的方向向量**。

**物理构成**：8,798,208 个 float32 数字，按层组织为 A×B 矩阵对：

```
每层 7 个模块 × 24 层 = 168 个矩阵对
每个矩阵对 = A(d_in×16) + B(16×d_out) 两个小矩阵
```

**几何直觉**：想象 880 万维空间。训练前，模型在原点上（无偏转）。每条训练数据都是"往这儿走一小步"。200 步后，模型走到了空间里某个位置——这个位置的坐标就是 33.6MB 文件的内容。

**函数直觉**：基座模型 = `f(x)`，LoRA 适配器 = `Δf(x)`。推理时的实际输出 = `f(x) + Δf(x)`。Δf 只在特定输入上有显著响应——那些和工具调用相关的输入。对无关输入（比如闲聊），Δf 几乎为 0，模型表现和原版一样。

**类比**：基座模型是一辆通用汽车，LoRA 权重是方向盘上贴的一张便签——"看到'读取文件'选 read，看到'执行命令'选 bash，参数放对象里别放数组里"。换一个任务（翻译、摘要），便签内容完全不同，但车还是那辆车。

**可组合性**：这就是 PEFT 的核心价值。10 个任务 = 10 张便签（10 × 33.6MB），而不是 10 辆新车（10 × 954MB）。

---

## 2026-07-01 — 评估结果

### 总体对比

```
              Before LoRA    After LoRA     提升
─────────────────────────────────────────────────
JSON valid     13/15 (87%)   15/15 (100%)   +2
Tool correct    6/15 (40%)   15/15 (100%)   +9
Params OK       1/15 ( 7%)   11/15 ( 73%)   +10
```

**核心成就**："一切皆 grep"被完全根治。15 个工具全部选对，JSON 格式 100% 合法。

### 全部通过的用例（11/15）

| # | 指令 | 工具 | 输出 |
|---|------|------|------|
| 1 | 读取 /home/project/README.md | read | `{"path":"/home/project/README.md"}` |
| 3 | 把 DEBUG=False 改成 DEBUG=True | edit | `{"path":"/app/config.py","old_string":"DEBUG=False","new_string":"DEBUG=True"}` |
| 6 | 运行 pip install requests | bash | `{"cmd":"pip install requests"}` |
| 7 | 抓取 github API | fetch | `{"url":"https://api.github.com"}` |
| 8 | 打开 /etc/hosts | read | `{"path":"/etc/hosts"}` |
| 9 | 写入 import sys | write | `{"path":"/tmp/test.py","content":"import sys"}` |
| 10 | ls -la /var/log | bash | `{"cmd":"ls -la /var/log"}` |
| 11 | 搜 import os | grep | `{"pattern":"import os","path":"/src"}` |
| 12 | port=8080→3000 | edit | `{"path":"/app/main.py","old_string":"port=8080","new_string":"port=3000"}` |
| 14 | 请求 example.com | fetch | `{"url":"http://example.com/api/data"}` |
| 15 | 读 ~/.bashrc | read | `{"path":"~/.bashrc"}` |

### 参数瑕疵分析（4/15）

| # | 工具 | 期望 | 实际 | 根因 |
|---|------|------|------|------|
| 2 | write | `content: "print('hello')"` | `content: "print(\"hello\")"` | 单引号→双引号，语义等价，字符级匹配失败 |
| 4 | grep | `path: "."` | `path: "/src"` | 指令未指定路径，模型自行脑补 |
| 5 | glob | `pattern: "src/**/*.py"` | `pattern: "src/"` + 多了个 `path` 字段 | `**/` 递归匹配写不全；混入了 grep 才有的 `path` 参数 |
| 13 | glob | `pattern: "**/*.json"` | `pattern: "*.json"` + 多了个 `path: "/src"` | 同上：glob 与 grep 参数结构混淆 |

**模式**：glob 是唯一有问题的工具（2 条全错参数）。模型没完全掌握 glob 的参数 schema——`**/` 递归模式学得不牢，且会混入 grep 的 `path` 参数。这跟训练数据中 glob 的 43 条样本覆盖面有关：`**/*.json` 这类递归 glob 的变体可能不够多。

### 下一步

可以在 `gen_data.py` 中增加 glob 的多样化 pattern（尤其是 `**/` 前缀和不同扩展名组合），重新生成数据后再训一轮。