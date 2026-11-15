# MIRAGE 绘画模型技术说明

本文介绍 模型的结构、训练与推理设计、Triton 执行优化，以及目前的问题。
面向希望理解代码或继续开发的读者。参数、公式和执行方式以本仓库实现为准。

这是一个约 **1.651B 参数**的二次元 Tag 条件绘画模型。使用预训练 Qwen 文本 Hidden States 和
FLUX.2 VAE 表示，在 latent 空间进行 Rectified Flow 速度预测。

## 1 快速了解使用要求

| 项目 | 要求与当前状态 |
|---|---|
| 提示词 | 使用逗号分隔的 Tag|
| 首个 Tag | 明确写 `sfw` 或 `nsfw`，不要省略|
| 分级含义 | 项目所用数据分级规则对 NSFW 的判定偏严格|
| 自然语言 | 未进行系统效果测试|
| Register | 当前已训练权重必须保留 **8 个文本侧可学习 Register**，由模型自动插入序列首部 |
| 分辨率 | 支持上限为768×1024 / 1024×768及不超过同等面积、宽高均为32倍数的尺寸 |
| 采样 | 默认 Euler 40 步，CFG 默认 4 |
| 当前不足 | 手臂、腿、手指和全身姿态容易出错，复杂构图也不稳定 |

推荐输入格式示例：

```text
sfw, 1girl, blue_hair, full_body, standing, white_shirt
```

`sfw/nsfw` 是参与文本编码的内容等级 Tag，不是程序执行的审核开关。
当前入口不会自动补上或校验这个前缀，使用者应主动提供；默认示例 Prompt 也不能代替此要求。
训练数据的 NSFW 分级偏严格，不表示模型能保证过滤某类输出，也不应把推理时的 Tag 理解成可靠分类器。

## 2 模型数据流

```text
逗号分隔的 Tags
  → 冻结的 Qwen3-Embedding-0.6B
  → 全部有效 Token Hidden States [B,L,1024]
  → 紧凑打包 + 每条样本首部插入 8 个可学习 Register
  → 2 层双向 TextAdapter，宽度 1024
  → 作为 20 个主体 Block 的文本 K/V 来源

训练图片
  → 冻结的 FLUX.2 VAE
  → 官方 latent 打包与归一化
  → [B,128,H/16,W/16]
  → RF 加噪
  → 展平成图像 Token + Linear(128,2048)
  → 20 个 HybridBlock，含 AttnRes 历史读取
  → 最终 AttnRes 读取 + RMSNorm + Linear(2048,128)
  → 预测同形状的 latent 速度
```

图中 `[B,L,1024]` 是便于理解的逻辑表示。主体执行时使用 `[总有效Token数,1024]` 的紧凑布局，
每条样本的边界由累积偏移记录，不让不同样本的文本相互注意。
Qwen 编码阶段为批处理临时做 Padding，并通过有效 Mask 排除；

### 2.1 主要配置

| 配置 | 默认值 |
|---|---:|
| 可训练参数量 | 1,650,703,936 |
| 主体深度 `num_blocks` | 20 |
| 图像宽度 | 2048 |
| 主体 FFN 隐藏宽度 | 8192，SwiGLU |
| 图像 Self-Attention | 16 Q Heads、16 KV Heads、HeadDim128 |
| 图文 Cross-Attention | 16 Q Heads、16 KV Heads、HeadDim128 |
| Qwen 输入特征宽度 | 1024 |
| TextAdapter | 2 层，16 Heads、HeadDim64、FFN4096 |
| 文本 Register | 8 个，每个宽度1024 |
| Latent | 128通道，有效空间步长16 |

### 2.2 支持分辨率的含义

本项目最大支持像素是 `768 × 1024 = 786432` 像素面积。
可使用相同像素预算内的其他长宽比，
但宽高必须分别为正的32倍数：

```text
W % 32 == 0
H % 32 == 0
W * H <= 786432
```

极端分辨率图片样本不足，效果不好。

在有效空间步长16下，图像Token数为 `(W/16) × (H/16)`，面积上限对应3072个图像Token。
同面积图像的Token数相同，但二维RoPE坐标和构图分布不同。


## 3 图像 Self 加文本 Cross

参考ANIMA实现。
每个主体 Block 按顺序包含：

1. 图像 Self-Attention：图像位置之间交换构图、几何和视觉信息。
2. 文本 Cross-Attention：图像 Token 作 Query，读取文本 Token 的 Key/Value。
3. 图像 SwiGLU FFN：对图像通道特征进行非线性变换。

每个分支有自己的归一化与 AttnRes 读取。图像 Self-Attention 使用二维 RoPE，
Cross-Attention 输出在输出投影前应用逐图像位置、逐 Head 的 Sigmoid Gate。

这里讨论的“全自注意力”是把图像和文本拼接后做 Joint Self-Attention，参考MOSS实现。
设图像有 N 个 Token，文本有 L 个 Token，

| 方案 | 主体每层 Attention 成对交互规模 |
|---|---|
| 拼接后 Joint Self-Attention | `(N+L)^2 = N^2 + 2NL + L^2` |
| 图像 Self + 图像读取文本 Cross | `N^2 + NL` |

后者不在每个主体层计算文本 Query 读取全部图像，也不反复计算主体内的文本自注意力，
主体 FFN 同样只处理图像。文本的双向交互集中在前面两层、较窄的 TextAdapter 中完成。
这使长文本训练成本更可控，也使推理时整段文本状态和每层 K/V 能跨采样步缓存。

项目选择它是为了训练预算的平衡，全自注意力成本较高。

## 4 文本使用Qwen3-Embedding-0.6B

本项目读取的是Qwen3-Embedding-0.6B `last_hidden_state` 的全部有效 Token，参考ANIMA。

TextAdapter 使用双向 Pre-RMSNorm Attention 和 SwiGLU，输出投影以较小标准差初始化，
让新学习的文本处理从相对温和的变化开始。

### 4.1 Tag 格式与自然语言的边界

数据准备使用 Tag 列表，去掉引号并按逗号分隔的 Tag 单位打乱顺序，然后重新编码，同tag连接词仍需要下划线。

推理请使用同类 Tag 格式，首项是 `sfw` 或 `nsfw`。
**自然语言长句没有完成系统测试**

## 5 为什么加入 Gate 和 Register

训练时需要文本注意力过度集中于首token以及逗号、引号等tag相关性较小的位置，因此采用gate和register token。

### 5.1 Gate 控制写入量

对图像位置 j、Attention Head h，文本读取近似为：

```text
a[j,h] = Attention(normalize(Q[j,h]), normalize(K), V)
g[j,h] = sigmoid(W_gate * X[j] + b_gate)
y[j,h] = g[j,h] * a[j,h]
output[j] = W_out * concatenate_heads(y[j])
```

Gate 让不同图像位置和不同 Head 可以控制文本分支的写入强度。


### 5.2 Register 提供额外可学习位置

标准 Softmax 的权重和为1，模型在不需要具体某个词时，也必须把注意力分给某些位置。
加入8个可学习 Register，是给模型提供额外的工作与信息汇聚位置，减少语义 Token 被迫承担这类角色。

Register 放在 **TextAdapter之前、每条文本序列首部**，经过两层双向更新后与完整文本一同成为主体的K/V。

### 5.3 必须保留8个Register

当前权重是在8个Register条件下训练的。它们存在于模型参数 `text_adapter.register_tokens` 中，
由模型自动插入。

## 6 AttnRes 

采用注意力残差机制改善残差连接。

## 7 RF 目标与精度

本仓库当前使用直接速度预测：

```text
t = sample_uniform_or_logit_normal_per_image()
noise = randn_like(z)
x_t = (1-t) * z + t * noise
target = noise - z
loss = mean((prediction.float() - target.float())**2)
```

**当前 t=0 为干净latent，t=1 为纯噪声**。推理从1积分到0，Euler的dt为负。
训练使用uniform timestep采样。

FLUX.2 latent按官方运行均值/方差做归一化，反向解码必须通过对应Codec还原。
主体主要BF16，Loss、积分状态和FP32主权重等保持原精度边界，VAE接口采用FP32。


### 7.1 分辨率课程与各阶段步数

本表针对本次公开权重所在训练链，截至 **540000次优化器更新**。
`step`是实际参数更新次数。


| 阶段 | global_step区间 | 本阶段更新次数 | 有效Batch | 常用micro-batch | 常规累积次数 | 学习率 |
|---|---|---:|---:|---:|---:|---:|
| 256面积档 | 1～106000 | 106000 | 256 | 32 | 8 | 1e-4 |
| 512面积档 | 106001～212000 | 106000 | 64 | 16 | 4 | 1e-4 |
| 原尺寸档，最高768×1024等面积 | 212001～540000 | 328000 | 16 | 4 | 4 | 5e-5 |
| 合计 | 1～540000 | **540000** | 随阶段变化 | — | — | — |

“256/512面积档”保持原图长宽比，
以接近256²或512²的面积准备latent，并使实际编码宽高对齐32。

### 7.2 优化器与时间采样的历史切换

这条训练链不是从第0步就使用Muon：

| 更新区间 | 优化器 | 时间采样 |
|---|---|---|
| 1～126000 | Muon＋AdamW | LogitNormal |
| 126001～540000 | Muon＋AdamW | Uniform |

前期用logitNormal加速收敛，后面采用uniform加强对纯噪音以及接近无噪音时模型推理能力。

## 8 使用 Muon 加 AdamW

参考MOSS。

| 参数类型 | 本实现的处理 |
|---|---|
| 符合规则的二维Linear权重 | Muon |
| 融合QKV、KV、SwiGLU权重 | 先按逻辑矩阵切分，再分别做Muon更新 |
| Bias、Norm、Register及被排除的控制/输入输出权重 | AdamW，按规则分配weight decay |
| 模型训练副本 | 主要BF16 |
| 主权重、动量和Adam状态 | 保留FP32 |

## 9 为什么写 Triton

因为实际训练成本：不同样本Tag长度不同，Padding浪费显存和计算；部分固定形状编译路径又会因packed行数改变反复编译。
此外，Norm、RoPE、Gate、历史读取、精度转换等许多小操作会生成大量中间张量，


| 优化方向 | 实现思路 | 降低的成本 |
|---|---|---|
| 变长文本 | 有效Token紧凑排列，用累积偏移记录边界 | Padding激活、无效计算 |
| 紧凑任务表 | 只为实际存在的序列分块派发任务 | 短文本的空任务块 |
| 运行时长度 | 主Attention长度不作为逐精确值的编译特化参数 | 无谓的长度重编译 |
| QKV/Norm/RoPE融合 | 同一Kernel完成相邻布局和归一化操作 | 中间张量与launch |
| Cross＋Gate融合 | 写出Attention结果时结合门控，并实现对应反向 | 额外特征读写 |
| AttnRes融合 | 历史评分/读取、Norm和调制衔接，复用历史RMS | 历史重复读取与归约 |
| 历史梯度合并 | 多读取分支的历史梯度协作累加 | 反复分配与相加 |
| RF与梯度处理 | 融合插值/MSE、梯度转换与范数归约 | 临时FP32张量与调度 |
| 推理文本缓存 | TextAdapter及每层K/V只计算一次 | 40步重复文本计算 |


## 10 与 FA4 和 PyTorch Flash 对比



### 10.1 PRO6000训练实测

RTX PRO 6000 Blackwell Server Edition，SM120；Torch 2.12.1+cu130
相同checkpoint、原始1024×768、有效B16=micro4×4、主体检查点0、固定256有效文本Token＋8Register，
计时包括打包/H2D/完整前反向/Muon＋AdamW更新。

| 方案 | 秒/更新，越低越快 | 已分配峰值GiB |
|---|---:|---:|
| MIRAGE默认训练执行方案 | 2.7412 | 53.774 |
| 全Attention用FA4 | 2.8397 | 51.889 |
| 默认Attention＋GEMM自动调优 | 2.7350 | 53.773 |
| FA4＋GEMM 调优 | 2.8281 | 51.891 |

全FA4在这组测试中约慢3.59%，但省约1.89GiB已分配显存。
候选后端当时的固定形状GEMM路径会因变长文本行数不同产生新编译，出现几十秒慢步；

### 10.2 RTX3090推理实测

RTX3090 24GB / SM86，Torch2.8.0+cu128、Triton3.4.0、FA4 4.0.0b38
Euler40、CFG4、B1、768×1024，预热后仅计主体采样：

| 方案 | 秒，越低越快 |
|---|---:|
| PyTorch自动路径，内置Flash已启用 | 51.697 |
| 原训练Triton前向 | 15.691 |
| 推理混合路径 | 15.288 |
| Triton融合＋FA4核心 | 15.391 |

主要提升来自启用一整套融合执行。
单层测试中图像Self-Attention是PyTorch Flash更快；短文本Cross＋Gate是自写融合核更快。

### 10.3 Attention 单层前向

单位毫秒，16Heads、HeadDim128、BF16；不含投影、Norm。
Cross＋Gate包含门控：自写路径内融合，其他路径后接同一个Triton Gate。

| 场景 | 自写 Triton | PyTorch Flash | FA4 |
|---|---:|---:|---:|
| Self 1024 | 0.21065 | 0.18614 | 0.18944 |
| Self 3072 | 1.51082 | 1.14361 | 1.16575 |
| Cross 3072/64 | 0.04899 | 0.04725 | 0.06085 |
| Cross＋Gate 3072/64 | 0.05258 | 0.08634 | 0.09919 |
| Cross 3072/256 | 0.12763 | 0.11555 | 0.12801 |
| Cross＋Gate 3072/256 | 0.13806 | 0.15444 | 0.16815 |
| Cross 3072/1024 | 0.42754 | 0.39112 | 0.40540 |
| Cross＋Gate 3072/1024 | 0.44611 | 0.42824 | 0.44480 |


### 10.4 测量口径和详细记录的存放

MIRAGE默认训练执行方案在PRO6000上采用图像cuDNN、变长文本Triton及其他融合操作，
训练每方案12次更新，排除前2次，用后10次中位数；不包含shard磁盘读取、保存、预览和诊断。
图中显存是PyTorch allocated峰值，不等于nvidia-smi整进程显存。

3090推理只计预热后主体采样，排除加载、文本编码、首次编译、VAE解码和写图。
单层Attention使用相同Q/K/V，16Heads、HeadDim128、BF16，

512×512的补充采样结果：PyTorch自动18.337秒，原训练Triton前向4.976秒，
MIRAGE推理混合4.857秒，Triton融合＋FA4核心4.916秒；已分配峰值分别约3.248、3.233、3.233、3.233GiB。
768×1024已分配峰值分别约3.561、3.515、3.515、3.515GiB。


## 11 当前画质问题与后续训练

目前生成效果仍不理想，较常见的问题是手臂、腿、手指、关节连接和复杂全身姿态不稳定；多人图肢体破碎。单画风以及多花费融合效果还行。
主体或局部纹理较好也不代表人体关系已经学会。增加采样步数或CFG不能可靠补上缺少的训练知识。

**当前的主要判断是：全身图片及姿态覆盖不足，加上总图片曝光量仍不足，可能是肢体问题的重要原因。**

后续会继续训练优化，通过补充高质量全身、多姿态、合理肢体和多人交互的图片数据集。

### 11.1 相同Prompt在不同种子下的画风不稳定

当前还有一个明确的使用问题：**相同提示词没有额外风格引导时，更换随机种子可能得到明显不同的画风，不能稳定保持一个相近的视觉风格，可能是有效batch过小。

这是当前模型在风格可控性和输出一致性上的局限，

## 12 推理操作与兼容要求

在仓库根目录运行，模型与外部编码器按README放在models目录：

```bash
python infer.py --prompt "sfw, 1girl, blue_hair, full_body, standing, white_shirt" --width 512 --height 512 --steps 40 --sampler euler --cfg 4
```

不指定seed则每次随机，实际seed会打印；需要可复现比较时显式设置 `--seed 42`。
默认负面提示词为空，CFG参考分支为编码后的空文本.

务必确认：

- 采用匹配的Qwen3-Embedding-0.6B与FLUX.2 VAE。
- 保留8个文本Register及对应权重，严格加载模型配置。
- 使用Tag列表，首项明确为sfw或nsfw。


## 13 代码导航与公开范围

| 内容 | 位置 |
|---|---|
| 主体及TextAdapter/Gate/Register | `mirage/architecture/model.py` |
| Packed文本与偏移 | `mirage/architecture/packed_text.py` |
| Attention调度与变长核 | `mirage/attention/` |
| Norm、RoPE、AttnRes和残差融合 | `mirage/ops/` |
| Muon/AdamW及梯度处理 | `mirage/optimizers/` |
| RF、缓存及权重读取 | `mirage/runtime/` |
| Qwen与VAE接口 | `integrations/` |
| 推理入口 | `infer.py` |

公开目录展示可复用模型、训练/推理算子及推理入口。




