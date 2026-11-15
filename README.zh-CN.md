# MIRAGE

[English](README.md) | [简体中文](README.zh-CN.md)

[Hugging Face 模型权重](https://huggingface.co/OQ-J/MIRAGE)

推荐推理设置：Euler 采样，40 步，CFG=4。

**当前 Hugging Face 模型效果不理想，详情见[技术报告](docs/TECHNICAL_DESIGN.zh-CN.md)。**

MIRAGE-1.65B — 使用 Tag 条件的 Rectified Flow 图像生成模型，配有 Triton 训练和推理内核。

[技术设计与工程说明](docs/TECHNICAL_DESIGN.zh-CN.md)


一份仅含代码的快照，包含文本条件 Rectified Flow 图像模型及其
Triton 训练和推理实现。

## 模型

- 20 个由图像自注意力、文本交叉注意力和图像 SwiGLU 组成的 Block。
- 图像宽度为 2048，FFN 宽度为 8192；16 个注意力头，每个头的维度为 128。
- 完整的 1024 维文本 Token 状态，两个 TextAdapter Block，八个前缀 Register。
- 每个分支使用 AttnRes；采用有效步长为 /16、128 通道的打包 latent。
- 文本嵌入由 Qwen3-Embedding-0.6B 生成；图像编码和解码使用 FLUX.2 VAE。
- 直接预测速度的 Rectified Flow：`t=0` 表示数据，`t=1` 表示噪声。

## 包含的实现

`mirage/architecture/model.py` 定义模型架构。配套的 Python 包装层与
Triton 内核放在一起：包括注意力及其梯度、RMS/AdaRMS、RoPE/QKV、
SwiGLU、带门控的紧凑文本注意力、AttnRes／历史梯度合并、残差写入、
RF 插值／MSE，以及梯度和参数更新内核。

推理还会跳过仅供反向传播使用的缓冲，并允许分别选择
图像和文本后端。可选的 Muon/AdamW 集成位于
`muon.py`、`muon_fast.py` 和 `optim.py`。

## 安装与检查

首先安装适合当前机器、支持 CUDA 的 PyTorch 版本，以及
与之匹配的 Triton。GPU 推理已在 Linux RTX 3090／SM86、
PyTorch 2.8.0+cu128 和 Triton 3.4.0 上测试。原训练优化器使用了
更新的训练环境；此次没有重新验证其编译后的 GPU 更新路径。

```bash
python -m pip install -e ".[inference]"
python infer.py --help
python examples/forward_backward.py
```

示例使用一个随机初始化的微型 CPU 模型，检查张量形状和梯度；
它不会生成训练模型的图像，也不对实际规模的内核进行性能测试。

## 使用自己的兼容权重推理

```bash
python infer.py --checkpoint models/model.pt \
  --qwen-model models/Qwen3-Embedding-0.6B \
  --vae-path models/FLUX.2-klein-4B \
  --prompt "sfw, 1girl, blue_hair, full_body" \
  --width 512 --height 512 --steps 40 --cfg 4 --num-images 1
```

VAE 目录中包含其 `vae/` 子目录；不需要 FLUX 去噪模型。
Qwen、VAE 和训练好的模型权重需要另行提供。
推理读取器使用 `weights_only=True`；支持由张量和元数据组成的导出文件，
不支持任意通过 Python pickle 序列化的训练对象。

默认 GPU 组合：Triton 融合算子＋PyTorch Flash 图像注意力＋自写的
Triton 文本注意力／Gate。安装可选的 FA4 依赖后，`--image-attention` 和 `--text-attention` 也接受
`fa4`。CPU／PyTorch 回退方式：

```bash
python infer.py --backend torch --image-attention sdpa --text-attention sdpa ...
```

## 测量结果与范围

历史 3090 Euler40／CFG4 采样结果：512² 为 18.34 秒 → 4.86 秒；768×1024
为 51.70 秒 → 15.29 秒，对应从 PyTorch 路径切换到混合融合路径。原有
Triton 路径的耗时已经是 4.98 秒和 15.69 秒：大部分收益来自启用
现有融合，而不是此次仅针对推理所做的小幅改动。

[详细测量结果](docs/TECHNICAL_DESIGN.zh-CN.md)分别记录注意力核心与完整
采样，包含显存指标，并保留对比结论的适用范围。



参见[算子映射](docs/KERNEL_MAP.md)，以及
[算法与依赖来源](ACKNOWLEDGEMENTS.md)。

## 许可证

可选的 `.[optim]` 扩展依赖要求 Torch>=2.12，可能升级现有
环境。推理不需要它；请保持 Torch、CUDA 和 Triton 相互兼容。

本仓库采用 MIT 许可证。
外部依赖和外部模型权重仍遵循各自的条款。


[目录](docs/DIRECTORY_GUIDE.zh-CN.md)


[训练与推理性能测试](docs/TECHNICAL_DESIGN.zh-CN.md)
