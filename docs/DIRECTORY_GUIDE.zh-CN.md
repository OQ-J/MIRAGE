# 代码目录与 Muon 文件分工

```text
mirage/
  __init__.py       # 统一模型入口
  architecture/    # 模型结构、完整文本 Token 的 packed 布局
  attention/       # Self / Cross Attention 调度、变长核及反向
  ops/             # RMSNorm、RoPE、QKV、Gate、AttnRes 等融合计算
  optimizers/      # Muon / AdamW、梯度范数与权重更新
  runtime/         # RF、采样、文本/常量缓存、权重读取
```

包装层和对应 Triton 实现放在同一功能目录，方便对照 PyTorch 与 GPU 内核。
训练和推理共用许多内核，因此不强行复制成两套文件。


