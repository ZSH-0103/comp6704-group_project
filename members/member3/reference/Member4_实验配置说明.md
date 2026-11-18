# Member 4 实验配置说明

我负责的对比实验是 **Adam vs. AdEMAMix**。以下是实际主实验使用的模型、优化器超参数和统一实验设置，可用于复现实验或与其他组员保持一致。

## 1. 优化器超参数

主实验中，Adam 和 AdEMAMix 在三个任务上最终都使用 **learning rate = 0.003**。该学习率通过独立的调参实验，从以下候选值中选出：

```text
{0.0003, 0.001, 0.003}
```

调参使用 seed 11，每个候选配置运行 100 updates，并仅根据 validation cross-entropy 选择最终配置。

### Adam

| 超参数 | 数值 |
|---|---:|
| Learning rate | 0.003 |
| Beta 1 | 0.9 |
| Beta 2 | 0.999 |
| Epsilon | 1e-8 |
| Weight decay | 0 |

### AdEMAMix

| 超参数 | 数值 |
|---|---:|
| Learning rate | 0.003 |
| Beta 1 | 0.9 |
| Beta 2 | 0.999 |
| Beta 3 | 0.9999 |
| Alpha | 2.0 |
| Alpha warmup | 前 50% 训练步数 |
| Beta 3 warmup | 前 50% 训练步数 |
| Epsilon | 1e-8 |
| Weight decay | 0 |

AdEMAMix 的 slow EMA 不进行 bias correction。对于 300-update 的主实验，Alpha 和 Beta 3 的 warmup 都持续前 **150 updates**。

## 2. 模型结构

### Fashion-MNIST

- 模型：MLP
- 网络宽度：`784 -> 128 -> 64 -> 10`
- 隐藏层激活函数：ReLU
- Batch size：64
- 训练集：6000 个样本
- 验证集：1000 个样本
- 测试集：1000 个样本

### CIFAR-10

- 模型：小型 CNN
- 第一层：`Conv2d(3, 16, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)`
- 第二层：`Conv2d(16, 32, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)`
- 分类层：`Flatten -> Linear(32 * 8 * 8, 10)`
- Batch size：64
- 训练集：5000 个样本
- 验证集：1000 个样本
- 测试集：1000 个样本

### Tiny Shakespeare

- 模型：character-level 单层 GRU
- Embedding dimension：32
- Hidden size：64
- GRU layers：1
- Sequence length：64
- 输出层：`Linear(hidden_size, vocabulary_size)`
- Batch size：32
- 固定数据切分边界：`[892312, 1003851]`

## 3. 统一实验协议

- Loss：cross-entropy，使用自然对数
- 主实验训练预算：300 updates
- 验证间隔：每 50 updates 验证一次
- 主实验 seeds：1、2、3
- 调参 seed：11
- 数据切分 seed：6704
- 运行设备：CPU
- PyTorch threads：4
- Dropout：0
- Batch normalization：不使用
- Weight decay：0
- 每组 Adam/AdEMAMix 实验使用相同的模型初始化和相同的 batch 顺序
- 测试集只在超参数冻结后使用一次
- 计时包括训练和定期验证，不包括数据下载与预处理

## 4. 额外诊断实验

为了检查 slow EMA 所积累的长历史是否有效，我还在 Fashion-MNIST 上进行了 history ablation：

- 训练长度：100 和 400 updates
- Seeds：1、2、3
- 对比方法：Adam、AdEMAMix、AdEMAMix without slow EMA
- 100-update 实验的 warmup：50 updates
- 400-update 实验的 warmup：200 updates

## 5. 实现说明

- Adam 使用独立实现，并与 `torch.optim.Adam` 的固定梯度更新结果进行过一致性检查。
- AdEMAMix 按论文更新规则实现。
- AdEMAMix 的 slow EMA 不进行 bias correction。
- 官方实现参考：`apple/ml-ademamix@eb517ed3bbc09227e5d062b682723543f6c71f1a`
