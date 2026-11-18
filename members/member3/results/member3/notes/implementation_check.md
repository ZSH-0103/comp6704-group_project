# 实现核对（Member 3）

自行实现：`src/members/optimizers.py`。训练仪表：`src/members/training.py`。共享数据和模型来自 Member 4，经 SHA-256 记录，保留原副本在 `reference/member4_shared/`。

`python -m pytest -q` 共15项通过：
- 两个 β₂ 水平下，RMSProp 与 `torch.optim.RMSprop` 逐步一致；设置 non-centered、momentum=0、weight_decay=0、foreach=False。
- 两个 β₂ 水平、裁剪/不裁剪两种 ADOPT 设置，与固定提交官方实现逐步一致。float64、含零梯度、极小梯度、突增梯度和随机梯度，rtol=atol=1e-12。
- 两种方法的状态保存/恢复、无梯度参数跳过处理，以及 ADOPT 首次仅初始化。
- 新训练日志与 Member 4 原训练循环的最终参数、训练/验证/测试 CE 完全一致（两种方法，固定小型数据集）；计时字段不做数值相等断言。
- 四个 NumPy 机制实验变体均与 PyTorch 优化器逐步一致，rtol=atol=1e-13。

官方代码在当前 PyTorch 有弃用警告，验证仍全部通过；没有修改官方代码以规避测试。版本、源码 SHA-256 和许可证见 `reference/adopt_source.json`、`reference/ADOPT_LICENSE`。本实现支持实数稠密梯度；不是对复杂数、稀疏梯度、AMP 或 foreach 的通用库实现。

β₁=0 时省略冗余的一阶动量 tensor，两者均存一个二阶矩 tensor；`optimizer_state_bytes` 只统计状态 tensor，不包括 Python 对象、模型参数、梯度、激活或进程峰值内存。ADOPT 的第一次梯度调用不移动参数，300次调用只产生299次参数更新。

新增日志保留批次序列 hash、冻结配置、配对初始参数 hash 和两个时间口径。额外的有限梯度检查也计入计时，因此不直接比较其他成员机器/循环的绝对秒数。
