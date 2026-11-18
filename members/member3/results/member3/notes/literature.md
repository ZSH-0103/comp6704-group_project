# Member 3 文献与复现范围

主文献：Taniguchi et al., **ADOPT: Modified Adam Can Converge with Any β₂ with the Optimal Rate**, NeurIPS 2024。
- 论文：https://proceedings.neurips.cc/paper_files/paper/2024/file/84d286e32bbee8fa3a86ee9c50e00081-Paper-Conference.pdf
- 官方实现：https://github.com/iShohei220/adopt/tree/6468572b4b3688a2f056b796b071da540568520e
- RMSProp 讲义来源：https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

## 读论文时需要抓住什么

1. §3.1：当前随机梯度同时进入分子和二阶矩，导致归一化后的随机更新可能出现偏差；使用前一时刻的二阶矩可以消除这种“同一样本同时影响分子和分母”的依赖。
2. §3.2：包含动量时还需注意先归一化还是先累积动量。本项目 β₁=0，不研究这部分动量机制，也不声称完整复现默认 ADOPT。
3. Algorithm 4 / Appendix Algorithm C 与官方代码：首先 g₀² 初始化 v；第一次梯度调用不更新参数；后续先用旧 v 归一化、更新参数，再更新 v。epsilon 是 max(sqrt(v),epsilon)，不同于 RMSProp 的 sqrt(v)+epsilon。官方代码默认逐坐标裁剪界限 step**0.25。
4. Theorem 4.1：目标下有界、无偏随机梯度、L-smooth、随机梯度二阶矩一致有界；适当的 α=Θ(1/√T) 得到指定平稳性度量的 O(1/√T) 上界。不能用“只有噪声方差有界”替代二阶矩条件。不能直接将定理用于 ReLU 网络或声称任意学习率均收敛。

## 我们实际复现了什么

独立实现 RMSProp 与 ADOPT 的 β₁=0 特例，并与外部实现逐步核对。β₂=.999、epsilon=1e-8 对齐 Member 4；官方 ADOPT 默认 β₂=.9999、epsilon=1e-6 不同，因此这是受控的课程实验，不是论文默认超参数的复现。

论文包含受约束的一维反例、MNIST MLP，以及视觉、语言、生成和强化学习实验。本项目只做 Fashion-MNIST、CIFAR-10 固定小子集和 Tiny Shakespeare 小 GRU，不复现其大规模任务、完整 Adam/ADOPT 排名或渐近速率。

原文玩具问题带 θ∈[-1,1] 约束；我们另行定义真正无约束的 log(cosh θ) 目标。非对称噪声取 -1（概率 .9）或 +9（概率 .1），均值0、方差9。目标导数 tanh θ、二阶导数不超过1，E[g²|θ]=9+tanh²θ≤10。最优值0、最优点0已知。该问题用于观察有限预算下的归一化偏差，不是假称论文原始实验，也不证明一般收敛定理。

## 消融如何排除替代解释

主对比同时改变初始化、epsilon 位置、裁剪和归一化时序，不能单凭两条曲线归因。加入 current_moment_matched_clip：采用和 ADOPT-0 相同初始化、第一次调用规则、epsilon 和裁剪，只把二阶矩更新放到归一化之前。再加入 adopt_no_clip，观察裁剪影响。每个 seed 共享同一串 oracle noise，扫描 β₂=.9/.99/.999，固定学习率.01、10,000次调用、20种子。控制组的结果单独记录，不混入三任务主实验。
