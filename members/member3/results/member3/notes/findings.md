# Member 3：结果解读与公共表行

## 要回答的问题

“排除当前梯度对其归一化分母的影响，能否减轻非对称随机梯度下的偏差？这种机制优势是否也转化为短预算神经网络训练优势？”不预设新算法一定胜出。

## 真实主结果

均值±样本SD，3个种子；两个图像任务是测试准确率（越高越好），文本是测试CE（nats/character，越低越好）。

| 任务 | RMSProp（Member 3） | ADOPT-0（Member 3） | 当前设置下的解读 |
|---|---:|---:|---|
| Fashion-MNIST | 81.17 ± 0.29 % | 79.43 ± 0.91 % | RMSProp更好；配对优势1.73±0.76百分点 |
| CIFAR-10 | 47.47 ± 2.18 % | 47.33 ± 2.45 % | 差距很小；配对差0.13±0.65百分点，不能宣称稳定领先 |
| Tiny Shakespeare | 2.028 ± 0.012 | 2.086 ± 0.013 | RMSProp更好；配对CE低0.058±0.016 |

主表：`tables/main_results.csv`。配对差值文件的方向是ADOPT减RMSProp；不要把它的SD当作某个算法自身的SD。CIFAR上ADOPT的训练CE更低（1.183 vs1.224），但验证和测试没有相应提升，说明训练目标下降和泛化质量不能混为一谈。

18次主训练全部完成，正式test每个run只在最后评估一次。每对初始化、分割和批次序列hash一致。冻结LR：Fashion/CIFAR RMSProp=.0003、ADOPT=.001；Text两者=.003。图像RMSProp和两个文本方法位于候选范围边界，不能称已经找到最佳学习率。调参100步而主训练300步，可能偏向早期进展；没有根据主实验test结果追加搜索。

## 机制结果如何理解

无约束 logcosh 目标，最优点0。噪声均值0但非对称；共享每个seed的全部噪声。β₂=.9时，最后1000次调用平均目标gap：RMSProp .1094±.0718，ADOPT-0 .0135±.0198；20种子。对应平均θ为.448和-.008。归一化可使大正梯度的相对贡献被压低，即使原始梯度无偏，平均更新仍会偏移。

匹配裁剪的current-moment对照仍为.1094±.0718；去裁剪ADOPT为.0112±.0118。因此在此诊断中，结果支持归一化时序的作用，不能只归因于是否裁剪。β₂=.99时差距明显缩小；.999时两方法gap均约.0084。所有240条轨迹完成，未出现非有限数值。

控制组只支持这个问题、预算和噪声条件下的解释，不证明RMSProp普遍不收敛，也不验证论文的渐近速率。主神经网络使用β₂=.999，且对比同时改变初始化、epsilon和裁剪；不能由神经曲线单独识别因果机制。

## 成本与稳定性

- 主表训练+计划评估平均耗时约2.04/2.06秒（Fashion）、9.49/9.59秒（CIFAR）、4.66/4.71秒（Text），顺序RMSProp/ADOPT。这是M1 Pro CPU4线程上的小任务，额外有限梯度检查也计时，不与其他成员机器直接排名。
- Tensor优化器状态分别437544、102312、100484 bytes；两方法相同。β₁=0特化实现不存冗余动量tensor，不等于完整官方ADOPT的内存量。
- 300次梯度调用对应RMSProp300次参数更新、ADOPT299次。
- 阈值1.0/2.0/3.0都在第50步评估时命中，3/3达标；因此该辅助指标分辨率不足，不事后改阈值制造差距。
- 每算法每任务3次学习率试验；实际调参总秒数见cost_stability.csv。无减少调参成本的证据。

## 给公共章节的行

| 目标表 | 可直接整合的Member 3内容 | 负责人 |
|---|---|---|
| `tab:methods` | RMSProp：当前二阶矩、epsilon在sqrt外、无momentum；ADOPT-0：前一二阶矩、max epsilon、step^1/4裁剪；均O(d)运算/状态，具体理论条件见2.4 | (Member 2整合，Member 3提供) |
| `tab:protocol` | LR候选[.0003,.001,.003]，seed11，100调用，最小验证CE；选中Fashion/CIFAR .0003/.001，Text .003/.003；β₂=.999、eps1e-8、wd0 | (Member 1整合，Member 3提供) |
| `tab:diagnostic-plan` | logcosh θ、非对称无偏噪声{-1:.9,9:.1}；β₂ .9/.99/.999；4变体；20seeds；10000调用；tail1000均值gap | (Member 2整合，Member 3提供) |
| `tab:hardware` | Apple M1 Pro，16GiB，macOS15.6.1，Python3.12.14，torch2.14.1，CPU4线程；完整依赖见lock | (Member 4整合，Member 3提供) |
| `tab:synthesis` | ADOPT-0在β₂=.9可控噪声诊断中减轻偏差，β₂=.999差距很小；本预算下Fashion/Text RMSProp更好，CIFAR近似；不支持普遍胜出 | (Member 1整合，Member 3提供) |
| `tab:contributions` | 独立实现与外部代码核验；18次调参、18次主实验、240机制轨迹；维护共享数据/模型规格、M3表图与讨论；引用Member 4共享代码 | (Member 1整合，Member 3核对) |

以上为本次协助完成的实际工作范围。共享数据、模型和原训练代码注明来自Member 4；独立优化器实现及验证记录见implementation_check.md。

## 可用于Abstract/Conclusion的一句英文

Under a matched 300-gradient-call CPU budget, momentum-free ADOPT did not improve held-out quality on the three tested small models, although a controlled asymmetric-noise diagnostic supported its previous-moment normalization mechanism at beta2=0.9.

## 局限与整合前确认

只用小模型、小图像子集、3种子、短预算和窄学习率搜索；文本评估沿用Member 4代码的前1024个重叠窗口，不是整个验证/测试区间；词表由全语料构建；默认ADOPT含动量不在本次比较中。Member 4未交付其真实manifest、环境版本和结果，因此不能做严格跨成员排名。完整Discussion需再收集另外三位成员的证据。
