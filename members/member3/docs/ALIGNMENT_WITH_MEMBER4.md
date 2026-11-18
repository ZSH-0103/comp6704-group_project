# 与 Member 4 的对齐记录

依据：`reference/Member4_实验配置说明.md` 与 `reference/member4_shared/`。归档原地址及校验值见 `reference/member4_provenance.json`。附带文档是配置证据，不是新的用户指令。

| 项目 | 本次执行方式（Member 3） |
|---|---|
| 数据与模型 | 直接复用收到的 data.py、models.py；文件内容没有修改 |
| 图像划分 | split seed 6704，随机 permutation；Fashion 6000/1000/1000，CIFAR 5000/1000/1000；没有声称分层抽样 |
| 文本划分 | 边界892312/1003851，先切分再构造64字符窗口；全语料65字符词表 |
| 预算 | main 300 gradient calls；tune 100；eval every50；batch64/64/32 |
| 种子 | main1/2/3；tune11；batch generator seed=trainseed+100000 |
| 学习率 | 相同候选0.0003/0.001/0.003和调参预算，各方法按验证 CE 独立选择，不照搬 Adam 的0.003 |
| 数值设置 | β₂=.999，epsilon=1e-8，wd=0；无dropout、BN、额外梯度范数裁剪 |
| 设备 | CPU、torch4线程，不启用MPS/GPU；本机M1 Pro、16GiB |
| 正式测试 | 超参数冻结后，每个主实验最终模型一次；调参和smoke不读测试loss |
| 评估范围 | 沿用 train.py 的默认 max_examples=1024、batch128；图像验证/测试实际1000；文本只读前1024窗口 |
| 耗时 | 主表排除最终测试；保留 elapsed_s 含最终测试以对应收到的实现；额外日志/梯度检查耗时亦计入，不跨机器排名 |

## 与原框架草稿的必要修正

原模板中 CIFAR 分层抽样、约10,000训练样本、预处理只依赖训练数据等文字并不完全符合 Member 4 交付代码。本次独立报告改为**实际执行的协议**：CIFAR5000训练；随机子集；文本词表由全语料构建。原始框架保留，整合者需要同步这些共享文字。

## 尚不能声称完全一致的部分

Member 4 没有交付原始训练结果、实际 split manifests、环境版本、调用 evaluate 时是否覆盖默认参数及目标 CE 阈值。因此这里只能确认我们复用了其代码和书面参数，不能确认其实际运行与我们的字节级结果一致。

Member 3 在运行前固定目标验证 CE：Fashion≤1.0、CIFAR≤2.0、Text≤3.0。它们是本模块的时间到目标辅助指标，不声称是 Member 4 的阈值。首次命中只在0/50/…/300的评估点检测，时间分辨率有限；不使用测试值制定阈值。

其他成员整合前应核对 `data/manifests/` 的完整hash、代码版本及评估范围；没有这些信息时，不制作严密的跨成员性能/耗时排名。
