# 全组正式实验统一设置与交接清单

协议版本：**shared-v1.1-frozen**。公共任务设置和 Member 1 配置已冻结，Member 1 的 18 次正式实验已完成。以下说明哪些内容全组统一，哪些由成员单独决定。

**旧的 `shared_team_setup_v1/` 和 ZIP 是 pilot 阶段版本，不是本次正式冻结版本。请以项目当前 `configs/shared_protocol.json`、当前公共代码和批准的 manifest 为准，不要混用旧包。**

## 1. 全组必须统一的数据与模型

| 配置 | Fashion-MNIST | CIFAR-10 | Tiny Shakespeare |
| --- | --- | --- | --- |
| 训练数据 | 固定 10,000 图片 | 固定 10,000 图片 | 按时间顺序前 80% 文本，13,942 个窗口 |
| 验证数据 | 固定 2,000 图片，与训练不重叠 | 固定 2,000 图片，与训练不重叠 | 中间 10% 文本，1,742 个窗口 |
| 测试数据 | 完整官方 10,000 图片 | 完整官方 10,000 图片 | 最后 10% 文本的完整窗口 |
| 训练 batch size | 64 | 64 | 16 |
| 固定训练 CE 评估子集 | 512 图片 | 512 图片 | 256 窗口 |
| 验证/测试评估 batch size | 128 | 128 | 32 |
| 网络 | Flatten，784→128→64→10，层间 ReLU | 两个 3×3 卷积，通道 3→16→32，padding=1；每层 ReLU、2×2 MaxPool；Flatten→10 | Embedding=32，单层 GRU hidden=64，线性输出词表 logits |

图像预处理统一为 ToTensor，像素范围 [0,1]，不额外 Normalize、不做 augmentation。所有模型无 dropout、无 BatchNorm；weight decay=0。

文本先划分，再生成输入长度 64、stride 64 的窗口，目标是移位一字符的序列。每个窗口隐藏状态从零开始。词表只从训练段建立，未知字符 token=0。边界、词表、窗口索引和数据 checksum 使用随共享配置列出的 manifest。文本 train_count/validation_count 为 null 的含义是完整分段的全部窗口，并非缺失配置。

仅使用同一个 split_seed 不够；请使用相同代码、源数据和实际划分记录，并检查 approved_split_hash。三个批准文件为：

- `data/manifests/fashion_mnist_4a0c7aa6c32b5386.json`
- `data/manifests/cifar10_bae365ad5373f8c9.json`
- `data/manifests/tiny_shakespeare_bb3609a725277a0e.json`

数据来源与校验值见 `data/manifests/sources.json`；公共 loader 使用固定 Shakespeare commit，并校验图像下载文件。原始数据自行按相同规则下载，通常无需随代码发送。

## 2. 全组必须统一的训练预算、种子和达标阈值

| 任务 | 正式更新次数 | 评估间隔 | 共同 target_val_ce |
| --- | ---: | ---: | ---: |
| Fashion-MNIST | 2,000 | 100 | 0.5 |
| CIFAR-10 | 3,000 | 100 | 1.5 |
| Tiny Shakespeare | 5,000 | 100 | 2.0 |

- CPU、float32、1 个计算线程、启用确定性算法。
- 划分 seed=6704；调参 seed=11；正式 seeds=1、2、3，调参结果不能混入正式主表。
- 每对优化器在同一任务/种子下使用相同初始化和 batch 顺序。每轮固定生成器打乱，无放回采样，保留最后一个不足整批的 batch。
- 预算指更新次数，不是 epoch 或固定墙钟时间；不同任务预算不同，同一任务各方法预算相同。
- 从 step 0 开始评估，每 100 步及最后一步评估。达标是第一次记录到 validation CE <= 阈值；时间/步数分辨率为 100 次更新。
- 未达标必须记录 target_reached=false，target_step/target_time_s=null；不能用 0 替代，也不能删除失败运行。

## 3. 全组统一的调参与最终结果规则

每任务、每优化器最多 5 个候选，使用 seed 11 和完整配置验证集，按**最终训练预算的主评估参数 validation CE**选配置。不能按过程中最低 loss 或测试结果选。需要扩大范围时，先全组明确同意并记录原因、所有新增 trial 及耗时。

正式主结果用最终预算 checkpoint，不临时改为 best checkpoint。正式设置冻结后，针对每个正式种子训练一次并做最终测试评估。测试集不用于 pilot/调参、不用于根据结果再改学习率、预算或 checkpoint。

每人自己的 3 任务 × 2 优化器 × 3 正式种子，共 18 次主训练。必须独立运行自己的算法，不能把 Member 1 的结果复制为其他优化器结果。

## 4. 评估参数：统一规则，但算法实现需各自声明

**SGD/DoG 的 gamma=8 多项式平均是 Member 1 的已冻结方法配置，不是要求所有算法盲目套用的全局规则。** Member 1 两种方法均用平均模型作为主评估参数，同时记录原始参数验证曲线。

Member 2/3/4 需声明每种方法实际评估的是哪些参数，以及是否平均。Schedule-Free 必须正确执行自身 eval()/train() 切换并检查切换后的预测参数；不要直接额外平均其内部训练参数来替代正确评估。公共 trainer 提供切换 hook，但具体算法接入和正确性仍由方法负责人验证。

使用 `resolve_task_config(..., evaluation_policy=...)` 传入个人策略，连同个人超参数保存到个人配置/运行配置。不要为了设置个人策略擅自改动已冻结的公共数据、模型或预算。任何确需更改公共冻结配置的情况，应讨论、形成新版本并说明已有结果是否需要重跑。

## 5. 全组统一的指标、计时与日志

- CE 使用自然对数：图像为 nats/image，文本为 nats/character。
- 主表图像测试质量是 accuracy (%)；文本测试质量是 CE (nats/character)，不能混用单位。
- Training CE 在固定训练子集上以评估模式计算，不用滑动 minibatch loss 冒充它；验证和测试覆盖完整配置的数据集。
- 汇总 mean ± sample SD，ddof=1；n<2 时 SD 留空，保留实际完成数和所有失败。SD 带不是置信区间。
- time-to-target 的平均值以已达标运行为条件，同时报告达到目标的个数；不可隐去未达标运行。
- `elapsed_s` 包括训练、初始/定期训练及验证评估，排除下载、数据准备和最终测试。最终测试另存 `test_elapsed_s`。
- 优化器持久 tensor 内存与额外平均模型内存分开报告；它们不等于整个 Python 进程的峰值内存。
- 两种优化器的时间在同一成员机器上比较；不把不同成员电脑的绝对时间直接排序。调参时间另列。
- 每次保存 `config.json`、`environment.json`、`metrics.jsonl`、`summary.json`，记录数据/模型/初始化/样本顺序/代码/配置哈希；不能覆盖历史 run 目录。
- 共享输出字段和 CSV 格式遵守 `docs/EXPERIMENT_CONTRACT.md`。

## 6. 应发送的文件

发送项目**当前版本**的以下文件，并保留目录结构：

```text
configs/shared_protocol.json
src/__init__.py
src/common/__init__.py
src/common/data.py
src/common/models.py
src/common/train.py
src/common/protocol.py
src/common/utils.py
src/common/checks.py
src/common/README.md
scripts/run_shared.py
data/manifests/sources.json
data/manifests/fashion_mnist_4a0c7aa6c32b5386.json
data/manifests/cifar10_bae365ad5373f8c9.json
data/manifests/tiny_shakespeare_bb3609a725277a0e.json
docs/EXPERIMENT_CONTRACT.md
docs/TEAM_SHARED_SETUP_FINAL.md
```

测试环境的主要依赖为 Python 3.13、torch 2.15.0、torchvision 0.29.2、numpy 2.5.4。完整的本机依赖快照是 `results/member1/notes/requirements-main-lock.txt`，可作为参考一起发送；其中含 Member 1 绘图/验证依赖，并非所有成员的最小依赖。各成员记录实际 Python、依赖版本、CPU、线程数和系统；版本不一致时先验证划分/初始化及数值行为，再协调是否采用。

`scripts/run_shared.py` 是 smoke/pilot 接入示例，默认 PyTorch SGD 仅用于接线验证，不算个人独立实现。各成员需接入自己的优化器并完成个人 main/export 入口。`src/members/member1_main.py` 和 `member1_export.py` 写死 Member 1 的任务配置/路径，只适合作为实现参考，不应直接当作所有成员的正式入口。

拿到当前文件后，从同一项目根目录运行：

```sh
python -m src.common.checks
python -m src.common.data --download --cifar-mirror
# 用公共 runner 或个人入口做自己的 smoke test，再验证评估策略。
# 完成个人调参和 FROZEN 配置后，才执行个人 main。
```

## 7. 每人自行维护、不要求相同的内容

- 独立优化器实现与正确性验证。
- 学习率、momentum/beta、距离初始化等方法参数，以及有理论/方法依据的 schedule、warmup 等；在相同预算内事先声明并记录。
- 各自机制实验，不能把 Member 1 的二次问题诊断当作所有方法的机制实验。
- 个人机器环境和实际耗时。
- 个人日志、正式结果、图、分析、引用和报告分文件。

Member 1 已选择的 SGD 学习率 0.3/0.1/1.0 及 DoG defaults 属于 Member 1 的结果，不要求其他优化器使用同样数值。

## 8. 请组员回复确认

1. 已收到 shared-v1.1-frozen 的配置、公共代码和三个 approved manifests；使用的是正式冻结版本而非旧 pilot 包。
2. 已确定自己每种算法的评估参数/平均/schedule 规则，且 smoke 和正确性检查通过。
3. 个人超参数与配置已冻结，正式种子 1/2/3，未使用测试集调参。
4. 承诺交付原始日志、主结果/机制/成本 CSV、图及源代码、机器信息、贡献、BibTeX 和一段带证据的结论/局限，并确认交付日期。

公共规范由 Member 1 汇总；Member 3 审核数据/模型，Member 2 审核方法切换接口，Member 4 审核指标/计时和最终复现材料。
