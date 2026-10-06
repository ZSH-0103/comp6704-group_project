# Member 3 独立工作 Prompt

请负责 COMP6704 项目的 Member 3 模块：**RMSProp 与无动量 ADOPT**。先阅读下面的全局思路、接口和自己的文件，再实施。你要完成自己的理论、独立实现、三个共同任务、机制实验、分析、报告内容和复现信息。不要覆盖其他成员的工作，不编造结果，不假装尚未实现的阶段已经运行成功。

## 整篇论文的思路

整个报告围绕“经典优化器的某个局限—近期改进机制—共同任务比较—可控机制实验—代价与边界”展开。四个独立模块分别为 SGD/DoG、Momentum/Schedule-Free SGD、RMSProp/ADOPT β1=0、Adam/AdEMAMix。目标不是预设新方法必胜，而是回答改进何时有用、是否跨任务保持、付出什么代价。

报告框架：Abstract；1 Introduction and Related Work；2 Problem Formulation and Methodology（2.1 公共目标，2.2–2.5 四人的方法对，2.6 机制/理论/成本汇总）；3 Experimental Setup；4 Experimental Results and Analysis（4.1–4.4 各人的独立案例，4.5 跨任务汇总）；5 Discussion and Conclusions；Reproducibility；Team Contributions；References。

共同任务是 Fashion-MNIST + 小 MLP、CIFAR-10 固定子集 + 小 CNN、Tiny Shakespeare + 字符 GRU。同一个任务所有人用相同网络、数据切分及预算，每对优化器使用匹配初始化和数据顺序。普通 Mac CPU 运行，不依赖 GPU/MPS。正式种子暂定 1、2、3，每人 18 次主训练，调参和机制实验另计。训练步数、划分数量和最终超参数必须先试跑再冻结。测试集不用来调参，不直接比较不同成员 Mac 的绝对耗时。

课程报告总计最多 15 页；每人理论与结果初步按约 2–2.5 页组织。提交时所有结论必须来自实际结果，论文中的理论条件不能省略。最终展示 30 分钟加 10 分钟 Q&A，每人都要展示。

## 先读哪些本地文件

- 最新分工：`docs/分工3.0.md`
- assignment：`docs/COURSE_REQUIREMENTS.md`
- 报告入口：`report/main.tex`
- 模板使用指南：`report/TEMPLATE_GUIDE.md`
- 代码目录：`.`
- 统一数据与结果接口：`docs/EXPERIMENT_CONTRACT.md`
- 公共配置：`configs/shared_protocol.json`
- 个人配置：`configs/member3.json`

以下代码路径均相对于仓库根目录 `comp6704-group_project/`。报告在 `report/`，成员 Prompt 在 `docs/prompts/`，课程要求摘要在 `docs/COURSE_REQUIREMENTS.md`。请从仓库根目录执行命令。

## 必须参考的论文

主要论文：[ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/84d286e32bbee8fa3a86ee9c50e00081-Abstract-Conference.html)。报告的引用键为 `taniguchi2024adopt`。

实现参考：https://github.com/iShohei220/adopt。引用代码来源和版本，用它验证你自己的实现，不把简单调用官方优化器描述成个人独立实现。

辅助来源：[RMSProp 课程讲义](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)，明确它是课程讲义来源。

先产出 `results/member3/notes/literature.md`：原方法问题、改进具体步骤、适用假设、原实验规模、我们复现的范围、与本项目的区别。核对论文算法框、初始化、更新次序、超参数、理论定理条件；不要只根据摘要实现。

## 你负责的机制实验

**主题：归一化顺序与 β2。**

优先读 ADOPT 第 3.1 节，选一个适合 CPU 的无约束随机目标，完整定义随机梯度的分布与期望。若使用论文反例，核对是否有显式约束，不能把有界投影问题不加说明地当成无约束实验。可从 β2 = 0.9、0.99、0.999 的小范围开始，但先核对理论条件和对应实现。两方法使用相同随机样本，固定初始化、步数及其他数值处理。记录目标值或已知最优时的 gap、梯度或更新范数、发散/未达标比例，分析前一时刻二阶矩归一化的影响。

必须避免的误读：使用 ADOPT 的 β1=0 设置，明确不是另起名字的独立 RMSProp 改进论文。必须核对初始 v、第一步、更新顺序、epsilon 位置和 clipping；如为公平归因增加 matched-clipping 消融，单独记录配置。不能把 RMSProp 的 sqrt(v)+epsilon 和 ADOPT 的 max(sqrt(v),epsilon) 当成完全相同。

## 要实现和运行的代码

当前项目提供的是实验框架和命令接口，不是已经完成的训练程序。计划命令和 ZIP 打包已经可用；`src/members/member3.py` 的训练阶段目前会明确抛出 NotImplementedError。你需要补齐该模块，再运行下列阶段。不要仅把占位状态改成“已完成”。

你拥有：

- `src/members/member3.py`：实现 `run_phase(phase, config, repo_root)`，组织 check/smoke/tune/main/diagnostic/export；可以在同目录增加以 `member3_` 开头的辅助文件。
- `configs/member3.json`：你的算法超参数、学习率候选、诊断因素和配置状态。
- `results/member3`：你所有日志、表格、图和分析笔记。

共享数据、模型、训练接口在 `src/common/README.md` 定义，相关 Python 实现仍待补齐。按共同规范实现或复用它们，不私改数据划分、网络或预算。可以立即并行完成自己的论文、固定梯度正确性检查及小型机制实验，不必等待别人的实验结果；正式三任务运行前共同协议必须冻结。

从代码根目录运行：

```sh
cd .
python3 scripts/run_member.py --member 3 --phase plan
```

以下是完成实现后按顺序运行的命令，不能把 dry-run 算成实验：

```sh
python3 scripts/run_member.py --member 3 --phase check
python3 scripts/run_member.py --member 3 --phase smoke
python3 scripts/run_member.py --member 3 --phase tune
python3 scripts/run_member.py --member 3 --phase main
python3 scripts/run_member.py --member 3 --phase diagnostic
python3 scripts/run_member.py --member 3 --phase export
```

check 应验证更新正确性与相关状态切换；smoke 只做少量 CPU 更新检查三个任务能跑通；tune 用验证集选择配置；只有共享和个人配置实际完备且标记 FROZEN 后才进入 main。每个 phase 都把结果写到自己的目录。精确依赖版本和实际运行时间需要从你的真实环境记录。

## 必须生成的数据和对应表格

所有数据放在 `results/member3`。数值字段、单位和缺失值规则以 `docs/EXPERIMENT_CONTRACT.md` 为准。

| 产物 | 内容 | 报告填写位置 |
| --- | --- | --- |
| `raw/<run_id>/config.json` | 配置、种子、划分和模型 hash、版本、预算 | Setup 和可复现证据 |
| `raw/<run_id>/environment.json` | CPU、系统、库版本和线程数 | `tab:hardware` 的 (Member 3) 行 |
| `raw/<run_id>/metrics.jsonl` | step、elapsed_s、训练/验证 CE 等过程记录 | 收敛图和效率图 |
| `raw/<run_id>/summary.json` | 最终指标、状态、达标时间、state bytes | 所有汇总表的来源 |
| `tables/main_results.csv` | 三任务 × 两算法共 6 行，均值、样本 SD、成功数 | **M3-MAIN / `tab:m3-main`** 全表 |
| `tables/diagnostic_results.csv` | 每个因素水平 × 方法的机制指标、单位、种子数和配置 hash | **M3-DIAG / `tab:m3-diag`** 全表 |
| `tables/cost_stability.csv` | 失败数、达标数、达标时间、内存与调参成本 | 4.3 分析段和综合表证据 |
| `notes/tuning_log.csv` | 每次调参配置、验证分数、耗时、失败及是否被选择 | `tab:protocol` 的本人算法行 |
| `notes/implementation_check.md` | 自己实现的文件、测试方法、参考来源和误差 | 2.4 与贡献表 |
| `notes/findings.md` | 假设、支持/反对证据、局限、供全组汇总的结论 | `tab:synthesis` 的本人行及摘要/讨论 |
| `figures/member3_overview.pdf` | loss 对更新、loss 对耗时、机制图；注明不确定性与单位 | 4.3 图占位 |

图像质量是 test accuracy (%)，文本质量是 test CE (nats/character)，不能混用单位。主表只汇总 main 阶段选定配置的正式种子，不能混入试跑或调参数据。失败不能删除；未达标时间用空值加标记，不能填 0。数值不可用时用 null/空白并说明，不能虚构“正常”结果。记录 SD 的定义和实际 n。

## 要写哪些 TeX 文件

你是以下文件的主要编辑者；其中公共整合文件需要收集其他成员的内容后统一填写：

- `report/sections/section2_methodology_member3.tex`
- `report/sections/section3_datasets_models_member3.tex`
- `report/sections/section4_experiments_member3.tex`
- `report/sections/section5_discussion_member3.tex`

核心方法节是 2.4，核心结果节是 4.3。方法节写更新公式、机制、假设、复杂度、实现细节与验证；结果节按假设、三任务结果、机制实验、稳定性和代价、局限组织。表格标题和行中已有括号负责人标记；保留这些标记，除非最终提交前全组决定隐藏。

公共职责：维护三个任务的划分与模型规格、共享 data/model 接口；整合 Discussion。共享规则先冻结，让大家复用相同索引和模型配置，不接管其他人的实验。

在自己的 `notes/findings.md` 提供给他人的共享行：`tab:methods`、`tab:protocol`、`tab:diagnostic-plan`、`tab:hardware`、`tab:synthesis`、`tab:contributions`。整合负责人统一修改共享表，避免多人反复覆盖同一个文件。新增文献先写到 `notes/references_member3.bib` 供合并，不覆盖其他人的 bibliography。

将报告使用的最终图复制到 `report/figures/member3_overview.pdf`；源数据和绘图代码仍保留在项目目录。不要把中间图、日志和编译文件散落在 report 根目录。

## 编译和交付

```sh
sh report/build_report.sh
```

源文件位于 `report/sections/`，所有编译中间文件和预览进入 `report/build/`，最终阅读版为 `report/main.pdf`。保持官方样式文件不变，全部报告内容先按最多 15 页规划。

建议未来 Git 分支名是 `member3-rmsprop-adopt`；成员分支按需从 main 创建。提交自己的源代码、配置、真实日志、CSV、图源与报告分文件，不提交下载数据、环境目录或大模型 checkpoint。

本仓库以 `report/` 与 `docs/` 为可编辑版本。打包命令：

```sh
python3 scripts/package_project.py --output dist/member3_project_snapshot.zip
```

最后全组可生成统一 ZIP；它会包含当时的全部报告、源码、配置、结果和文档，但不上传 GitHub。最终交付需列出你改了哪些文件、真实执行过的命令、填好了哪些表、尚有哪些缺口。不要把别人的 TBD 换成猜测，也不要因自己完成就宣称整个项目已完成。
