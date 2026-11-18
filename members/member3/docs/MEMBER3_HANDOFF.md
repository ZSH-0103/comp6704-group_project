# Member 3 GitHub 交接

本模块位于仓库 `members/member3/`，是可独立运行的代码、配置和真实实验记录。
所有命令先 `cd members/member3`，具体步骤见本模块 README。

## 找到什么文件

| 内容 | 相对于本模块的位置 |
|---|---|
| RMSProp / ADOPT-0 独立实现 | `src/members/optimizers.py` |
| 保持 Member 4 原循环语义的日志扩展 | `src/members/training.py` |
| 正式实验入口 | `scripts/run_member3.py` |
| 无约束机制实验 | `scripts/run_diagnostic.py` |
| 冻结参数、主实验配置 | `configs/member3.json`, `configs/shared_protocol.json` |
| 实际分割索引和hash | `data/manifests/` |
| 完整日志、CSV、图 | `results/member3/` |
| 结果解释、公共表行 | `results/member3/notes/findings.md` |
| 实现检查 | `tests/`, `results/member3/notes/implementation_check.md` |
| 可分享ZIP生成器 | `scripts/package_member3.py` |

## 报告是实验时的快照

`report/member3_report.pdf` 和四份 `report/sections/*member3.tex` 来自2026年10月17日
个人实验交付，包含方法、共享Setup、结果和局限。其关于“尚未发布”“等待其他成员”的
文字描述的是当时状态；当前代码以本分支为准。不要用这份个人快照覆盖已经更新的全组报告。

如需整合到仓库根 `report/sections/` 或 Overleaf：

- `section2_methodology_member3.tex` 对应2.4；`section4_experiments_member3.tex` 对应4.3。
- `section3_datasets_models_member3.tex` 对应3.1/3.2，但其中300调用、四线程、前1024样本等
  设置必须明确归属于Member 3，不能当成所有成员统一协议。
- `section5_discussion_member3.tex` 是当时M3证据总结；全组讨论须结合其他成员后来提交的结果。
- `report/figures/member3_overview.pdf` 是主结果图；`member3_diagnostic.pdf` 用于机制展示。
- 引用键保持原模板不变，个人引用清单在 `results/member3/notes/references_member3.bib`。

四份分文件自身包含subsection标题。主文件应在对应位置只写input，不能保留重复旧正文。

## 公共表映射

| 产物 | 使用位置 |
|---|---|
| `tables/main_results.csv` | M3-MAIN / `tab:m3-main`（Member 3） |
| `tables/diagnostic_results.csv` | M3-DIAG / `tab:m3-diag`，主指标tail_mean_objective（Member 3） |
| `tables/cost_stability.csv` | 4.3成本/达标时间/状态存储（Member 3） |
| `tables/paired_differences.csv` | 配对差值与对应SD（Member 3） |
| `notes/tuning_log.csv` | `tab:protocol`（Member 1整合） |
| `environment.json` | `tab:hardware`（Member 4整合） |
| `notes/findings.md` | methods、diagnostic-plan、synthesis、contributions各表的M3行 |

上述路径的共同前缀为 `results/member3/`。详细对齐说明在 `ALIGNMENT_WITH_MEMBER4.md`。

## 与根目录的关系

代码留在独立子目录，保持产生原日志的实现、配置和import路径，避免改变队友的运行结果。
此处的data/models/train与仓库基线 `f2d2f79` 的Member 4共享代码逐字节一致。
M3自己的验证阈值是1.0/2.0/3.0，不覆盖根目录Member 4的0.75/1.8/2.2。
根目录的 `results/member3/README.md` 指向这里；此处是M3结果的权威副本。

根目录旧dispatcher的Member 3入口没有承载本次实验，请使用本模块README中的真实命令。
大数据、环境、模型checkpoint、build和本地ZIP不进入Git；ZIP可由打包脚本按需生成。
