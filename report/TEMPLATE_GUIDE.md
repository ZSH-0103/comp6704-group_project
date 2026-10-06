# 报告模板填写指南

报告主文件：`report/main.tex`。先读最新 [分工 3.0](../docs/分工3.0.md) 和个人 Prompt。

## 已写与待写

已写入可修改初稿：Abstract 的问题和设计、Introduction、无约束目标、四对算法的机制概述、数据与模型方案、训练公平性、指标、限制。所有未实际运行的部分使用将来时或明确计划标记。模型和预算仍需试跑冻结。

待成员完成：准确更新公式与理论条件、独立算法和训练实现、所有实验、结果分析、真实环境和超参数、姓名学号、真实组号、GitHub URL。蓝色方括号和 TBD 是填写提示，不能当成结论。结果表以括号标明谁写。

## 按成员编辑

### Member 1

- `report/sections/frontmatter_member1.tex`
- `report/sections/section1_introduction_member1.tex`
- `report/sections/section2_methodology_member1.tex`
- `report/sections/section3_protocol_member1.tex`
- `report/sections/section4_experiments_member1.tex`
- `report/sections/section4_synthesis_member1.tex`
- `report/sections/contributions_member1.tex`

### Member 2

- `report/sections/section2_formulation_member2.tex`
- `report/sections/section2_methodology_member2.tex`
- `report/sections/section2_comparison_member2.tex`
- `report/sections/section3_diagnostics_member2.tex`
- `report/sections/section4_experiments_member2.tex`
- `report/sections/section5_limitations_member2.tex`

### Member 3

- `report/sections/section2_methodology_member3.tex`
- `report/sections/section3_datasets_models_member3.tex`
- `report/sections/section4_experiments_member3.tex`
- `report/sections/section5_discussion_member3.tex`

### Member 4

- `report/sections/abstract_member4.tex`
- `report/sections/section2_methodology_member4.tex`
- `report/sections/section3_metrics_member4.tex`
- `report/sections/section4_experiments_member4.tex`
- `report/sections/section5_conclusions_member4.tex`
- `report/sections/reproducibility_member4.tex`

各人核心结果表为 `tab:mN-main` 和 `tab:mN-diag`。CSV 和 TeX 手工填表之间必须核对数值、单位和种子数；目前没有自动将 CSV 导入 TeX 的逻辑。

## 表格对应

| 标签 | 当前用途 | 填写人 |
| --- | --- | --- |
| tab:methods | 算法机制/假设/状态存储 | (每人各填自己的行，Member 2 合并) |
| tab:data | 数据划分和模型配置 | (Member 3) |
| tab:protocol | 训练与调参设置 | (每人填自己的算法行，Member 1 合并) |
| tab:diagnostic-plan | 控制变量与机制设计 | (每人填自己的行，Member 2 合并) |
| tab:hardware | 机器和环境 | (每人填自己的行，Member 4 合并) |
| tab:m1-main / tab:m1-diag | SGD/DoG 主结果与机制结果 | (Member 1) |
| tab:m2-main / tab:m2-diag | Momentum/SF-SGD 主结果与机制结果 | (Member 2) |
| tab:m3-main / tab:m3-diag | RMSProp/ADOPT-0 主结果与机制结果 | (Member 3) |
| tab:m4-main / tab:m4-diag | Adam/AdEMAMix 主结果与机制结果 | (Member 4) |
| tab:synthesis | 跨任务结论 | (每人填自己的行，Member 1 合并) |
| tab:contributions | 实际独立实现和贡献 | (每人填自己的行，Member 1 合并) |

表号会随内容移动，Prompt 使用稳定的 label。主表的训练和验证指标是 CE，图像测试列是 accuracy (%)，文本测试列是 CE (nats/character)。论文里解释这一差异。所有实际填表数值来自代码目录中的原始日志与汇总 CSV。

## 编译与中间文件

```sh
sh report/build_report.sh
```

`.latexmkrc` 和脚本都把 aux、log、bbl、blg、fls、fdb_latexmk、临时 PDF 放入 `report/build`。最终 PDF 复制到 `report/main.pdf`。原始入口与文献库的备份、此次生成脚本和预览也在 build 中，不打包。不要直接运行一个把中间文件写回根目录的 pdflatex 命令。

图表放 `report/figures`。缺图时会显示明确占位框，避免因尚未运行实验而编译失败。增加图后重新编译并查看全文，避免宽表越界、参考文献未解析和超过 15 页。官方 `neurips_2025.sty` 保持不变。

## ZIP 中的路径

仓库和 ZIP 都使用 `report/` 作为报告目录，运行 `sh report/build_report.sh`。课程原始 handout 不公开，主要要求见 `docs/COURSE_REQUIREMENTS.md`。最终仍需按课程要求合并报告与 slides。
