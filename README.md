# PPO-HK 多维意见软共识实验代码

本仓库提供论文所展示实验的环境、PPO训练、确定性评估、指标统计和绘图代码。
模型采用个体干预与HK同步更新交替执行的方式，以总方差准则判定软共识，
并保留终态归一化重心偏移约束。软共识不等同于所有意见完全一致。

仓库不包含训练好的模型。`paper_results/`是论文图表对应的归档数值结果，
可以直接用于核验与重画；重新训练得到的模型只保存在本地`runs/`目录。

## 目录

```text
consensus_hk/       环境、指标、奖励、PPO及实验入口
configs/           论文参数与实验协议
paper_results/     正文展示结果的CSV及来源校验记录
tests/             环境、指标、基线和协议单元测试
tools/             发布前文件检查
docs/              章节对应、复现说明、发布检查清单
```

## 安装与检查

在仓库根目录执行。推荐使用Python 3.9的独立环境；依赖版本见`requirements.txt`。
这些版本对应整理代码时核验的运行环境，不表示所有依赖均有历史安装记录。

```bash
python -m venv .venv
```

Windows PowerShell激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS激活环境：

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python tools/check_release.py
```



## 核验论文结果与重画图表

```bash
python -m consensus_hk.plot --section all --output outputs/paper_figures
python -m consensus_hk.tables --source paper_results/section_35_ablation.csv --output outputs/ablation_table
python -m consensus_hk.tables --source paper_results/section_36_sensitivity.csv --output outputs/sensitivity_table
python -m consensus_hk.experts --output outputs/typical_experts
```

绘图需要可用的中文字体，优先采用宋体，并自动选择Times New Roman或serif西文字体。
若系统未安装中文字体，可通过`--font /path/to/font.ttf`指定已有字体。
仓库不分发字体文件。PNG默认450 dpi，PDF采用矢量线条和嵌入字体。
归档数据来源、筛选规则与SHA256见`paper_results/provenance.json`。
绘图不执行策略评估，也不更改原始CSV。

## 主模型训练

```bash
python -m consensus_hk.train --configuration main
```

默认训练种子42，预算10,000,000个交互步，单环境、2048步rollout。
SB3完成最后一个rollout，因此实际步数为10,000,384。
学习率采用原实验的`3e-4 * progress_remaining`线性schedule。
环境每回合独立生成均匀初始意见，再采样`epsilon ~ Uniform(0.05,0.30)`；
信任阈值在回合内保持不变。没有按评估结果选择checkpoint的步骤。

运行中断时会保存`interrupted_model.zip`，每500,000步另保存检查点。
这些文件不属于发布内容；本入口不承诺从检查点恢复完整随机状态或逐位相同的训练轨迹。
`--timesteps`可用于短小程序检查，但非10M预算的输出不能作为论文实验结果。
已有输出目录不会被覆盖。

## 论文中展示的两个消融变体

```bash
python -m consensus_hk.train --configuration no_mask
python -m consensus_hk.train --configuration no_gravity_penalty
python -m consensus_hk.evaluate --section ablation --output outputs/ablation
```

- `no_mask`仅关闭边界方向动作掩码，保留动作缩放、意见截断、全部奖励和联合终态判据。
- `no_gravity_penalty`仅将重心偏移惩罚权重设为0，仍以`V <= 0.05`且`G <= 0.45`判定成功。

这两项变体均从头训练。完整模型在消融评估中复用`main`，不再训练。


## 序数反转惩罚权重敏感性分析

```bash
python -m consensus_hk.train --configuration lambda_rev_0p5
python -m consensus_hk.train --configuration lambda_rev_1p5
python -m consensus_hk.train --configuration lambda_rev_2
python -m consensus_hk.evaluate --section sensitivity --output outputs/sensitivity
```

权重1.0复用主模型。四个权重均采用训练种子42，按相同20个评估种子在三个信任阈值下测试。
评估标准差反映初始状态之间的差异，不表示不同训练种子之间的不确定性。

## 不同专家规模与方案维度

```bash
python -m consensus_hk.train --configuration N_50_M_5
python -m consensus_hk.train --configuration N_200_M_5
python -m consensus_hk.train --configuration N_100_M_3
python -m consensus_hk.train --configuration N_100_M_10
python -m consensus_hk.evaluate --section scalability --output outputs/scalability
```

基准`N=100,M=5`复用主模型，其余规模独立训练相应策略。
每个规模--信任阈值组合仅评估一个seed 42回合；折合轮次为干预步数除以N，
不表示计算耗时，也不表示单一策略跨规模零样本迁移。

## 典型案例、基线与泛化测试

下列主模型评估命令要求先完成`main`训练。

```bash
python -m consensus_hk.evaluate --section case --output outputs/case
python -m consensus_hk.evaluate --section ordinal --output outputs/ordinal
python -m consensus_hk.evaluate --section baselines --output outputs/baselines
python -m consensus_hk.evaluate --section stability --output outputs/stability
python -m consensus_hk.evaluate --section epsilon_ood --output outputs/epsilon_ood
python -m consensus_hk.evaluate --section distribution_ood --output outputs/distribution_ood
```

仅运行五种自然演化基线时，无需模型：

```bash
python -m consensus_hk.evaluate --section baselines --natural-only --output outputs/natural_baselines
```

基线仅包含经典HK、SCOD、逐边自适应信任HK、噪声HK和惯性HK。
参数与多维适配说明见`configs/paper_protocol.json`和`consensus_hk/baselines.py`。

## 随机生成器与统计口径

- 第3.3节及3.4.2的典型案例使用`RandomStatelt_rng((42)`生成同一初始矩阵。
- 其他评估使用`defauseed)`。同一个种子编号在两种生成器下不对应相同初态。
- 配对评估种子为10000--10019；同分布稳定性测试为10000--10099。
- 阈值OOD只测试0.03和0.35；分布OOD只包括正文展示的四种分布，并在六个训练区间内阈值下测试。
- 成功步数/轮次仅统计联合成功回合；终态V、G和OPRR统计全部回合，包括失败回合。
- 标准差采用样本标准差`ddof=1`；只有一个成功回合时不计算标准差。
- OPRR以百分数表示，仅严格反向排序计为反转，终态并列不计为反转。

详细章节对应见[docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md)，
复现边界见[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。




