# 旧版9750步动力学驱动的Avatar训练

服务器1独立实验目录：
`/home/s21_yhr/lzh/MyDualTalk/runs/avatar_old9750_20260911_gpu23/`。

源权重为 `runs/v3_2_1_staged_20260910_gpu23/formal/best.pt`，global_step=9750。
新实验复制为 `initial_dynamics_step009750.pt`，并核验模型权重哈希：
`b1f656f069b3b86cdb18f9c59021d84746b606c84a93dd33d5e76d1283db860e`。

这是一个新的Avatar生成实验。observer、模态适配器、标签头、teacher和完整动力学保持冻结；
训练原DualTalk生成主干及FiLM，原始音频卷积特征提取器继续冻结，后续语音层可以训练。
初始化生成主干使用 `model/dualtalk_baseline.pth`，不会恢复旧动力学或其他Avatar实验的优化器。
冻结表示权重不更新，情感状态仍按每条对话的当前观测和历史持续推进。

运行代码在该实验自己的 `code/` 中。它使用经过评测的v3.3代码快照，叠加本次生成训练开关与启动脚本，
不覆盖正在训练的v3.4代码。只允许物理GPU2、3，torchrun的两个进程分别使用这两张卡。
不停止、重启或修改已有动力学任务。

训练设置：

| 项目 | 设置 |
|---|---|
| 生成变体 | dyadic |
| 输入输出 | 每秒25个新增帧；生成器最多保留此前3秒 |
| 状态历史 | 对话内持续保留；不按生成器上下文长度重置 |
| 数据 | 8,766条DualTalk训练记录，940条验证记录 |
| 正式预算 | 30,000个优化步，每步全局32个有效新增块 |
| 随机种子 | 6666 |
| 优化器 | AdamW，初始学习率0.0001，余弦衰减 |
| 混合精度 | AMP，梯度溢出重试不计入优化步 |
| 显存控制 | 生成器梯度检查点；原始token提取器驻留CPU |
| 额外情感损失 | 关闭，避免重复计算冻结模块的训练目标 |
| best选择 | 完整验证清单的generation_total |
| 验证频率 | 正式训练每1,000步；最终保留last.pt与best.pt |

流程先用相同双卡和32块预算执行2步启动检查，只使用2条验证记录快速确认实现。
检查要求生成器梯度大于0、observer/state梯度为0，完整checkpoint内情感模块的权重哈希仍匹配9750步源模型。
通过后从同一原始初始化开始正式30,000步，不沿用启动检查的优化器或余弦学习率。
正式训练结束后，在现有test和OOD划分评测best；它们不参与选点。

主要文件：

- `pipeline.log`：各阶段切换及流水线报错。
- `smoke.log`、`smoke_gate.json`：双卡启动检查及验收。
- `formal.log`：正式训练标准输出。
- `formal/train_metrics.jsonl`：训练MSE、有效帧、梯度、速度和显存。
- `formal/metrics.jsonl`：完整验证指标。
- `formal/training_status.json`：已完成步数。
- `experiment_protocol.json`：源权重、数据清单哈希和训练约定。
- `formal_config.json`：可复核的正式训练配置。
- `formal/best.pt`、`formal/last.pt`：通过验证后保存的完整Avatar权重。

这次没有创建定时巡检。`scripts/run_avatar_old9750.py`仅负责同一训练流程的阶段衔接。
