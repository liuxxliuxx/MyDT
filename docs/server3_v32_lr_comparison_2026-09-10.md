服务器3同卡学习率对照（2026-09-10）

按用户要求，保留原 `1e-4` 任务，在同一GPU0上另开 `3e-5` 任务。新任务于20:54:06启动；20:55:07核验到10010步，两组实际学习率均为 `3e-5`，无报错。原任务同时推进到11580步，PID、进程启动标识、配置及代码SHA256均保持不变。

| 项目 | 原任务 | 新任务 |
| --- | --- | --- |
| 运行目录名 | v3_2_adaptive_flow_extend20k_20260910_gpu0 | v3_2_adaptive_flow_lr3e5_20260910_gpu0 |
| 动力学核心学习率 | 1e-4 | 3e-5 |
| event/action头学习率 | 1e-4 | 3e-5 |
| 分支起点 | 原10000步last | 同一个原10000步last |
| 总停止步数 | 20000 | 20000 |
| 训练进程PID | 2831788 | 3045833 |
| 每步新块数 | 32 | 32 |
| 随机种子 | 6666 | 6666 |

共同起点checkpoint的SHA256为 `321dd8e962e35417a8a5c5bfd47aa3905be0c01ba3018e6da3fdb3e77572275a`。新任务读取已保存的10000步断点，未从原任务当前的一万多步权重初始化。模型、损失、模态缺失策略、预测时距、数据顺序、验证子集和编译设置保持相同。AdamW动量和步数、持续状态、数据游标、随机数以及训练均值累计量一并恢复；随后显式改变两组优化器学习率。此次是学习率分支实验，权重配置中记录 `learning_rate_branch`。

入口新增 `--resume-lr`。只修改外部JSON不能改变完整恢复后的学习率，因为配置和优化器参数组都来自checkpoint。实现先恢复优化器状态，再覆盖组学习率；同时更新保存配置中的 `train.lr` 和 `train.state_lr`。不改变未用于该阶段的 `observer_lr`。新学习率必须是有限正数，并要求显式指定不同的输出目录，避免将学习率分支写回原实验。

```bash
python -m emotion_ssm.train.dynamics_v3 \
  --config /path/to/original/config.json \
  --resume /path/to/original/last_10000.pt \
  --resume-until-step 20000 \
  --resume-output /path/to/lr3e5_branch \
  --resume-lr 3e-5
```

首次恢复后写出 `resume_receipt.json`，记录源学习率、实际学习率和源优化器步数。训练日志也记录每组 `learning_rates`。新任务完成首次保存后，再从其自身完整checkpoint普通恢复时不需重复传入学习率参数，保存的 `3e-5` 会继续生效。

新任务目录：

```text
/data/lzh/MyDualTalk/runs/v3_2_adaptive_flow_lr3e5_20260910_gpu0
```

- `dynamics.log`：训练日志。
- `training/seed6666/dynamics/metrics.jsonl`：验证及训练指标。
- `training/seed6666/dynamics/resume_receipt.json`：实际优化器学习率恢复记录。
- `training/seed6666/dynamics/last.pt`、`best.pt`：断点及最佳权重，每250步更新。
- `source/last.pt`：未改写的共同10000步起点。
- `comparison_protocol.json`、`launch_health.json`：对照协议和原任务未改动的核验。
- `code`：独立源码副本，仅训练入口与对应测试相对原任务变化。

新目录携带原10000步之前的指标及7750步历史最佳权重。首次新保存前，输出目录的 `last.pt` 仍是源10000步文件，此时查询当前学习率应读取 `resume_receipt.json` 或新训练日志中的 `learning_rates`，不能把继承的旧断点学习率当成当前优化器学习率。首个新增验证在10250步进行；启动时尚未得到新的效果结论。

本地和服务器各通过44项测试。学习率分支测试与“手工仅修改源优化器参数组学习率的参考断点”完成同一下一步训练后，权重、优化器、持续状态、随机数及指标完全一致；后续普通恢复仍保留新学习率，源断点未被改写。数值一致性测试使用CPU，实际GPU启动则核验了源优化器步数为10000、实际学习率为3e-5并成功推进。

两个进程共享GPU计算资源。后续效果比较应按相同优化步数及验证协议进行；不能按相同墙钟时刻比较。原验证起点随候选模型变化，判断传播器本身是否改善仍需要共同冻结起点的评测。没有创建定时巡检，也没有修改服务器1。
