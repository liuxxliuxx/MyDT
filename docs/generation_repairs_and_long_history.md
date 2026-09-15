# 生成监督、条件读出与长时训练修复

本次实现位于本地代码。原服务器训练没有被本次操作重启或替换。新目标需要建立新实验，旧权重可用于初始化，旧优化器不能沿用。

## 实现范围

| 部分 | 代码 | 行为 |
|---|---|---|
| 跨块损失 | `utils/generation_losses.py`、`train/generation_v3.py` | 使用上一块实际输出末帧；同会话、相同角色、物理帧连续才计数；同一TBPTT段保留两侧梯度，段后和优化步间停止梯度 |
| 视觉教师 | `models/visual_affect_teacher.py` | 独立冻结V-only副本，只接收FLAME；保持对生成动作的梯度；校准域、FLAME语义和共享坐标必须匹配 |
| 视觉监督 | `train/visual_teacher_calibration.py` | 仅校准FLAME适配器，共享affect及标签头保持固定；独立train/val来源验证；默认生成视觉损失均为0 |
| FiLM接口 | `models/conditioned_dualtalk.py` | 原参数名称与公式不变；拆出`modulation`、`encode_interaction`、`decode_interaction` |
| 条件路由 | `models/condition_router.py` | 状态产生过程和生成器条件选择分开；支持完整状态、当前affect、均值记忆、全均值、关闭FiLM、实际语义和目标派生语义诊断 |
| 选点与恢复 | `utils/checkpoint_v3.py`、`utils/generation_selection.py` | 记录新增构造、权重、损失和协议；独立保存重建best；缺少独立语义时不能产生语义优胜checkpoint |
| 条件评测 | `train/generation_condition_eval.py` | 完整因果重放；每块交互特征只编码一次；训练均值和匹配供体仅来自train |
| 连续数据 | `data/continuous_dialogues.py` | 只根据已验证的时间和角色映射连续推进；真实缺口按缺失输入推进；跨文件保留可见文本与音频历史 |
| 成熟状态训练 | `train/history_sampling.py` | 一半自然窗口、一半按非空历史年龄层和来源采样；只重放冻结上游前缀，生成器仍读当前及之前3秒 |
| 长时动力学 | `train/staged_v34/` | 复用统一动力学；增加按历史年龄采样和物理时间遮挡；可用当前权重重放完整前缀，随后按32事件截断梯度 |

## 损失与公平对照

`generation_total`仍为表情、jaw、neck、块内速度四项之和。

```
generation_total_with_boundary = generation_total + boundary_weight * boundary_velocity_mse
train_total_loss = generation_total_with_boundary + 已启用的视觉损失 + 已启用的上游辅助损失
```

边界损失匹配真实跨块速度，不要求相邻帧相同。数据mask在平方和其他非线性运算之前作用。边界与视觉窗口的分母在本优化步前规划，各卡归约一次，梯度使用SUM。无效尾帧、会话结束和物理缺口不会被错误连接。

初始比较为`film_off/full_state × boundary_weight=0/0.1`。`film_off`仍重放相同的dyadic观测和状态，只关闭生成器FiLM；它是本次“无情感生成条件”控制，不等同于200帧原始离线baseline。可增加`1/24`作为边界权重候选，不能把0.1当作已证明最优值。

```bash
python scripts/run_generation_repair_suite.py --base-config OLD_RUN/config.json --initialize COMMON_AVATAR.pt --output runs/repair_diagnostic --steps 1000
```

默认只生成配置和`suite.json`，不占用显卡。增加`--execute`才按顺序执行，每组由`torchrun --nproc_per_node=2`启动，子进程限定`CUDA_VISIBLE_DEVICES=2,3`。默认种子为6666、6667、6668，每步32个有效新增块，双卡每卡4个对话、每对话4块。使用同一个完整Avatar初始化文件、相同数据顺序和优化预算。

正式实验把`--steps`设为30000。诊断和正式训练都是新实验，不将不同目标的优化器状态混合恢复。新输出目录应与正在训练的目录分开。已有预加载、延迟指标统计、冻结卷积和分组学习率策略继续保留。

其他路由由`--routes`选择：

| 路由 | 输入变化 | 使用前提 |
|---|---|---|
| `affect_only` | 保留双方当前affect，持续状态和relation为0 | 与full_state共用观测和状态推进 |
| `mean_memory` | 当前affect不变，持续状态与relation换成训练均值 | 匹配当前observer和state权重的train均值 |
| `mean` | 全部FiLM条件换成训练均值 | 同上 |
| `actual_semantic` | 当前Avatar状态经固定标签头读出，再由投影器转为FiLM条件 | 通过视觉校准的有效语义头 |
| `oracle_visual_pseudo` | 当前真实目标FLAME经视觉教师读出，再经相同投影器 | 仅用于诊断；禁止普通流式部署 |

实际语义与oracle共享种子、投影结构、有效头和合格目标窗口。低于`min_visual_frames`的尾块继续推进状态，但两组都不计监督与有效块预算。oracle的无效窗口使用明确的零覆盖值。oracle不能使用含目标的长历史预热；脚本会把这一组合标为不可执行。低维oracle不代表严格的情感能力上界。

## 视觉教师、人工标签与坐标

```bash
python scripts/prepare_visual_calibration_data.py --templates-only --output runs/visual_calibration
python scripts/prepare_visual_calibration_data.py --config OLD_RUN/config.json --selection train_val_windows.json --output runs/visual_calibration
```

选择文件包含`train`和`val`两个键，每项有`dataset_digest`及`windows`，窗口记录`name/start/end`，第一版窗口至多1秒。导出的标签全部留空，`annotation_origin=unannotated`。人工确认后才填`human_manual`、`annotator_ids`及有效标签。工具拒绝test/OOD作为校准来源。

类别顺序为`angry, disgusted, fear, happy, neutral, sad, surprise`。VAD需要明确归一化到[-1,1]并逐维提供mask。人工强度若只是0–4等级，不能直接用于教师强度MSE；没有明确的强度尺度校准时保留缺失。

```bash
python scripts/calibrate_visual_affect_teacher.py --checkpoint COMMON_AVATAR.pt --train-manifest runs/visual_calibration/train.json --val-manifest runs/visual_calibration/val.json --output runs/visual_teacher --steps 500 --device cuda:0
```

校准保持affect坐标和标签头固定，只训练声明域的FLAME适配器。校验包含真实标签表现、训练多数类/均值对照、均值动作响应、不同情感动作的读出方向、有限非零输入梯度。通过的头及VAD维度分别记录。失败只保存`candidate_unvalidated.pt`；没有人工标签则输出待标注状态。

通过之后，使用`--visual-teacher teacher.pt --visual-losses-json visual_losses.json`同时给所有对照增加相同损失。示例配置如下；数值是实验候选，不是验收结果：

```json
{
  "visual_feature_weight": 0.1,
  "visual_class_distill_weight": 0.1,
  "visual_vad_distill_weight": 0.0,
  "visual_intensity_distill_weight": 0.0,
  "temperature": 2.0,
  "min_visual_frames": 12,
  "vad_dimensions": [0]
}
```

日志中的`visual_*_loss`是蒸馏代理指标，不能作为人工情感改善结论。真实FLAME语义必须匹配`flame56-expression50-jaw3-neck3-native-v1`。教师权重hash、训练域、校准结果和共享坐标hash都会检查；不能因为输入同为56维就换用不同归一化或不同标签头。

## 增加长时训练机会

```bash
python scripts/audit_long_history.py --config DYNAMICS_CONFIG.json --split train --output runs/history_coverage.json
python scripts/audit_long_history.py --config OLD_RUN/config.json --split train --avatar-timeline-template --output runs/train_timeline.template.json
```

审计分别报告历史年龄`a`和未来查询时距`h`。原始数据审计的向量数量是时间戳候选，经过教师编码后的bank报告有效向量起点。真实标签端点单独计数，并列出16/32/64秒的事件—后续partner标签覆盖；这种覆盖不能解释为人工标注的因果影响。

连续映射的协议为`verified-continuous-dialogues-v1`，包含`split/token_manifest_digest/conversations`。每个会话必须有`session_id/source_id/roles/segments`；每段必须有`name/start/end/roles/verified/evidence`。时间必须落在25FPS物理帧上。模板保持未验证，不按文件编号自动接成长对话。

训练配置中增加：

```json
{
  "data": {"continuous_timelines": {"train": "train_timeline.json", "val": "val_timeline.json"}},
  "long_history": {"enabled": true, "age_edges": [0,8,16,32,64], "natural_probability": 0.5}
}
```

也可先对现有真实片段使用`--long-history`，覆盖不足的年龄层会记为0。重采样不能增加独立数据量。生成训练首块如果只有观测前缀而没有实际已生成末帧，首边界不计损失。

```bash
python scripts/prepare_long_dynamics.py --base-config DYNAMICS_CONFIG.json --initialize DYNAMICS.pt --output-config runs/long_dynamics.json --run-directory runs/long_dynamics
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --standalone --nproc_per_node=2 -m emotion_ssm.train.staged_v34.trainer --config runs/long_dynamics.json
```

新配置启用`full_origin_replay`：每步对选中的对话共享重放完整前缀，传播训练的起点停止梯度；更新阶段只给最近32事件保留图。锁定已有校正速率、event/coupling强度及统一坐标约束。旧起点bank不能替代当前权重重放。

新动力学训练完成后，先用显式工具构造新的完整Avatar初始化文件：

```bash
python scripts/prepare_avatar_from_dynamics.py --avatar OLD_AVATAR.pt --dynamics NEW_DYNAMICS.pt --output runs/new_upstream_avatar.pt
python scripts/run_generation_repair_suite.py --base-config runs/new_upstream_avatar.json --initialize runs/new_upstream_avatar.pt --output runs/new_upstream_generation --long-history
```

该步骤保留原生成主干，替换为新observer/teacher/动力学，检查特征来源，清除旧训练均值和视觉校准声明，将FiLM初始化为恒等变换，不恢复优化器。随后所有对照共享该初始化。完整checkpoint的日常恢复仍禁止加载后再用外部动力学覆盖。

`train_context`在同一对话的所有重叠缓存中使用同一物理遮挡计划：约80%的遮挡类型是连续多模态时间段，约20%是指定周期内的整个单模态缺失；干净教师目标保持不变。配置按训练阶段循环`clean/clean/clean/train_context`，不同阶段长度可能不同，不能把它写成严格75%的优化样本比例。音频、韵律、文本及视觉都按真实依赖时间屏蔽；不兼容的上下文特征缓存会报错并要求重建。

## 可复用条件与长历史评测

固定清单格式为`{"split":"train或val","dataset_digest":"...","names":[...],"score_start_seconds":4}`。先在train上形成均值与完整供体轨迹，再评测固定清单：

```bash
python scripts/evaluate_generation_conditions.py --checkpoint AVATAR.pt --manifest train_panel.json --output runs/conditions_train.pt --device cuda:0
python scripts/evaluate_generation_conditions.py --checkpoint AVATAR.pt --manifest val_panel.json --bank runs/conditions_train.pt --output runs/conditions_val.json --history-ablations --device cuda:0
```

供体依据已可见的前4秒观测匹配，禁止同来源，评分从匹配前缀后开始。匹配供体长度不足时报告覆盖率，不对缺失候选计成功。`paired_true_on_matched_coverage`提供相同覆盖范围内的真实条件重建指标。

`--history-ablations`增加最近4/16/32秒持续状态，以及从对话开始关闭双方耦合的重放。它们共享当前因果观测，观测本身可能包含16秒音频，因此这些是持续状态消融，不能称为完全无历史。关闭耦合会重新推进动力学，不只是隐藏解码器partner字段。

报告包括部位/速度/边界MSE、条件差异、FiLM gamma/beta及其时间变化、交互特征变化、生成输出响应、可用时的视觉教师响应。错配条件下与原目标的MSE只表示敏感度，不证明错配回应是否正确。对话级分子、分母和来源保留在报告中，用于配对统计和多种子汇总。

## checkpoint与验收

`best_reconstruction.pt`按原`generation_total`保存。`best.pt`按事先声明的`selection_score`保存。`last.pt`包含边界状态、采样RNG、优化器、损失配置、路由、投影器、均值来源及完整视觉教师权重；恢复不调用外部教师初始化文件。

`best_conditioned.pt`需要独立验证报告，使用：

`reconstruction_frontier.json`另外保留验证集重建指标的非支配候选，不把代理分数解释为人工语义优势。它是指标记录，只有best策略保留的checkpoint保证有对应权重文件。

```bash
python scripts/select_conditioned_checkpoint.py --checkpoint CANDIDATE.pt --reference reference_val.json --independent-report independent_val.json --output runs/accepted
```

训练前的`generation_selection.conditioned_policy`必须声明`maximum_relative_regression`中的边界、jaw、neck、expression四个容忍值，以及`semantic_metric/minimum_improvement/population_sha256`。独立报告必须与checkpoint SHA、验证来源和配对参考匹配。蒸馏教师内部得分不能触发语义优胜保存。使用`--selection-json`把该策略加入对照配置。

新增测试见`tests/test_generation_repairs.py`与`tests/test_generation_long_history.py`。验收记录和数据缺口见`runs/generation_repairs_20260913/`。完整效果实验尚未执行，不能据代码测试声称情感指标或MSE已超过无情感条件控制。
