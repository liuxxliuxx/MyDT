统一情感 Avatar v3 训练说明（更新于 2026-09-09）

当前新实验默认 v3.2：跨维度、状态相关的演化和持续双人反馈，以及以片段遮挡为主的混合遮挡。具体方程、配置、测试和重训边界见 [v3.2 修改说明](v32_adaptive_dynamics_2026-09-09.md)。下文 v3.1.2/v3.1.3 小节记录旧实验协议；旧结果仍按旧构造评测。

本版根据历史和当前双方输入学习连续 FLAME 表达，不要求新增离散的“安慰/关切”策略标签。模块名继续使用 `_v3.py`，数据协议为 `emotion-token-packets-v3.1`，checkpoint 格式为 4。旧 v2/v3 实验的代码包和 checkpoint 保留，必须使用原协议解释；不能恢复旧优化器后混称同一次实验。

核心文件与职责：

| 文件 | 职责 |
|---|---|
| `emotion_ssm/data/packets_v3.py` | 一秒双人包、真实时间、角色、token mask、句末标签、DualTalk 目标隔离 |
| `emotion_ssm/preprocess/tokens_v3.py` | 局部音频 token、无上下文词嵌入、AU/FLAME 独立来源和缓存版本 |
| `emotion_ssm/models/token_observer.py` | 上下文编码前遮挡、共享 affect 读出、相对角色查询、事件与行为 |
| `emotion_ssm/models/state_core.py` | 唯一快慢状态更新、定向关系、自主演化、条件预测和纯衰减对照 |
| `emotion_ssm/models/streaming_v3.py` | 25 新帧、最多过去 75 帧、持久会话状态、事件去重 |
| `emotion_ssm/train/observation_v3.py` | A0 遮挡/情感监督和 FLAME 校准 |
| `emotion_ssm/train/dynamics_v3.py` | 完整对话顺序、真实秒数预测、固定教师坐标保持 |
| `emotion_ssm/train/generation_v3.py` | 四种对照、连续 TBPTT、表达和未来损失联合反传 |
| `emotion_ssm/utils/checkpoint_v3.py` | 自包含工厂、权重/配置/特征来源/优化器恢复 |

输入先形成局部 token，再在观测器内部遮挡。音频以独立 500 ms 为依赖单元，使用完整冻结的预训练音频骨干，并按有效输出长度池化。遮挡时整个依赖单元及其韵律特征一起遮挡。每个单元保留原始波形的 8 维能量、幅度和过零率等统计，补充局部标准化丢失的信息；其中没有基频估计。观测器保留截至当前的最近 16 秒音频，生成主干的原始输入仍限于当前秒和此前 3 秒。available 与 fresh mask 分开，历史音频不会反复产生新动作或事件。超过历史窗口的句子在质量报告中单独计数。

文本使用词嵌入、独立 token 位置和可用时间，保留语序且不读取未来台词。AU 35 维和 FLAME 56 维经过不同适配器，再进入同一个情感读出。Avatar 当前目标 FLAME 始终只提供生成监督。

A0 的表征约束在关闭 dropout、保留梯度的路径上计算，直接约束归一化后的 affect，并同时统计每域和全局样本。新增固定投影摘要重建必须经过 affect 瓶颈，保留 EMA 教师约束及原 token 重建辅助项。token 重建下降不能单独作为 affect 有效的证据。A0 每批混合多个对话和角色，在该模态有标签可用时至少一半为有效句末监督；类别权重、常量分类器和连续标签均值只从 train 计算。分布式监督按全局有效标签数归一化。

首次 A0 启动需要建立全量句末索引。CPU 索引完成并原子发布后才初始化分布式进程组，其他 rank 在共享文件系统等待，避免长扫描占用 NCCL 的 collective 超时窗口。每个数据来源的索引缓存绑定 manifest、训练文件大小/修改时间及索引版本，后续种子和校准可复用；日志每30秒报告构建或等待进度。

双方状态各有同坐标的 fast、slow 和 baseline；关系保存两个方向。v3.2 在每次积分时根据双方状态和关系计算衰减率、跨维度作用和持续 partner 消息，无新事件时也继续相互反馈。旧 v3.1.x 每维自主矩阵 `A=[[-rf,w],[-w,-rs]]` 仅保留在旧构造读取路径。无未来输入预测只接收起点状态和查询秒数，真实未来 token 仅作为停止梯度的目标。条件预测必须显式声明 planned 或 oracle_conditional，禁止未来 affect 校正。

慢状态的观测校正率有每秒上限；历史文本存在、当前新观测、事件存在和新行为持续时间分别记录。相同事件不会每秒重新注入。静音下有效视觉仍可提供校正和双方影响。

生成阶段从第一步训练原 DualTalk 主干，保留原波形卷积冻结。observer 使用低学习率和固定教师约束，状态核心同时接受表达及未来预测梯度。生成器使用激活重计算降低连续序列显存。梯度只在 TBPTT 或优化器边界截断，数值状态继续保留；不会每秒 detach。

四组对照为 none、affect、self、dyadic，均输出当前 25 帧并读取相同的原始输入窗口。self 与 dyadic 保留相同观测输入，区别为是否启用双方耦合及关系条件。每优化步共 32 个有效新块，两卡各 16 块，最大 TBPTT 为 32 秒。额外观测器和状态参数、上游训练费用需要单独报告。

none 仅训练生成器，不优化无用的情感辅助目标；affect 训练当前观测及坐标保持，不训练持久状态。self/dyadic 联合优化未来预测。生成器、观测器、状态核心分别裁剪梯度，各对照使用同一个生成器裁剪阈值，避免新增情感参数间接缩小生成主干梯度。持久状态至少使用 FP32 保存，即使观测编码和生成器启用了混合精度，也不会每秒把微小慢变化量化掉。

服务器 1 启动方式：

```bash
cd /home/s21_yhr/lzh/MyDualTalk
export CUDA_VISIBLE_DEVICES=GPU-5b497823-4a84-bde7-5670-2172ee96245d,GPU-9117039b-5194-5d46-d4a7-088ff06ce552
/home/s21_yhr/miniconda3/envs/lzh_DDG/bin/python -B scripts/server1_v3_pipeline.py --run-id v3_1_retrain_20260908_gpu23
```

脚本先并行预处理 EmotionTalk/IEMOCAP，再在两卡分片预处理 DualTalk，然后运行 A0。A0 best checkpoint 必须通过独立放行检查，检查 A、AT、AVT 的分类分布、对 train 常量对照的情感指标及确定性 affect 表征；不合格时停止，不能自动把单一类别的观测器送入下游。通过后运行 FLAME 校准、动力学、四组各 1000 步诊断。诊断通过后，各组独立训练 30000 步，种子 6666、6667、6668。`--smoke` 使用小数据和每阶段两步验证运行链路，显式跳过情感效果门槛；这不能作为模型效果证据，也不能复用其小数据缓存作正式训练。`--pilot-only` 只完成 1000 步诊断。

`--source iemocap` 或 `--source emotiontalk` 生成单数据源上游对照，自动绑定 DualTalk 的对应音频／文本提取器。校准时根据已训练 checkpoint 的特征来源与 revision，复制有标签域的音频、韵律和文本适配器到 domain 2，再固定 A/T 教师校准 FLAME。默认三域配置使用 IEMOCAP domain 1；不能按 768 维相同就认为来源兼容。

外层日志和状态在 `runs/<run-id>/*.log`、`status.json`；每个阶段的权重、训练曲线、验证结果在 `runs/<run-id>/training/seed<seed>/`。原始数据和旧实验不移动。代码同步使用 `scripts/deploy_v3.py` 的哈希清单，覆盖前保存旧文件。

数据限制需要保留在报告中：

- EmotionTalk 只使用有效 valence 和独立 intensity，不伪造 arousal/dominance。IEMOCAP 使用真实 VAD，无法确认身份的 AU 保持缺失。
- IEMOCAP 的 `(arousal + 1) / 2` 是候选匹配代理，不能当作独立 intensity 监督。`intensity_mask`、`intensity_valid` 的显式缺失会保留；旧 IE 标签没有来源字段时也默认缺失。额外的独立强度标注必须同时声明 `intensity_source=independent_annotation` 和有效 mask。
- 句子级标签只在真实终点监督，不复制成每秒真值。未来教师只读取目标时刻已可见的信息。
- DualTalk 复用通过检查的词级对齐；旧预处理报告有 6572 条文本流对齐失败记录（5790 条低置信、782 条不支持符号，报告共处理 5763 对）。这些文本继续标记缺失，不补写台词。每次 v3 预处理另报实际文本覆盖率。
- DualTalk 没有可靠跨文件映射时，每个文件独立作为对话；文件编号不能充当连续时间。短片段测试不能证明分钟级情感记忆。
- 验证只从原 train 来源划出，test 与 OOD 不参与选 best。生成 best 按有效元素加权的 `generation_total` 选择。

独立评测可用 `python -m emotion_ssm.evaluate_v3 --checkpoint <best.pt> --split test --output <report.json>`，可通过 torchrun 两卡运行。指标保留每对话 SSE/有效元素数，支持后续配对比较和多种子汇总。代码通过、训练能稳定运行、独立情感有用以及表达优于 none 是不同的验收项，不能互相替代。

IE 缓存需要 `supervision_revision=endpoint-intensity-mask-v3.1.1`。新预处理把监督解释版本写入每个 dialogue 的缓存指纹；相同原始标签配合不同解释规则不能静默复用。已有特征可用 `scripts/repair_v31_supervision.py --source-root <旧IE-token-root> --output-root <新IE-token-root>` 迁移。工具逐条核对 arousal 代理公式，保留音频、文本、视觉张量、时间、角色、VAD 和类别，仅关闭代理 intensity 监督并补充来源；原始缓存不改写，目标缓存绑定来源 SHA256 与新监督版本。

迁移后使用新 run-id 重训模型和优化器。`scripts/server1_v3_pipeline.py` 支持对每个数据源传入 `--reuse-token-root dataset=path`：ET、DualTalk 可直接引用原有完整特征根目录，IE 引用迁移后的目录。必须显式提供所有选定来源，仍执行数据版本、完整性和正式数据规模检查。此参数只复用特征，不加载旧上游权重或优化器；正式 A0 质量门槛和四组训练预算保持原设置。
# V3.1.2 动力学训练修复

`train.dynamics_revision=v3.1.2-state-memory-forecast` 固定共享情感观测器的特征、融合和标签头，只训练 event/action 与状态核心。观测器保持 eval 模式，event/action 仍接收梯度。状态送入 A0 标签头前单位化；近零状态输出零读出，不改变状态积分本身。未来训练误差按每个 affect 向量的平方 L2 计权，验证仍报告逐元素 SSE/count MSE。原始训练预测 MSE 与含标签的复合 future loss 分开记录。

每段 TBPTT 开始时，训练器显式重绑当前可训练 baseline，保留 fast、slow、relation 和事件去重状态。生成阶段仅 self/dyadic 的可训练状态核心执行重绑，推理的自定义 baseline 保持原行为。日志增加 fast/slow/affect 模长、fast-slow 余弦、baseline 梯度与实际每卡梯度段上限，用于识别快慢态抵消。

本次保留事件、慢态和双人耦合，不加入每秒强制状态贴合教师的目标。训练修复是否改善长期预测，需要后续独立验证；不能由测试通过推断优于 none。

旧动力学/生成 checkpoint 可读取和推理，但不能恢复到此次改变梯度语义的优化器。新实验可用 `scripts/server1_v3_pipeline.py --reuse-observation-calibration 6666=旧实验/training/seed6666` 复用完成训练且通过原始 A0 gate 的同种子 A0/校准导出。入口核对模型、特征 manifest SHA、预算和原始门槛，在新目录保存独立副本及 `reused_upstream.json`；动力学与生成重新初始化优化器，未指定种子从上游重新训练。

V3.1.3 uses gold_endpoint_latest_complete_origin_v1 for direct future gold supervision: latest complete origin t <= e-h, forecast actual e-t seconds without future observations. Existing exact-grid teacher targets and best selection remain. future_endpoint_unique metrics average horizons per real endpoint; duplicate role/end labels raise errors. V3.1.2 evaluation keeps its original unit readout and grid protocol. New dynamics requires a fresh optimizer; validated same-seed A0/calibration may be reused.
