这份文档仅审查 2026-09-06 本地工作区的新版代码，不是服务器2旧版 all_three 实验的归因结论。实际服务器诊断见 [服务器审查报告](emotion_training_audit_2026-09-06.md)。

服务器核验已更正三点：服务器2旧版生成器从第1轮训练；旧版评测没有加载完整 checkpoint 后再次覆盖 observer；旧版 observer 默认 domain 1，因此单数据源的适配器问题发生在 EmotionTalk-only。下文第1–3项只适用于新版脚本。服务器1 fold5_default200 使用新版，其第7轮快照确实处于生成器冻结期。两个服务器的 Phase B checkpoint 和模型结构也不同，不能混用指标。

以下内容保留作为新版实现的独立静态检查。远端数据核验已经在主报告补充；“本地缺少数据”不代表没有进行服务器只读检查。

**1. 首要问题：3 轮对照并没有提供相同的优化机会。**

证据：

- `scripts/run_data_source_ablations.sh:46` 设置 `TRAIN.STOP_AFTER_EPOCHS 3`。
- `configs/dualtalk_conditioned.yaml:25` 和 `:26` 将状态模型、生成器冻结期都设为 10 轮。
- `emotion_ssm/train/dualtalk.py:98` 在冻结状态模型时仍训练 domain 2 音频适配器。
- `emotion_ssm/train/dualtalk.py:281` 冻结整个 baseline，只在冻结期结束后开放 synthesis head。
- `emotion_ssm/models/conditioned_dualtalk.py:68` 至 `:92` 对 joint encoder、temporal enhancer、interaction module 使用 `torch.no_grad()`，所以它们在当前条件模型训练入口中始终不能微调。
- `emotion_ssm/train/dualtalk_baseline_control.py:139` 至 `:142` 直接训练 `wrapper.baseline`。原模型只冻结了两路 Wav2Vec2 的卷积 feature extractor，其 Transformer、投影层、交互模块和合成模块都可接受梯度。

| 比较项 | baseline control | 3 轮情感模型 |
| --- | --- | --- |
| 生成器主干 | 从第 1 轮微调，卷积特征提取器除外 | 全程冻结 |
| synthesis head | 从第 1 轮微调 | 全程冻结 |
| 新情感模块 | 无 | FiLM、projector、domain 2 音频适配器训练 |
| 情感 SSM | 无 | 全程冻结 |
| 每个 dataset item | 一个 chunk | 一段对话的全部 chunk |

baseline 的 3 轮直接优化重建，情感模型的 3 轮主要学习在固定生成器上施加调制。训练 epoch 数相等也不代表 optimizer step 数相等：两个训练入口分别使用 `DualTalkChunkDataset` 和 `DualTalkDialogueDataset`。同样的 batch size 对应的 chunk 数不同，必须记录实际 `global_step`、处理帧数和显存/耗时。

建议先建立同一训练入口的对照：相同初始化、数据顺序、chunk、可训练生成器模块、学习率、optimizer step 数和重建损失；只切换情感条件。若使用 synthesis-only 微调，两组都采用 synthesis-only。若使用全主干微调，条件模型中的 `no_grad` 也必须调整，单独把 `FREEZE_BASELINE_EPOCHS` 改为 0 不够。

**2. 单数据源评测会把另一组情感编码器覆盖进来。**

`scripts/run_data_source_ablations.sh:54` 的 `evaluate_source` 只传入 conditioned checkpoint、baseline checkpoint、split 和 output，没有传入该 source 对应的 A0/Phase B 路径。默认 YAML 的 Phase B 路径是 `runs/phase_b_coupling/fold5/phase_b_best.pt`。

`emotion_ssm/evaluate_dualtalk.py:205` 先建立系统，再在 `:206` 加载单数据源 conditioned checkpoint；紧接着 `:207` 调用 `restore_shared_observer`，从默认 Phase B 路径重新加载情感编码器。

`emotion_ssm/train/dualtalk.py:265` 的恢复函数只保留已经训练的 domain 2 音频适配器，然后覆盖其余 observation encoder。结果是单数据源 adapter/SSM/FiLM 与默认混合数据 encoder 拼接。形状一致使 `strict=True` 无法发现这种语义坐标不匹配。

这是评测正确性错误。影响方向需要重评确定，不能假设必然提高或降低 MSE。对现有 checkpoint，可先显式传入训练时同源的 `TRAIN.OBSERVATION_CHECKPOINT`、`TRAIN.PHASE_B_CHECKPOINT`、`DUALTALK.PHASE_B_CHECKPOINT` 并重评；长期应优先从 conditioned checkpoint 自带的完整权重恢复，不应依赖外部模型覆盖内部组件。确有旧版本兼容需求时应使用明确版本标记和来源校验。

评测还需恢复训练时的 `CHUNK_FRAMES`、`CAUSAL_STATE_CONTEXT` 等非参数配置。目前评测入口不自动采用 checkpoint 内保存的 config，旧模型的权重即使可以加载，也不代表其推理行为被正确恢复。

**3. IEMOCAP-only 使用了错误的适配器来源。**

`emotion_ssm/train/common.py:26` 附近将 EmotionTalk 固定映射到 domain 0，IEMOCAP 固定映射到 domain 1。`configs/dualtalk_conditioned.yaml:31` 的 `ADAPTER_INIT_SOURCE` 固定为 0，单数据源训练脚本没有覆盖它。

`emotion_ssm/train/dualtalk.py:563` 将 source adapter 复制到 DualTalk 的 domain 2。因此 IEMOCAP-only 复制的是没有接受 IEMOCAP 数据训练的 domain 0，而不是已训练的 domain 1。即使未被选中的适配器可能受优化器权重衰减影响，它也没有获得该数据的任务学习信号。

EmotionTalk-only 应使用 source 0，IEMOCAP-only 应使用 source 1；混合数据模型的初始化来源需要验证。修复评测覆盖不会修复已经从错误来源初始化并训练的 domain 2；IEMOCAP-only 的 DualTalk 适配阶段需要重新训练。

**4. 当前情感一致性损失没有独立的真实表情锚点，并且参与选择 best checkpoint。**

`emotion_ssm/train/dualtalk.py:408` 计算：

`L_state = 1 - cosine(projector(generated), target_aff)`

projector 是随机初始化、与生成器一起优化的小 MLP；它对 56 维生成参数做时间平均后投影到 128 维。此训练目标没有要求 `projector(真实表情)` 对应可信的情感标签或固定教师特征。`target_aff` 也没有 detach；经过可训练 domain 2 adapter，以及解冻后的 SSM，可以跟随目标一起移动。

因此损失下降可以来自 projector 或潜变量的适应，不能独立证明生成表情更符合情感。反过来，错误或滞后的 target_aff 也可能让生成参数偏离真实轨迹。当前证据不能确定实际梯度冲突幅度，需要测量重建项与情感项对 FiLM/synthesis 参数的梯度范数和余弦。

`LOSS.GENERATION_STATE` 默认为 0.1；`emotion_ssm/train/dualtalk.py:498` 将该项与 expression/jaw/neck/velocity 损失直接相加；`:853` 又按包含该项的 `val_total` 选择 checkpoint。baseline 则按纯重建项选择。情感模型的 best 并不一定是生成 MSE 最小的 checkpoint。

先运行同训练协议的 `GENERATION_STATE=0` 对照，单独验证状态条件是否有用，并另存按纯 `generation_total` 选出的 best。之后若保留一致性约束，可先用真实表情和可信的教师情感训练投影器，在独立验证集验证后冻结，并明确教师目标是否 stop-gradient。不要仅根据 scalar loss 大小选择权重。

**5. 当前 FiLM 以 8 秒为单位使用上一片段的状态。**

当前 YAML 为 `CHUNK_FRAMES=200`、`FPS=25`，即每个 chunk 8 秒；`CAUSAL_STATE_CONTEXT=true`。

`emotion_ssm/train/dualtalk.py:344` 在读取当前 chunk 的情感观测前，取旧 state 作为当前 chunk 的 context 和 `target_aff`。`StateFiLM.forward` 在整个 chunk 内广播同一个 gamma/beta。当前 chunk 的新情感证据要到下一 chunk 才影响 FiLM。于是 8 秒内的情感变化没有进入当前片段的情感调制，并且一致性损失把当前生成参数拉向 chunk 开始前的状态。

历史心境作为慢变量有合理用途，但它与当前情感证据承担的职责不同。可以保留慢变量，同时增加更短的因果观测更新或当前短窗情感条件。比较 25/50/100/200 帧时，baseline 与条件模型必须使用相同切片和训练预算。不要只在测试时切换 chunk 大小：它会同时改变 Wav2Vec2 上下文、生成器注意力范围、情感更新频率和状态时长。

FiLM 调制作用于共享 interaction feature，最终影响 expression、jaw、neck 全部参数。情感信号对这些部位的价值并不相同；应分别记录说话/倾听时的误差，以及上脸、口型、头部运动的变化，再决定是否对不同输出采用不同调制强度。单个部位 MSE 上升不能用“情感更自然”直接解释，必须有独立的情感、口型或主观证据。

**6. 三数据集的使用方式是分阶段迁移，baseline control 没有读入三个数据集。**

`build_feature_stores` 在 A0/A/B 阶段读取 EmotionTalk、IEMOCAP。条件 DualTalk 的训练入口和 baseline control 均只从 `DATA.DUALTALK_ROOT/train` 读取 WAV/FLAME。`DATA.SOURCES` 在 baseline control 中不会让它读取 EmotionTalk/IEMOCAP。

因此这些脚本对应“在情感数据上预训练，然后迁移到 DualTalk 生成”的实验。它们没有让三份数据同时为 Avatar FLAME 重建提供标签。若服务器另有三数据集 FLAME 转换或联合训练入口，需要另行审查。

更具体的迁移差异是：A0 训练七种模态子集；Phase B 只用 `FULL_AVT_MASK`，继续学习 event/action/coupling；DualTalk 接入时仅用 A-only。Phase B 新学到的多模态 action 行为没有在同一阶段得到 A-only 模态缺失训练，迁移时可能失效。DualTalk 的情感路径也没有消费用户描述中的文本 Context 和视频，它们只在前置情感数据训练中参与。

**7. A-only 的零 event 仍会触发非零自身注入。**

`emotion_ssm/models/observation.py` 在文本缺失时将 event 清零；但 `DyadicEmotionSSM.event_to_delta` 是两个带 bias 的 Linear 加 GELU。`transition` 和 `step_parallel` 无条件计算这条映射，故 `event=0` 不等于 `stimulus=0`。

对本地 `.transfer/phase_b_best.pt` 用 NumPy 重建该 MLP 的零输入前向，得到 `||event_to_delta(0)||₂ = 0.642273`。公开的 observation affect 经 L2 归一化，其范数约为 1；这说明零 event 注入不能忽略。该数值是衰减前的 stimulus 范数，不是最终状态变化，也不是生成 MSE。

这可能表示模型在训练中学到的事件偏置，不能直接称为数值错误。但在“无文本证据”部署条件下，反复注入同一偏置需要单独建模。可引入 event presence gate，或区分“缺失事件”和“已知的中性事件”，并验证 `event_to_delta(e)-event_to_delta(0)` 这类有零点约束的形式。整个 chunk 无语音时 conditioner 已经走 decay-only；此处问题发生在有语音、没有文本的 A-only 事件。

**8. 已有 Phase B 权重显示短期 partner 信号有收益，反事实与长跨度证据仍弱。**

从 `.transfer/phase_b_best.pt` 提取到 epoch 8、global_step 848、EmotionTalk+IEMOCAP、joint rollout 的配置与以下指标。完整原值、配置和 SHA-256 见同目录 `checkpoint_audit_metadata.json`。

| 保存指标 | 数值 | 可支持的判断 |
| --- | ---: | --- |
| val_partner_gain | 0.143882 | 关闭同一模型的 partner 路径后，一步复合损失更高 |
| val_cf_ranking_accuracy | 0.004954 | 保存的带 margin 反事实排序指标约 0.50% |
| val_counterfactual | 0.146835 | 反事实排序约束仍有未满足部分 |
| val_conditional_h1 | 1.576276 | 一步条件预测复合损失 |
| val_conditional_h32 | 1.858452 | 使用后续 event/action 的 32 步条件预测复合损失 |
| val_open_loop_h32 | 2.689226 | 后续只衰减的 32 步开放环复合损失 |
| val_h1_count / val_h32_count | 242.1 / 6.96 | 每 batch 平均有效目标数，长跨度监督明显更稀疏 |

这些是情感动力学复合损失，不是 FLAME 部位 MSE。`partner_gain` 来自同一模型的推理关闭消融，没有与独立训练的 self-only 模型比较，不能视为已经证明的因果效应。joint 模式把 h1 的 conditional 和同值 open-loop 按权重相加，保存的 partner_gain 也受这一定义影响。

CF 指标要求真实 action 比最难替代 action 至少好 0.2；无匹配候选的 batch 返回 0，验证又按 batch 平均。因此 0.50% 不能解释为普通二分类正确率，更不能与 50% 随机基准直接比较。应记录候选覆盖率、有效 anchor 数、无 margin 排序率和带 margin 成功率。当前 matcher 没有强制跨域语义距离可比，event/action 又与预测目标共同学习；观测数据上的匹配排序不能单独确认真实干预效果。

实际时间常数约为 `[0.5, 3.259, 6.423, 14.589, 30.473, 58.334, 118.137, 259.643]` 秒。这表明参数存在多时间尺度，不能证明每组已经对应可解释的心境持续/积累。训练每个 33-turn 窗口都从 baseline 和零 relation 重置，缺少窗口开始前的历史；h32 每窗口只有一个起点，长期状态需要 burn-in 或连续对话评估。

`dynamics_core.py:365` 在 open-loop 的 offset>0 时只执行 `decay_only`，按固定 1 秒推进，再对齐未来第 h 轮的标签。该定义后续没有 action 或 relation 演化，且固定秒数未必对应第 h 轮的实际时间。它适合验证无后续刺激的衰减假设；若要声称预测未来自然交互，应分别定义真实时间预测、给定候选动作的条件预测和未知未来输入的预测，并与 last-state、global mean、decay-only 等简单基线比较。

**9. 上游数据还有三个必须核实的风险。**

第一，`feature_store.py:354` 将 EmotionTalk 的 `intensity_abs/2` 当作 arousal，并设 `vad_mask=[True,True,False]`。IEMOCAP 的 arousal 则从原始 1–5 标注映射到 -1–1。EmotionTalk 原论文描述的是五级正负 sentiment 标注；没有证据证明本项目派生字段 `intensity_abs` 是同一心理量的独立 arousal 标注。应追溯该字段的生成方式，未经验证先把 arousal mask 设为 false，或独立训练 intensity head。强烈悲伤也可以是低唤醒；情感强度不等于唤醒度。来源：[EmotionTalk 原论文的标注字段与量表](https://arxiv.org/html/2505.23018v1#A2)。

第二，IEMOCAP 视频预处理只取一个对话视频；OpenFace CSV 读取没有保留人物身份，`slice_face_frames` 按时间戳选最高置信度人脸，对每句不传 active_role。这个实现没有保证视频人脸属于该句的说话人；若画面只有一个人，也没有据此屏蔽另一人的 utterance。需要人工抽查角色与视频轨迹对应关系，再重建受影响特征。没有真实视频，无法确定实际错误比例。

第三，IEMOCAP 音频预处理默认 `padding=True`，但未显式要求返回长度 mask；缺 mask 时直接平均所有输出帧。默认 `facebook/wav2vec2-base-960h` 官方 processor 的 `return_attention_mask=false`，所以批内 padding 可以进入平均池化。应由原始有效采样数构造输出池化 mask，不能把“模型 forward 是否需要 attention_mask”和“均值池化是否排除 padding”混为一谈。DualTalk 数据加载器已经归一化音频，本次没有发现所谓“DualTalk 完全未做音频归一化”的问题。来源：[官方 processor 配置](https://huggingface.co/facebook/wav2vec2-base-960h/blob/main/preprocessor_config.json)。

另外，特征读取只检查 A/T/label 数量与维度，没有完整验证 audio/text 的 utterance ID、抽取模型和归一化版本；同为 768 维不能保证特征分布一致。缺失时间戳默认使用序号，每步变为一“秒”，也会改变时间常数意义。必须用真实派生数据核验这些字段。

**10. 研究目标与验证方式需要对齐。**

当前 A0 前 10 轮为自监督阶段，之后加入 emotion/intensity/VAD 等监督；应准确描述为自监督预训练加监督约束。EMA 一致性主要约束 predictor 输出匹配教师，而下游使用 predictor 前的 affect。防坍塌、跨模态对齐和情感语义质量仍需在真实数据、每个域和每个模态子集上分别验证。

event 是当前文本 token 的投影，action 由融合特征映射得到；模块名和去相关损失不自动赋予“事件评价”或“安慰/攻击”等行为语义。Phase B 预测监督可以使它们具有统计用途，但解释 partner 作用时应另外提供可识别的事件/行为证据和反事实评价。测试目录中的合成对齐测试使用人工确定的适配器，测试通过只说明代码契约成立，不能证明实际 checkpoint 的表征已经对齐。

当前自然交互任务还不能仅由 raw coefficient MSE 覆盖。保留 MSE 作为明确的重建指标，同时报告口型同步、情感识别/强度、倾听反应、边界连续性、长程状态和有盲评的主观结果。没有后几项证据时，不应把 MSE 上升解释为模型在追求“更自然”的表达。

**11. 最小排查顺序。**

1. 归档现有结果及两组实际 config、commit、global_step；修复外部编码器覆盖，先重评同一个 checkpoint。核对两组 `evaluated_chunks`、切片、方向、test/ood 完全一致。
2. 修复 IEMOCAP-only 的 adapter source，重跑其 DualTalk 适配阶段，保持其他配置一致。
3. 在同一训练入口比较无情感条件与完整情感条件，两组使用相同生成器训练权限和优化预算。两组先只用重建损失，按纯生成误差选模型。
4. 增加“当前 A-only affect，无动力学”和“self-only 动力学”两组训练，从 observation、个人记忆到 partner influence 逐层确认增益。已有 `film_off` 可做诊断，但联合微调后关闭 FiLM 不能替代独立训练的 baseline。
5. 最后再加入经过校准的情感一致性目标、缩短因果更新窗，逐项记录收益，避免同时改变多项。
6. 对真实样本完成 speaker–face 对齐、VAD 标签语义、padding、模态缺失和真实时间戳检查。按对话进行配对误差统计或 bootstrap；模型训练重复使用多个种子，判断“小幅 MSE 差异”是否稳定。

只重评可以修复第 2 节的结果污染；它无法补偿第 1 节的优化差异或第 3 节的错误适配器训练。修复后是否优于 baseline，仍需要上述受控实验给出答案。
