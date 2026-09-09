审查时间：2026-09-06，北京时间约 10:40–10:57。只读检查了服务器1和服务器2的进程、训练日志、保存配置、checkpoint 元数据、评测 JSON 和历史源码。另用单线程 CPU 对 128 条现成 IEMOCAP 特征做了编码器检查。未修改服务器文件，未调用 GPU 重评、未改变训练进程。本地只新增审查文档、证据和历史源码副本。

**最需要处理的是生成微调后的情感表征退化，以及旧版生成阶段没有跨 chunk 保存状态。** 同时，已有比较混有旧版/新版、25/200 帧、3/7/18/23 轮。中间 checkpoint 已经足以定位这些问题，不需要等训练全部结束；各项问题对最终 MSE 的因果贡献仍需受控消融。

**1. 先纠正本地代码与实际运行版本的对应关系。**

服务器2实际项目根目录是 `/data/home/kxrgzn/lzh/MyDualTalk`，日志是 `runs/dualtalk_all_three_ddp.nohup.log`。进程在 9 月 4 日 22:23 启动；当前磁盘上的训练代码在 9 月 5 日 18:07 被更新。因此，直接阅读当前文件不能代表正在运行的 Python 进程。

| 实验 | 模型结构与训练方式 | 审查时状态 |
| --- | --- | --- |
| 服务器2 all_three | 旧版，2 个 domain，aff_head、self_input；生成器从第1轮训练 | 已完成27轮，best 为23轮、136183 step |
| 服务器2 fold5 第3轮、epoch22/25 快照 | 旧版，chunk 独立训练 | 已有完整生成评测与消融 |
| 服务器1 fold5_default200 | 新版，3 个 domain，shared_affect_projector、event_to_delta；连续 chunk、生成器冻结10轮 | 已完成44轮，best 为39轮 |
| 服务器2保存的 default200 第7轮快照 | 来自新版 fold5_default200，checkpoint 内保存了 FREEZE_BASELINE_EPOCHS=10 | 该快照确实仍在生成器冻结期 |

对服务器2旧版，前面的“3轮模型生成器被冻结10轮”推断应撤回。旧版只冻结 conditioner，生成器一直训练。旧版 evaluator 也没有在载入完整 checkpoint 后再覆盖 observer，不能把本地新版的覆盖风险当作旧版结果的原因。

同一个 `runs/phase_a_observation/fold5/observation_encoder.pt` 路径目前已经是新版3-domain权重，而正在运行的 all_three checkpoint 仍包含旧版2-domain编码器。评测和恢复必须绑定模型内保存的版本，不能仅依赖可被覆盖的路径。

历史代码来源为服务器2 `legacy_eval_655d31a` 目录，并通过 checkpoint 的参数名称、维度和保存配置交叉核验。本地副本：[训练入口](server2_legacy_training_2026-09-06.py)、[条件模型](server2_legacy_conditioner_2026-09-06.py)、[观测编码器](server2_legacy_observation_2026-09-06.py)、[动力学训练](server2_legacy_dynamics_training_2026-09-06.py)。

**2. 实际指标显示，小幅退化主要集中在速度项；新版早期快照另有冻结因素。**

以下均为 test 的200帧评测，覆盖2580个chunk。generation_total 是 expression、jaw、neck、velocity 四项 MSE 的直接相加，不是56维统一加权MSE。

| checkpoint | expression | jaw | neck | velocity | generation_total |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline control，第3轮，8880 step | 0.325635 | 0.00105794 | 0.01703374 | 0.01781591 | 0.361543 |
| 旧版 fold5，第3轮，8880 step | 0.326252 | 0.00106681 | 0.01713764 | 0.01820560 | 0.362662 |
| 旧版 all_three，第23轮，136183 step | 0.315285 | 0.00096940 | 0.01687175 | 0.01871473 | 0.351841 |
| 新版 default200，第7轮，8057 step | 0.366044 | 0.00107170 | 0.01722093 | 0.01997915 | 0.404316 |

等3轮的旧版比较中，情感模型 expression 高0.19%、jaw高0.84%、neck高0.61%、velocity高2.19%，总和高0.31%。这组差异存在，但只有单次训练和汇总指标，尚不能判断是否稳定超过训练随机性。

当前 all_three 第23轮相对于3轮baseline，前三项分别低3.18%、8.37%、0.95%，velocity高5.05%，总和低2.68%。因此不能概括为“当前旧版所有部位都更差”；同时，两者训练预算不同，这也不能证明情感模块带来了2.68%的提升。

另有25帧测试：`conditioned_test_legacy.json` 保存的是 all_three 第18轮，`baseline_test.json` 是3轮baseline，25748个chunk。前者总和0.502294，后者0.499642，差0.53%；主要差在velocity高8.13%。两模型训练配置都是200帧，测试改成25帧同时改变了语音上下文、生成注意力范围和状态更新时间。25帧表与200帧表不能混用，也不能用不同轮次的两张表单独量化切片的因果影响。

新版第7轮总和高11.83%，但它仍处于生成器冻结期，baseline已经直接微调主干。这一比较中的训练权限确实不一致。

原始来源：服务器2 `runs/comparison_epoch3/`、`runs/comparison_current_baseline/`、`runs/comparison_epoch24/`。完整数值已保存在 [证据JSON](server_training_evidence_2026-09-06.json)。

**3. 生成微调后，情感编码器在原域上的输出方向几乎变成常量。**

为区分“损失很低”和“保留了情感信息”，抽取 IEMOCAP fold5 test 的8个对话，每个均匀选择16句，共128条固定的预提取音频特征。对所有旧版编码器使用相同输入、A-only、domain 1、eval模式，CPU float32、单线程。只读取 checkpoint 内的 observer 权重，没有加载几GB的完整生成器，也没有重新抽取波形特征。

| 编码器来源 | 不同样本aff平均余弦相似度 | 单位化aff各维标准差的均值 |
| --- | ---: | ---: |
| Phase B 第4轮最佳 encoder | 0.9258909 | 0.0225800 |
| fold5 条件模型第3轮 | 0.9258909 | 0.0225800 |
| fold5 条件模型第22轮 | 0.9999981 | 0.00011673 |
| 当前 all_three 第23轮 | 0.9999995 | 0.00006307 |

第23轮的方向变化尺度约缩小358倍；event/action平均余弦也分别达到0.9999942和0.9999951。这个检查直接确认了原IEMOCAP特征上的表征退化。它不是完整DualTalk波形路径评测，不能把128个样本外推为所有生成输入的坍缩比例。

源码给出了明确的退化通路：

- 旧版训练入口第71–75行，`set_state_frozen(False)` 对整个 conditioner 调用 `requires_grad_(True)`。第260行从第11轮开始解冻。配置里的 `TRAIN.FINETUNE_OBSERVATION=false` 没有阻止这个入口解冻观测编码器；此前音频卷积层设置的冻结也被整个模块的开关覆盖。
- 第94–97行，解冻后的 `target_aff` 不再 detach。第115–118行同时训练 `projector(generated)` 与情感目标，使二者余弦接近1。
- 这个阶段没有持续的情感分类/多模态对齐监督，也没有固定教师锚定旧情感坐标。投影器和编码器可以共同靠近同一方向来降低一致性损失。
- 对当前all_three，验证一致性损失从第10轮0.012503，降至第11轮0.0003278、第12轮0.0000732、第23轮0.000004249；表情MSE没有出现同量级改善。

代码机制、解冻时点、损失变化和固定输入检查一致支持“情感表征在生成微调中退化”的判断。尚未通过重训消融测出它造成了多少MSE损失。仅把一致性权重调大可能继续强化这条退化通路。

**4. 旧版实际没有把情感状态沿对话传下去。**

旧版训练入口第83–97行每次调用 conditioner 都没有传 `state`，并用 `_` 丢弃返回状态；数据集是独立 `DualTalkChunkDataset`。条件模型第161–165行在 `state is None` 时从个体baseline和零relation初始化。

所以每个8秒chunk只经历“target半步、partner半步”，然后状态被丢弃。跨chunk的情感持续、衰减、累积和关系记忆，没有进入这个生成训练/评测流程。Phase B 学过时间常数，也无法自动补上调用方丢弃的历史。

同一时间窗的双方音频被强制解释为两个顺序事件，各推进4秒，没有按实际说话轮次或静音情况驱动更新。它与上游按utterance和真实间隔推进的动力学训练不一致。FiLM又把一个固定context广播到整个8秒片段，不能单靠这条路径表达片段内的情感变化。

服务器1新版已经串联同一NPZ的chunks，但每条NPZ仍单独初始化。实查训练集：9212条有效样本，23684个200帧chunk，61.10%的chunk有前一段状态；多数样本只有3个完整chunk。因此它已有短历史，不应称为“每段都重置”，但仍缺少跨原始切片的长期对话连续性，300秒时间尺度不能仅凭当前生成样本得到验证。

**5. 现有partner消融未显示稳定的附加收益。**

服务器2旧版fold5第22轮，在同一2580-chunk test上：

| 评测条件 | generation_total |
| --- | ---: |
| full | 0.3574473024 |
| random_partner | 0.3574473466 |
| self_only | 0.3574302256 |
| film_off | 0.3625493698 |

打乱情感路径的partner音频，误差总和只变约4.42e-8；关闭SSM的partner influence后略好；关闭FiLM后明显变差。说明联合训练后的模型使用了FiLM，但当前证据没有显示SSM的partner influence提供了可辨认的MSE收益。表征趋同可以解释其中一部分。

这里的 `self_only` 仅关闭状态模型的influence计算。conditioner仍可把partner状态拼到context，baseline也仍接收partner音频和表情。因此不能把这张表解释为“整个Avatar没有使用对方信息”。关闭已联合训练的FiLM也不能替代独立重训的无情感baseline。

当前all_three第23轮尚未找到对应的完整partner消融，不能把第22轮fold5表直接标成all_three结果。

它使用的旧Phase B最佳权重是第4轮：val_partner_gain=0.028084，val_cf_ranking_accuracy=0。后者要求满足0.2的margin，且无匹配候选时也返回0；不能解释成普通分类准确率。服务器1新版Phase B第8轮的0.143882 partner_gain属于另一套模型和损失定义，不能替代服务器2的证据。

**6. 多模态情感与生成阶段之间还有训练任务落差。**

三数据集的实际使用是“EmotionTalk+IEMOCAP预训练情感，再用DualTalk训练FLAME生成”。baseline control只读取DualTalk的WAV/FLAME；传 `DATA.SOURCES` 不会让它读取另外两份数据。

旧版情感编码器把A/V/T融合后的同一个hidden分别映射成aff、event、action，没有结构上隔离三者的信息来源。当前音频驱动生成时，face/text全部置零，情感分支只消费A-only；partner blendshape仍进入原DualTalk主干，但没有进入多模态情感观测。用户设想的Audio、Context、video联合情感观测尚未在这个生成入口落实。

Phase B只用AVT路径训练动力学。旧版多步预测虽然关闭correction，却继续将未来每轮的 `observation.aff/event/action` 输入 `step`；`self_input` 直接读取aff。它是已知中间观测条件下的预测，不能当作“不知道未来情感时预测32轮”的证据。缺少A-only、无未来aff的对应训练与评估，会高估预训练动力学向生成任务迁移的能力。

单数据源还有明确接线错误：旧版 `AudioOnlyStateObserver.forward(..., dataset_id=1)` 固定走IEMOCAP适配器。EmotionTalk-only只训练domain 0，生成阶段却使用domain 1，并在前10轮冻结conditioner；因此其3轮结果没有使用训练好的EmotionTalk音频适配器。新版代码的问题方向不同：固定复制source 0到domain 2，会影响IEMOCAP-only。必须按模型版本修正，不能统一按同一个domain数字处理。

**7. 上游数据中已确认的语义问题。**

EmotionTalk的实际19250条记录都满足 `intensity_abs=abs(sentiment_score)`，没有独立VAD标注。代码却将 `intensity_abs/2` 作为arousal，并将其mask设为有效；IEMOCAP的arousal来自真实VAD评分。两者监督的心理量不一致。该字段应保留为强度/情绪极性幅度，缺少证据时不要当作真实唤醒度。字段定义也与 [EmotionTalk论文的情绪极性标注说明](https://arxiv.org/html/2505.23018v1#A2) 一致。

服务器1新版还把EmotionTalk的Chinese HuBERT域适配器复制到DualTalk域，而DualTalk情感音频特征来自English Wav2Vec2-base；同为768维不代表同一特征空间。这条风险不适用于旧版all_three默认使用domain 1的情况。

IEMOCAP人脸按置信度选择、未显式跟说话角色绑定，以及音频预处理均值池化是否排除padding，仍需要进一步数据QA。没有本次人工视频抽样或完整特征重算，不能把它们列为已确认的MSE主因。

**8. 目前继续训练能回答什么。**

all_three从第23到27轮，训练expression从0.07548降到0.06838，验证expression仍在0.315–0.318附近，best仍是23轮。当前存在较大的训练/验证差距；更多epoch可能继续改善重建，也可能继续拟合训练数据，但不会自动修复状态丢弃、模态缺口或表征退化通路。

训练入口把DualTalk的 `test` 目录直接当作validation并据此选best，所以这些test指标已参与模型选择。独立泛化结论应使用未参与选择的数据。旧fold5快照的OOD总和从第3轮0.607080到第22轮0.616640、第25轮0.609601，并未随test改善而单调改善；这些是fold5证据，不是当前all_three第23轮的OOD结果。

**9. 优先修正与验证顺序。**

1. 固定每个实验的代码快照、模型内config、checkpoint标识、chunk长度、训练步数和可训练模块。先统一200帧评测；若最终需要1秒交互，训练与评测都按相应时间粒度设计。保留独立验证/测试划分。
2. 优先保护情感表征：冻结预训练observer与固定教师；使 `FINETUNE_OBSERVATION` 在生成入口真正生效；先移除未校准的一致性损失做对照。后续需要微调时，加入原情感数据/固定教师保持约束，并持续检查情感指标与样本间方差。仅detach目标也不能阻止其他生成梯度改变observer，冻结或保留监督才是完整措施。
3. 在同一训练入口、相同初始化和主干训练权限下，比较baseline、仅当前aff的FiLM、持续self状态、dyadic状态。先只用重建项，按纯generation_total选best，逐步定位哪一层产生收益或损失。
4. 按对话ID和角色保存/传递状态，边界正确重置；分离短时表达和长时心境，使用真实时间差与语音活动。采用更短的因果观测更新，检查跨chunk连续性。
5. 恢复Context/video情感输入，或明确将生成目标限定为A-only，并在上游做相同缺失模态条件的训练。修复EmotionTalk-only旧domain映射及新版IEMOCAP-only初始化来源。
6. 修正伪arousal监督。预测评估分开报告条件预测和未知未来输入的预测；先对比last-state、mean、decay-only等简单基线，再评价partner增益和长时动力学。
7. 用多种子和按对话配对的误差统计判断0.3%左右差异。保留部位MSE，同时测速度/边界连续性、口型和独立情感质量；没有这些证据时，不能把MSE升高解释为“更自然”。

上述为待实施方案。本次未改训练代码或任何服务器文件，未重训验证修正后的收益。

完整数值与配置：[服务器证据](server_training_evidence_2026-09-06.json)。编码器CPU检查方法：[只读脚本](readonly_observer_probe_2026-09-06.py)。仅针对当前新版代码的其他静态风险：[独立附录](emotion_training_static_audit_2026-09-06.md)。本地旧Phase B分析文件 `checkpoint_audit_metadata.json` 对应服务器1新版来源，不能当作服务器2旧版all_three的上游checkpoint。
