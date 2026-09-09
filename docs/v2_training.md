# 新版情感 Avatar：训练、推理与验收

本地实现日期：2026-09-07。此次修改没有写入两台训练服务器，也没有启动或中断服务器上的训练。完整重训和真实数据效果比较尚未执行。

## 当前协议

| 项目 | v2 行为 |
| --- | --- |
| 输出块 | 1 秒、25 帧；最后不足一秒时返回实际完整帧，不丢弃尾块 |
| 生成器上下文 | 当前块和此前最多 3 秒的音频、用户 FLAME、逐帧情感条件 |
| 状态时间 | 状态属于观测结束时刻；先按实际间隔衰减，再当前校正、一次事件注入和持续行为影响 |
| 文本 | 最近 256 token 的可用双方历史，保留角色；新增文本单独编码为事件 |
| 视觉 | 用户 FLAME56 使用独立时序适配器；AU35 保留独立接口和身份 mask |
| Avatar 目标 | 当前目标 FLAME 只用于重建目标、离线适配器校准或已校准投影器训练；流式条件明确移除目标视觉 |
| 固定模块 | 生成阶段固定观测器、FLAME/域适配器、情感 SSM、情感特征提取模型 |
| 可训练主干 | 所有对照从第一步训练同一个 DualTalk 主干；原音频卷积特征提取器固定 |
| best | 验证集的 `generation_total`：expression、jaw、neck、块内 velocity 四项 MSE 的和 |
| 情感一致性 | 默认权重为零。开启时必须提供在真实 FLAME 上校准、与教师坐标和特征来源匹配的固定投影器 |

`none`、`affect`、`self`、`dyadic` 使用相同观测输入、数据顺序、主干训练权限、学习率和步数预算。`affect` 仅提供双方当前 affect；`self` 再提供 Avatar 的持续状态；`dyadic` 再启用 partner 影响、partner 持续状态和 relation。FiLM 从恒等映射初始化。`film_off` 是同一 checkpoint 的诊断消融，不代替独立训练的 `none` 对照。

生成阶段直接使用当前观测更新后的条件，没有上一块情感滞后。历史音频和历史条件由 `StreamState` 保存，生成器每次只返回新增帧。

## 环境与代码验收

新版使用独立的依赖文件，避免覆盖原论文环境约定：

```bash
python -m pip install -r requirements-test.txt
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 python -m pytest tests -q
```

Windows PowerShell：

```powershell
$env:OMP_NUM_THREADS='1'
$env:HF_HUB_OFFLINE='1'
.\.venv\Scripts\python.exe -m pytest tests -q
```

本次本地验证使用 Python 3.12、CPU PyTorch 2.5.1、Transformers 4.44.2。测试包含小型真实 Wav2Vec2/文本网络、A0、Phase A/B、FLAME 校准、四组生成器优化、完整权重恢复、预处理因果性和双进程 Gloo 加权指标。正式 CUDA 多卡、大模型、完整数据训练由用户安排执行。

2026-09-07 本地验收记录：`82 passed, 53 warnings in 12.94s`；59 个 Python 文件语法检查通过，`git diff --check` 通过。测试不下载预训练权重。这里的通过范围是代码协议与小模型执行，不包含服务器现有权重或完整数据集效果。

## 先重建情感特征

训练默认拒绝没有 v2 provenance 的旧特征。每个对话需要真实 `start_time`、`end_time`、角色、音频路径、文本可用时间和有效模态 mask。缓存记录提取模型、可解析的模型 revision、归一化版本、标签摘要与处理版本。本地模型目录按权重、配置和 tokenizer 文件内容计算 SHA-256，同一路径中的模型被替换也会被识别为不同来源。

EmotionTalk 只监督实际 valence 和独立 intensity，不再把 `abs(sentiment)` 当作 arousal，dominance 同样缺失。IEMOCAP 保留真实 VAD；其由 arousal 换算的 intensity 被标记为候选匹配用代理值，不作为独立 intensity 标签。`*_count=0` 表示该项没有有效标签，不能把输出的零误差解释为预测正确。

EmotionTalk 的 WAV 映射必须由现有数据索引提供，不能按相似文件名猜测。例：

```json
{
  "dialogue_001/utt_001": "/data/emotiontalk/audio/actual_utterance.wav",
  "dialogue_001/utt_002": "/data/emotiontalk/audio/another_utterance.wav"
}
```

重建工具读取原 labels、AU 和 splits，把新 A/T 特征写到独立目录：

```bash
python -m emotion_ssm.preprocess.emotion_features \
  --input-root datasets/emotiontalk/processed \
  --output-root artifacts/features/emotiontalk_v2 \
  --audio-manifest artifacts/emotiontalk_wavs.json \
  --audio-model TencentGameMate/chinese-hubert-base \
  --text-model hfl/chinese-macbert-base \
  --dataset emotiontalk --device cuda:0
```

上面的模型 ID 是重训配置示例，必须与实际选择的模型一致。工具重新提取音频，不会把另一骨干的 768 维缓存直接标为兼容。已有骨干可通过模型 ID 或本地模型目录指定。

IEMOCAP 可以用 `emotion_ssm.preprocess.iemocap` 从官方目录直接重建。输出设为 `artifacts/features/iemocap_v2`。设置 `PREPROCESS.FACE_ROLE_MAP` 指向人工核验的轨迹映射，例如：

```bash
python -m emotion_ssm.preprocess.iemocap --config configs/iemocap.yaml \
  DATA.IEMOCAP_RAW_ROOT /data/IEMOCAP \
  PREPROCESS.FACE_ROLE_MAP artifacts/iemocap_face_roles.json
```

映射文件格式：

```json
{
  "Ses01F_impro01": {"Ses01F": "0", "Ses01M": "1"}
}
```

数字必须对应该视频实际 OpenFace `face_id`，这里的值只展示格式。未映射或无法确认的人脸被标记缺失。`PREPROCESS.SKIP_OPENFACE True` 可以明确采用 A/T 模式。已有 IEMOCAP 特征也可通过 `emotion_features` 和显式 WAV 映射重建；未带身份核验记录的视觉不会被静默沿用。

预处理输出 `preparation_report.json` 或 `metadata/preprocess_summary.json`。有缺失文件、时间戳或映射时先检查报告；不要修改 mask 把失败项当作有效观测。重建使用 `--rebuild` 或 `PREPROCESS.REBUILD True`。原数据文件不移动。

## 重训入口

所有命令从仓库根目录运行。`--print-only` 仅打印待执行参数，便于先检查路径。

```bash
python scripts/run_v2_experiments.py \
  --baseline checkpoints/dualtalk_baseline.pth \
  --source all --mode pilot --stages all --nproc 2 --print-only
```

去掉 `--print-only` 后依次执行：A0 → Phase A → Phase B → 特征来源清单 → DualTalk 时间对齐和特征缓存 → FLAME/域适配器校准 → 四个生成对照 → test/OOD 评测。

`pilot` 的生成阶段预算为每组 1,000 步、种子 6666。上游阶段仍需形成可用的固定情感坐标，按各自配置训练；1,000 步不代表从随机情感编码器起训练整个系统即可完成校准。若上游与校准已完成，默认 `--stages controls` 只运行四个生成对照。

正式对照：

```bash
python scripts/run_v2_experiments.py \
  --baseline checkpoints/dualtalk_baseline.pth \
  --source all --mode full --stages all --nproc 2
```

正式预算固定为三个种子 6666、6667、6668，每组 30,000 次优化，每次 32 个有效新块。使用梯度累积在各 rank 内逐块处理；`--nproc` 必须整除 32。尾块按真实帧数计权。默认四组使用 FP32；若开展 AMP 实验，所有对照必须统一精度，发生非有限梯度时会报错，避免把跳过的更新计作完成的优化步。

独立配置位于：

- `configs/dualtalk_v2_none.yaml`
- `configs/dualtalk_v2_affect.yaml`
- `configs/dualtalk_v2_self.yaml`
- `configs/dualtalk_v2_dyadic.yaml`

单数据源对照将 `--source` 改为 `iemocap` 或 `emotiontalk`。`scripts/run_data_source_ablations.sh BASELINE_CHECKPOINT ...` 调用同一入口依次运行两组，已移除旧的三轮生成训练安排。IEMOCAP-only 自动使用 domain 1，EmotionTalk-only 使用 domain 0，并各自绑定来源清单、Phase-B checkpoint、音频/文本骨干和校准权重。

阶段可以单独指定，如 `--stages upstream`、`--stages features,calibration`、`--stages controls,evaluate`。完整流程的目录为：

```text
runs/v2/{source}/seed6666/phase_a_observation/upstream_v2/
runs/v2/{source}/seed6666/phase_a_dynamics/upstream_v2/
runs/v2/{source}/seed6666/phase_b_coupling/upstream_v2/
runs/v2/{source}/seed6666/dualtalk_calibration/upstream_v2/
runs/v2/{source}/{pilot|full}/seed6666/dualtalk_conditioned/{variant}/
runs/v2/{source}/{pilot|full}/seed6666/evaluation/
artifacts/v2/{source}/seed6666/feature_sources.json
artifacts/v2/{source}/timed/
```

校准必须在验证集优于固定均值教师对照，并保持非零样本方差，才会标记 `calibrated=true`。不通过时生成训练会停止，保留校准指标供定位；不会静默使用未校准的视觉编码器。

## DualTalk 文本和验证集

`emotion_ssm.preprocess.dualtalk` 使用提供的 TXT 做 CTC 时间格与回溯对齐，不生成或补写转录内容。输出词级 `start/end/available_at`。不支持的文字、无法对齐或低置信度片段记录为文本缺失。默认英文 CTC 模型不适合任意语言；换用外部时间戳输入时使用下面的协议。

有外部 ASR 输出时给流程传入 `--words-root`。目录结构为 `{words-root}/{train|test|ood}/{stem}.json`，每个文件是数组：

```json
[
  {"id": "speaker1:word0", "role": "speaker1", "text": "hello",
   "start": 0.12, "end": 0.39, "available_at": 0.68}
]
```

`available_at` 不能早于 `end`。音频不足一秒时按实际长度池化；模型 attention mask 策略和输出池化 mask 分别实现。已存在的历史文本不再次产生事件。

离线协议标为 `aligned_transcript`，外部 ASR 标为 `external_asr`。两者单独评测。情感数据集完整转写的 A0/A/B 特征使用 `offline_transcript_endpoint`：整句只在句子结束时可见。这些离线成绩不代表真实 ASR 延迟与识别错误下的成绩。

验证集从原 train 的视频来源 ID 按固定种子 6666 划出 10%，通过 `splits.json` 引用文件。该划分与生成训练种子分开。test、OOD 不参与选 best。缓存目录包含 `dataset_manifest.json`；原始 WAV/TXT/NPZ、时间对齐、模型来源或 mask 策略变化后需要重建，不能继续恢复同一实验的优化器。

目前没有经过核验的 DualTalk 跨文件对话映射和时间戳，因此每个成对片段独立重置状态。文件编号不用于推断连续性。A/B 则按完整原始对话顺序运行，状态持续保存，梯度每 32 个事件截断。

## checkpoint 与推理

生成阶段保存 `last.pt` 和按纯生成损失选择的 `best_generation.pt`。完整 checkpoint 包含构造配置、生成器和观测器全部权重、SSM、音频/文本模型配置与权重、tokenizer、特征来源、实验预算、划分/缓存摘要，以及恢复训练所需的优化器、RNG 和数据游标。恢复后不再用外部 A0 或 Phase-B observer 覆盖它。

```bash
python -m emotion_ssm.train.generation --config configs/dualtalk_v2_dyadic.yaml \
  --resume runs/v2/all/full/seed6666/dualtalk_conditioned/dyadic/last.pt

python -m emotion_ssm.evaluate_generation \
  --checkpoint runs/v2/all/full/seed6666/dualtalk_conditioned/dyadic/best_generation.pt \
  --split test --device cuda:0 --output runs/v2/dyadic_test.json

python -m emotion_ssm.evaluate_emotion_v2 \
  --checkpoint runs/v2/all/seed6666/phase_b_coupling/upstream_v2/phase_b_best.pt \
  --split test --device cuda:0 --output runs/v2/emotion_test.json
```

恢复时模型和训练配置以 checkpoint 为准，默认保留保存的数据路径。移动数据后，在命令末尾显式传入 `DATA.DUALTALK_ROOT`、`DUALTALK.TIMED_FEATURE_ROOT`、`DUALTALK.SPLIT_MANIFEST` 等路径覆盖；配置文件中的默认值不会自动替换旧路径。生成训练恢复要求同一划分、特征摘要和 world size；A0/A/B 也核对提取模型、对话 provenance 和划分摘要。旧权重可以通过 A0/Phase-A/Phase-B 的显式初始化参数重新开始实验；旧 optimizer 不能作为 v2 连续训练恢复。旧 DualTalk 的组件评测仍需显式设置 `DUALTALK.PROTOCOL_VERSION 1`，输出带 legacy 协议标签，不能混入 v2 汇总。

流式文件示例：

```bash
python -m emotion_ssm.infer.streaming_demo \
  --checkpoint runs/v2/all/full/seed6666/dualtalk_conditioned/dyadic/best_generation.pt \
  --target-audio inputs/avatar.wav --partner-audio inputs/user.wav \
  --partner-flame inputs/user.npz --words inputs/available_words.json \
  --roles avatar user --output-dir outputs/session1 --device cuda:0
```

程序接口使用 `load_generation()` 和 `StreamSessions(model)`。每个 packet 包含 `session_id`、有序的 `roles=(avatar,user)`、观测结束 `time`、双方 `[1, samples]` 原始 16kHz 音频、用户 `[1,frames,56]` FLAME。缺失音频用等长零数组和 `*_audio_length=0` 表示；FLAME 缺失用 `partner_visual_mask=False`。AU 输入使用 `partner_au/partner_au_mask`，生成器的 FLAME 槽位仍需提供对应帧数的占位数组。静音默认按与预处理相同的 RMS 阈值判断，也可以显式传入 `*_speech_active`。

返回 `(新增帧, 更新状态, 观测诊断)`。迟到文本从实际传入的当前块开始生效；不会修改已返回帧。切换角色顺序必须 `reset(session_id)`；不同会话分别保存状态。超过一秒的缺口需要以逐秒缺失 packet 表示。

动力学查询接口为 `ssm.predict_at(state, current_time, query_times)`。Open-loop 只按实际秒数纯衰减；conditional 可另传 `(时间, 双方 event/action 观测)`，必须显式给出行为持续时长，内部始终 `correct=False`，不会使用未来 affect。两类预测、last-state、训练均值和纯衰减的误差分别记录。

## 评测与效果验收

生成评测按 SSE / 有效元素数汇总，保存每个方向片段的分子和分母。验证 sampler 不补齐重复样本。跨块边界速度只在相邻两帧均有效时计入。

`observation_diagnostics` 报告 domain 2 的七种模态子集方差、样本余弦相似度和固定 A/T 教师误差，确定性抽样前 16 个片段的前 32 块。A0 与独立情感评测另报告每个训练域/模态的 F1、UAR、有效标签数和 VAD 指标。没有独立情感标签的 DualTalk 不会伪造情感准确率。

反事实指标同时给出候选覆盖率、有效候选普通排序率、margin 成功率及分母；`cf_ranking_accuracy` 保留为最难候选的按 anchor margin 指标。训练候选来自批次内其他对话、同一数据域。验证按数据集固定顺序每 4 个对话组成候选池，再把整组分配到各卡，候选池不随验证 batch size 或卡数变化；池大小写入配置和指标。没有有效候选的样本不计入排序分母。

多种子配对汇总示例：

```bash
python -m emotion_ssm.paired_report \
  --baseline runs/v2/all/full/seed6666/evaluation/none_test.json \
             runs/v2/all/full/seed6667/evaluation/none_test.json \
             runs/v2/all/full/seed6668/evaluation/none_test.json \
  --condition runs/v2/all/full/seed6666/evaluation/dyadic_test.json \
              runs/v2/all/full/seed6667/evaluation/dyadic_test.json \
              runs/v2/all/full/seed6668/evaluation/dyadic_test.json \
  --output runs/v2/all/full/dyadic_vs_none_test.json
```

工具核对种子、划分、特征来源、实验协议和有效目标。先合并成对片段的两个方向，再按原始视频来源做 cluster bootstrap，避免把相关片段当成独立对话。没有跨文件映射时汇总单位明确写为 paired clip。输出 condition − baseline，各部位 MSE、速度和边界误差分别报告三个种子的差值、均值/标准差和来源聚类置信区间。

代码验收通过只表示实现遵守上述数据与时序约定。效果验收仍需完成正式重训，再结合配对误差、情感指标和表征诊断判断；当前不能声称 MSE 已低于 baseline。

## 历史、事件与 partner 实验

`emotion_ssm.evaluate_influence` 对已有四组 1000 步 checkpoint 进行固定选样的推理对照：重置历史状态、删除或重复真实文本事件、置换 partner 状态观测及关闭耦合。正常输入的真实目标误差与干预响应分别汇总；原始音视频、当前 affect 和文本在对应状态干预中保持一致。`none` 对状态干预没有响应是结构预期，不能直接算作质量劣势。

协议和窗口见 [实验协议](influence_experiment_protocol_2026-09-08.md)。服务器队列入口为 `scripts/run_influence_experiments.py`；生成报告使用 `python -m scripts.report_influence_experiments --root result_DualTalk/influence_20260908`，绘图依赖 `matplotlib==3.9.4`。配对原始统计、选样清单和各组结果保存在同一输出目录，可复算误差和原始视频聚类区间。
