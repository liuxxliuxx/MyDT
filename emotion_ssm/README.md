# 耦合情感状态空间实现

新版训练、秒级 token、快慢状态、双人耦合、联合梯度和完整 checkpoint 以 [v3 使用说明](../docs/v3_training.md) 为准，依赖使用 `requirements-v2.txt`。旧 v2 实验保留 [v2 使用说明](../docs/v2_training.md)。下文保留早期实现供追溯，其中外部编码器恢复和旧权重命名不适用于 v3。

`emotion_ssm` 是独立于原 `DualTalkModel` 的正式训练包。原始数据只读；IEMOCAP 派生特征写入 `artifacts/`，训练输出写入 `runs/`。`emotion_ssm.utils.paths` 会拒绝把输出目录放进 `datasets/`。

## 1. 环境

目标版本为 Python 3.8、PyTorch 1.12.1、Transformers 4.26 和 YACS 0.1.8。先使用原项目环境安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 不固定 CUDA 对应的 PyTorch wheel；服务器需要先按已安装的 CUDA 版本准备 PyTorch 1.12.1。

IEMOCAP 视频特征需要 OpenFace `FeatureExtraction`。配置中的 `PREPROCESS.OPENFACE_BIN` 可以写绝对路径。当前实现直接处理整段对话视频的 OpenFace CSV，再按 utterance 时间戳切片，不生成临时视频。

建议在服务器设置：

```bash
export PYTHONDONTWRITEBYTECODE=1
```

## 2. 数据目录

代码默认使用以下结构：

```text
datasets/
  emotiontalk/processed/
    dialogues/<dialogue_id>/audio_features.pt
    dialogues/<dialogue_id>/text_features.pt
    dialogues/<dialogue_id>/face_au_features.pt
    dialogues/<dialogue_id>/labels.json
    splits/train_dialogues.txt
    splits/val_dialogues.txt
    splits/test_dialogues.txt
  iemocap/raw/Session1 ... Session5
  dualtalk/train|test|ood/*.wav|*.npz
artifacts/features/iemocap/
runs/
```

服务器上的 EmotionTalk 目录可以直接通过 CLI override 读取，无须复制、改名或建立缓存：

```bash
DATA.EMOTIONTALK_ROOT /home/s21_yhr/lzh/Emotiontalk_work/processed
```

所有写入位置仍由 `TRAIN.OUTPUT_ROOT` 和 `DATA.IEMOCAP_FEATURE_ROOT` 单独指定。

## 3. IEMOCAP 预处理

```bash
python -B -m emotion_ssm.preprocess.iemocap \
  --config configs/iemocap.yaml \
  DATA.IEMOCAP_RAW_ROOT /absolute/read-only/IEMOCAP \
  DATA.IEMOCAP_FEATURE_ROOT /absolute/writable/artifacts/features/iemocap
```

输出包含五个 speaker-disjoint fold。fold 5 使用 Session 1-3 训练、Session 4 验证、Session 5 测试。`ang/dis/fea/hap+exc/neu/sad/sur` 映射到七类；`fru/oth/xxx` 的 `emotion_id=-1`，只参加 VAD 监督。V/A/D 从 `[1,5]` 映射到 `[-1,1]`。AU 归一化统计只使用各 fold 的训练 session。

## 4. 分阶段训练

先执行每阶段 dry run。A0 命令：

```bash
torchrun --nproc_per_node=2 -m emotion_ssm.train.phase_a_observation \
  --config configs/phase_a_observation.yaml --dry-run \
  DATA.EMOTIONTALK_ROOT /home/s21_yhr/lzh/Emotiontalk_work/processed
```

A0 同时训练 A、V、T、AV、AT、VT、AVT 七个子集，EMA teacher 只读取完整 AVT。输出：

```text
runs/phase_a_observation/fold5/
  observation_encoder.pt
  ema_teacher.pt
  emotion_heads.pt
  best.pt
  last.pt
```

个人动力学阶段：

```bash
torchrun --nproc_per_node=2 -m emotion_ssm.train.phase_a_dynamics \
  --config configs/phase_a_dynamics.yaml --dry-run \
  DATA.EMOTIONTALK_ROOT /home/s21_yhr/lzh/Emotiontalk_work/processed
```

该阶段关闭 partner influence，训练 baseline、八组正时间常数、自身注入、观测校正和 `{1,2,4,8,16,32}` 开放环预测，输出 `phase_a_best.pt`。

双人耦合阶段：

```bash
torchrun --nproc_per_node=2 -m emotion_ssm.train.phase_b \
  --config configs/phase_b.yaml --dry-run \
  DATA.EMOTIONTALK_ROOT /home/s21_yhr/lzh/Emotiontalk_work/processed
```

Phase B 从 `TRAIN.PHASE_A_CHECKPOINT` 恢复，启用 A→B、B→A 两套参数和 relation GRU。每个 batch 随机交换角色命名。在线 matcher 仅使用干预发生时已知的上下文：在不同 dialogue 中筛选发送方向、情感标签、强度、轮次位置和事件语义接近、发送方 action 不同的样本，并从前八个候选中选最难负例。接收方下一次发言的 affect 只作为后续排序监督目标，绝不参与候选筛选。无候选的行不计算 `L_cf`。输出 `phase_b_best.pt`。

`TRAIN.FINETUNE_OBSERVATION=true` 时，Phase B 以 `TRAIN.STATE_LR_SCALE` 倍学习率继续微调 Observation Encoder；`phase_b_best.pt` 会同时保存这部分权重，评测和条件 DualTalk 会优先读取微调后的 encoder。

dry run 通过后去掉 `--dry-run`。四个训练入口均保存 optimizer、scheduler、AMP scaler 和 Python/NumPy/PyTorch RNG；恢复方式为：

```bash
python -B -m emotion_ssm.train.phase_b --config configs/phase_b.yaml \
  --resume runs/phase_b_coupling/fold5/last.pt
```

## 5. 条件 DualTalk

配置 `DUALTALK.BASELINE_CHECKPOINT` 指向旧模型权重，`DUALTALK.PHASE_B_CHECKPOINT` 指向 `phase_b_best.pt`：

```bash
torchrun --nproc_per_node=2 -m emotion_ssm.train.dualtalk \
  --config configs/dualtalk_conditioned.yaml --dry-run
```

模型从双路原始音频提取 Wav2Vec2-Base A-only observation，得到 `[z_target,z_partner,r]`，再用零初始化 FiLM 调制 512 维 interaction feature。前 `FREEZE_STATE_EPOCHS` 轮冻结情感系统，之后以主学习率的 `JOINT_FINETUNE_LR_SCALE` 倍联合微调。关闭 FiLM 时前向路径与旧 `DualTalkModel` 相同。输出 `dualtalk_emotion_best.pt`。

独立 demo 输出 blendshape、FLAME 参数和情感状态：

```bash
python -B -m emotion_ssm.infer.dualtalk_demo \
  --config configs/dualtalk_conditioned.yaml \
  --checkpoint runs/dualtalk_conditioned/fold5/dualtalk_emotion_best.pt \
  --target-audio target.wav \
  --partner-audio partner.wav \
  --partner-flame partner.npz \
  --output-dir runs/dualtalk_demo
```

需要直接生成视频时，在 YAML 中设置渲染器命令模板，例如 `DUALTALK.RENDER_COMMAND: "python render_one.py --flame {flame} --audio {audio} --output {output}"`。命令成功后必须生成模板中的 `{output}`，否则 demo 会报错，不会把缺失的视频当成完成结果。

## 6. 评测

```bash
python -B -m emotion_ssm.evaluate \
  --config configs/phase_b.yaml \
  --split test \
  --observation-checkpoint runs/phase_a_observation/fold5/observation_encoder.pt \
  --heads-checkpoint runs/phase_a_observation/fold5/emotion_heads.pt \
  --ema-checkpoint runs/phase_a_observation/fold5/ema_teacher.pt \
  --dynamics-checkpoint runs/phase_b_coupling/fold5/phase_b_best.pt \
  --output runs/phase_b_coupling/fold5/test_metrics.json
```

输出七种模态子集的 Macro-F1/UAR、intensity MAE、VAD CCC、跨模态同句/随机句余弦间隔、latent 方差、speaker/domain leakage probe、多跨度 loss、PartnerGain 和反事实排序准确率。消融可用 CLI override，例如：

```bash
python -B -m emotion_ssm.evaluate ... DYNAMICS.FIXED_RELATION true
python -B -m emotion_ssm.evaluate ... DYNAMICS.SYMMETRIC_COUPLING true
python -B -m emotion_ssm.evaluate ... DYNAMICS.DISABLE_LONG_TIMESCALES true
python -B -m emotion_ssm.evaluate ... DYNAMICS.CORRECTION_MODE none
python -B -m emotion_ssm.evaluate ... DYNAMICS.RANDOM_PARTNER true
```

可把一个已确认的评测 JSON 作为参考，检查 AU 时序、跨模态对齐、latent 非坍塌、单模态 VAD 可用性、泄漏接近随机猜测，以及 emotion F1 没有明显下降：

```bash
CUDA_VISIBLE_DEVICES=2 python -B -m emotion_ssm.evaluate \
  --config configs/phase_b.yaml \
  --split test \
  --observation-checkpoint runs/phase_a_observation/fold5/observation_encoder.pt \
  --heads-checkpoint runs/phase_a_observation/fold5/emotion_heads.pt \
  --ema-checkpoint runs/phase_a_observation/fold5/ema_teacher.pt \
  --output runs/phase_b_coupling/fold5/test_metrics.json \
  --reference-metrics runs/reference_metrics.json \
  --assert-observation-contracts
```

## 7. 候选动作轨迹

输入 `.pt` 至少包含：

```python
{
    "user_observation": {
        "aff": Tensor[T, 128],
        "event": Tensor[T, 128],
        "action": Tensor[T, 128],
        "reliability": Tensor[T, 3],
        "modality_mask": Tensor[T, 3],
    },
    "candidate_actions": Tensor[C, T, 128],
    "dt": Tensor[T],
    # 可选："target_user_aff": Tensor[128]
}
```

运行：

```bash
python -B -m emotion_ssm.infer.trajectory \
  --config configs/phase_b.yaml \
  --checkpoint runs/phase_b_coupling/fold5/phase_b_best.pt \
  --input candidate_actions.pt \
  --output runs/candidate_rollout/trajectories.pt
```

输出包含每个候选的 User/Avatar `z`、affect、relation、分数和排序。未提供 `target_user_aff` 时，默认按双人 affect 轨迹的平均余弦一致性排序。

## 8. 测试

```bash
pytest -q tests
```

测试使用临时 `.pt/json/WAV` fixture，不读取真实数据或网络。完整 DualTalk dry run 会加载配置指定的 Hugging Face 权重，所以服务器应提前准备本地缓存，或把模型路径写成服务器上的只读绝对路径。

`tests/test_emotion_contracts.py` 还固定检查以下模型契约：AU 帧顺序敏感且 padding 值不影响输出；同 utterance 的 A/V/T 相似度高于随机 utterance；latent 不坍塌；A、V、T、AVT 都能经过 emotion/VAD 预测头；open-loop 不读取未来 affect；sender action 的替换主要影响 receiver；连续 DualTalk chunk 传递状态；speaker/domain probe 接近随机猜测，同时受参考指标约束的 emotion 性能不得明显下降。
