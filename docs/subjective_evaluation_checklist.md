# 主观评测清单

以下项目由受试者或标注员填写，不生成伪造的自动分数。

## 实验设置

- [ ] 同一段对话比较 baseline DualTalk 与 emotion-conditioned DualTalk。
- [ ] 模型名称随机化，播放顺序随机化。
- [ ] 音频响度、渲染分辨率、帧率和时长一致。
- [ ] 每个样本至少由三名标注员独立评分。
- [ ] 记录标注员是否看过训练集角色。

## 量表

- [ ] Empathy：Avatar 的反应是否符合 User 当前情绪和此前对话状态，1-5 分。
- [ ] Naturalness：表情、下颌和头部运动是否自然，1-5 分。
- [ ] User satisfaction：如果这是实时 Avatar，用户是否愿意继续交互，1-5 分。
- [ ] Pairwise preference：baseline / conditioned / 无明显差异。
- [ ] 失败类型：情绪方向错误、强度错误、延迟、过度同步、表情僵硬、动作抖动。

## 汇总

- [ ] 报告样本数、标注员数、均值、标准差和 95% bootstrap 置信区间。
- [ ] Pairwise preference 使用双侧 binomial test。
- [ ] 报告标注员一致性，不删除低分或无差异样本。
- [ ] 将 OOD 角色和 OOD 对话场景单独汇总。
