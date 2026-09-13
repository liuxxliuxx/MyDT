# 新旧checkpoint的同坐标评测

入口为 `scripts/compare_staged_v33_checkpoints.py`。它调用现有
`emotion_ssm/train/staged_v33/evaluation.py`，单独写评测目录，不修改训练权重、优化器或训练清单。

共同参考为旧版9750步；公共起点包括双方fast、slow、relation、baseline及事件记录。
在线起点从每条完整对话开头，以被测模型自身event/action头及状态参数重放产生。
复用的只是已冻结的观测特征和教师目标，每次完整推进状态，不从训练结束时的状态缓存替代最佳权重的在线状态。

脚本拒绝以下不一致：教师、观测坐标、情感标签头、构造配置、验证/训练校准清单、
特征manifest、公共起点生产者、查询时间、真实标签及有效mask。训练中可更新的event/action头不属于冻结坐标。
每个checkpoint只打开一次并另存快照，因此训练同时替换best文件不会导致一次评测混用两版权重。

在服务器1使用物理GPU2的示例：

```bash
ROOT=/home/s21_yhr/lzh/MyDualTalk
SUITE="$ROOT/runs/v3_3_suite_20260911_gpu23"
CUDA_VISIBLE_DEVICES=GPU-5b497823-4a84-bde7-5670-2172ee96245d \
PYTHONPATH="$SUITE/code_repair2" \
/home/s21_yhr/miniconda3/envs/lzh_DDG/bin/python -u -B \
  "$SUITE/checkpoint_comparison_20260911/compare_staged_v33_checkpoints.py" \
  --reference "$ROOT/runs/v3_2_1_staged_20260910_gpu23/formal/best.pt" \
  --protocol-checkpoint "$SUITE/02_vector_only/best_vector.pt" \
  --fixed-validation "$SUITE/02_vector_only/fixed_validation.pt" \
  --fixed-calibration "$SUITE/02_vector_only/fixed_calibration.pt" \
  --model "old9750=$ROOT/runs/v3_2_1_staged_20260910_gpu23/formal/best.pt" \
  --model "vector_only=$SUITE/02_vector_only/best_vector.pt" \
  --model "joint025=$SUITE/02_joint025/best_vector.pt" \
  --model "joint100=$SUITE/02_joint100/best_vector.pt" \
  --output "$SUITE/checkpoint_comparison_joint_final" \
  --device cuda:0 --execution optimized --horizon 32 --cpu-threads 2
```

后续模型训练完成后，应使用新的输出目录保存最终比较，保留本次中途快照。
这条命令没有定时运行功能；不存在的权重会明确记录为未评测。

向量MSE严格沿用已有输入块时间格的32秒预测，保留教师freshness mask。
真实语义选择句末之前至少32秒的最近完整输入块，使用实际的32至不足33秒间隔推进。
在单个时距，每个真实终点只计一次；分类F1/UAR按当前标签集合中出现的类宏平均。
类别、VAD和强度分别使用有效标签mask，未标注的值不按0处理。

MSE与真实标签不是同一目标集合；每个集合在不同模型间完全相同。
DualTalk在本次验证清单中没有32秒有效向量目标，也没有真实情感标签。
EmotionTalk只在valence维度参与VAD，IEMOCAP保留三个真实维度。

强基线只在固定训练校准清单拟合。固定起点对照共享拟合结果；在线对照重新适配各自起点分布。
连续均值收缩的rate在本入口仅用32秒训练目标拟合，因此不要与训练日志中联合六个时距拟合的rate混称为同一基线。

`comparison.json`记录汇总、checkpoint哈希、坐标和清单哈希。
`模型_fixed.json`和`模型_online.json`记录误差总和、有效计数、混淆矩阵及逐真实终点的learned/hold预测。
`report_checkpoint_comparison.py`独立重算指标，并在数据域内按对话做配对bootstrap。
这些区间只衡量当前验证对话的差异，不替代独立测试集和多个训练种子。

测试：

```bash
python -m pytest tests/test_checkpoint_comparison_v33.py -q
```
