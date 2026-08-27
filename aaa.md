# 面向长时人–Avatar交互的耦合情感状态空间学习

## 1. 研究问题

现有交互式 Avatar 系统通常采用逐轮处理方式：

$$
\text{当前音视频文本}
\rightarrow
\text{当前情绪}
\rightarrow
\text{生成本轮回复和动作}
$$

这种方法难以描述双方情感在多轮交互中的持续、积累、衰减、增强、抑制和转移。例如，用户连续遭受负面刺激后，短时间的安慰可能降低其当前唤醒程度，但不会立刻消除长期积累的负面心境；同一句安慰在不同关系阶段也可能产生不同作用。

本研究拟解决的问题是：

> 如何从用户与 Avatar 的多模态交互中学习一个持续演化的双人统一情感状态，并显式建模双方之间动态、非对称的情感影响，从而预测长期联合情感轨迹并驱动具有情感连续性的 Avatar？

---

## 2. 核心假设

双人交互过程可以表示为一个结构化情感状态空间：

$$
Z_t=
\left[
z_t^U,\,
z_t^A,\,
r_t
\right]
$$

其中：

* \(z_t^U\)：User 在第 \(t\) 轮的统一情感状态；
* \(z_t^A\)：Avatar 在第 \(t\) 轮的统一情感状态；
* \(r_t\)：双方当前的关系与耦合状态。

这里的 \(z_t^p\) 同时承担三个功能：

1. 统一 Audio、Face、Text 中的情感证据；
2. 保存前面多轮交互形成的持续情感状态；
3. 作为预测双方未来情感轨迹的动力学状态。

因此，统一情感表征不再是耦合模型前面的独立特征模块，而是耦合动力系统本身的状态。

---

## 3. 多模态情感观测

每个交互事件包含：

$$
X_t^p=
\left\{
X_t^{p,audio},
X_t^{p,video},
X_t^{p,text}
\right\}
$$

多模态观测编码器将其分解为：

$$
O_t^p=
\left[
o_t^{p,aff},
o_t^{p,event},
o_t^{p,action}
\right]
$$

其中：

* \(o_t^{p,aff}\)：当前观察到的情感证据，如 VAD、情感强度和变化趋势；
* \(o_t^{p,event}\)：当前事件、语义内容和 Appraisal 信息；
* \(o_t^{p,action}\)：这个人通过文本、语气、表情和动作向对方实施的交互行为。

人物身份信息单独表示为 \(d_p\)，只用于估计个人情感基线和衰减速度：

$$
b_p=f_b(d_p)
$$

$$
\tau_p=f_\tau(d_p)
$$

人物身份不直接进入当前情感识别和耦合预测。训练中通过跨人物情感对比、条件身份对抗和 Speaker-disjoint 测试，减少统一情感状态中的身份、背景和录音环境泄漏。

---

## 4. 结构化双人情感动力学

为显式建模情感状态向个人基线的自然恢复，定义人物 \(p\) 相对于个人情感基线的状态偏移：

$$
\bar z_t^p=z_t^p-b_p
$$

其中 \(b_p\) 为人物 \(p\) 的个人情感基线。后续动力学主要描述偏离个人基线的部分
\(\bar z_t^p\) 如何持续、衰减以及受到对方影响。

双人状态按照以下结构化转移矩阵演化：

$$
\begin{bmatrix}
\bar z_{t+1}^{U,-}\\
\bar z_{t+1}^{A,-}
\end{bmatrix}
=
\begin{bmatrix}
D_t^U & C_t^{A\rightarrow U}\\
C_t^{U\rightarrow A} & D_t^A
\end{bmatrix}
\begin{bmatrix}
\bar z_t^U\\
\bar z_t^A
\end{bmatrix}
+
B
\begin{bmatrix}
O_t^U\\
O_t^A
\end{bmatrix}
$$

$$
z_{p+1}^{p,-}=b_p+\bar z_{t+1}^{p,-}
$$

$$
r_{t+1}=
GRU_r
(
r_t,[
o_t^{U,action},o_t^{A,action},\bar z_t^U,\bar z_t^A
]
)
$$

上标 \(-\) 表示这是尚未看到下一轮真实多模态反馈时的预测状态。

### 自身动力学

对角块：

$$
D_t^U,\qquad D_t^A
$$

描述每个人相对于个人情感基线的状态偏移如何持续和衰减。其内部具有多个可学习的时间尺度：

$$
D_t^p
=
\operatorname{diag}
\left(
e^{-\Delta t/\tau_{p,1}},
\ldots,
e^{-\Delta t/\tau_{p,d}}
\right)
$$


$$
\bar z_{t+1}^p=D_t^p\bar z_t^p
$$
在没有新观测和 partner influence 时，由于

$$ 0<e^{-\Delta t/\tau_{p,i}}<1 $$

因此：

$$ \bar z_t^p\rightarrow0 $$

等价于：

$$ z_t^p\rightarrow b_p $$

部分相对基线的 latent 偏移在一两轮内快速衰减，部分偏移可以持续十几轮或几十轮。因此，短时情感反应、中期情绪和长期心境偏移可以在同一状态空间中由不同时间常数表示。

### 方向性耦合动力学

非对角块：

$$
C_t^{A\rightarrow U},
\qquad
C_t^{U\rightarrow A}
$$

分别描述 Avatar 对 User、User 对 Avatar 的动态影响。两个方向不要求对称：

$$
C_t^{A\rightarrow U}
\neq
C_t^{U\rightarrow A}
$$

耦合算子根据关系状态、接收方状态和施加方行为动态生成：

$$
C_t^{A\rightarrow U}
=
P_U
\operatorname{diag}
\left(
g^{A\rightarrow U}
(r_t,\bar z_t^U,o_t^{A,action})
\right)
Q_A^\top
$$

其中 \(g_t\) 可以取正值或负值，分别表示增强和抑制作用。

这种设计允许模型学习：

* Avatar calm \(\rightarrow\) User arousal 降低；
* User angry \(\rightarrow\) Avatar concern 上升；
* Avatar dismissive \(\rightarrow\) User negative valence 增强；
* 相同安慰行为在不同关系状态下具有不同作用。

---

## 5. 多模态观测校正

状态转移首先根据旧状态和双方行为产生预测：

$$
Z_{t+1}^{-}=F_\theta(Z_t,O_t)
$$

当下一轮真实音频、面部和文本到达后，观测编码器产生新的情感证据：

$$
\widetilde O_{t+1}
=
E_{\mathrm{obs}}(X_{t+1})
$$

模型根据模态可靠性生成校正门：

$$
K_{t+1}
=
\sigma
\left(
G[
Z_{t+1}^{-},
\widetilde O_{t+1},
M_{t+1}^{modality}
]
\right)
$$

并校正预测状态：

$$
Z_{t+1}
=
Z_{t+1}^{-}
+
K_{t+1}
\odot
\left(
\widetilde O_{t+1}
-
HZ_{t+1}^{-}
\right)
$$

当新音视频证据清晰时，模型更多相信当前观测；当视频遮挡、音频噪声较大或文本缺失时，模型更多依赖历史状态和动力学预测。

整个系统由此形成持续的：

$$
\text{预测}
\rightarrow
\text{观测}
\rightarrow
\text{校正}
\rightarrow
\text{再次预测}
$$

循环。

---

## 6. 情感变化机制

模型中的不同情感现象具有明确对应关系：

| 情感现象 | 模型中的实现                                      |
| ---- | ------------------------------------------- |
| 持续   | \(D_t^p(z_t^p-b_p)\) 中较大的时间常数使相对于个人基线的情感偏移跨多轮保留                      |
| 衰减   | \(e^{-\Delta t/\tau}\) 逐步缩小 \(z_t^p-b_p\)，使 \(z_t^p\) 自然回归个人基线 \(b_p\)   |
| 积累   | 连续多轮情感观测和事件输入反复注入慢时间尺度状态，使 \(z_t^p-b_p\) 的偏移逐步累积                             |
| 增强   | 耦合增量与接收方当前状态方向相同                            |
| 抑制   | 耦合增量与接收方当前状态方向相反                            |
| 转移   | \(C_t^{q\rightarrow p}\) 将对方状态和行为映射为自己的状态变化 |
| 关系变化 | \(r_t\) 改变后续耦合算子                            |
| 延迟影响 | 慢时间尺度状态跨多轮保留早期影响                            |

增强可以通过下式定义：

$$
\cos
\left(
z_t^p-b_p,
\Delta_{q\rightarrow p,t}
\right)>0
$$

抑制可以定义为：

$$
\cos
\left(
z_t^p-b_p,
\Delta_{q\rightarrow p,t}
\right)<0
$$

其中：

$$
\Delta_{q\rightarrow p,t}
=
C_t^{q\rightarrow p}
\phi(z_t^q-b_q,o_t^{q,action})
$$

---

## 7. 未来联合情感轨迹预测

模型使用同一个状态转移函数预测未来：

$$
\hat Z_{t+1}=F_\theta(Z_t,O_t)
$$

$$
\hat Z_{t+h}
=
F_\theta^{(h)}(Z_t)
$$

训练时同时预测：

$$
h\in\{1,2,4,8,16,32\}
$$

轮后的 User 和 Avatar 状态：

$$
\left(
\hat z_{t+h}^U,
\hat z_{t+h}^A
\right)
$$

未来预测不是额外增加一个独立模型，而是对同一耦合状态转移的多步展开。

在 Avatar 交互阶段，还可以输入不同候选行为：

$$
\hat Z_{t+1:t+H}^{(i)}
=
F_\theta
\left(
Z_t,a_{t,i}^A
\right)
$$

比较安慰、鼓励、中性回应或转移话题等行为可能造成的不同用户情感轨迹。

---

## 8. 训练方法

整个方法是一套模型，使用两个训练阶段进行优化。

### Phase A：自身情感动力预训练

暂时关闭非对角耦合块：

$$
C_t^{A\rightarrow U}
=
C_t^{U\rightarrow A}
=
0
$$

训练：

* 多模态统一情感状态；
* 缺失模态恢复；
* 当前情感识别；
* 自身未来情感预测；
* 多时间尺度持续和衰减；
* 身份和背景去除。

这一阶段训练最终模型中的 Observation Encoder、统一状态 \(z_t\) 和自身动力块 \(D_t\)，不是额外增加一个上游模块。

### Phase B：双人耦合联合训练

打开：

$$
C_t^{A\rightarrow U},
\qquad
C_t^{U\rightarrow A}
$$

在完整双人对话序列上联合训练：

* 双方未来联合轨迹；
* 动态方向性耦合；
* 关系状态；
* 多跨度状态预测；
* Matched Partner Intervention。

Phase A 的参数不完全冻结，而是使用较小学习率继续联合优化，使统一情感状态适应双人动力学。

---

## 9. 耦合识别目标

为了防止模型忽略 partner information，使用匹配反事实行为。

真实 Avatar 行为：

$$
a_t^A
$$

替代行为：

$$
\tilde a_t^A
$$

替代行为来自相似用户状态、相似话题和相似对话阶段，但使用不同回应策略的样本。

要求：

$$
D
\left(
F(Z_t,a_t^A),
Z_{t+1}^{target}
\right)
<
D
\left(
F(Z_t,\tilde a_t^A),
Z_{t+1}^{target}
\right)
$$

从而迫使模型学习：

$$
\text{不同 partner 行为}
\rightarrow
\text{不同未来情感变化}
$$

最终损失可以归纳为：

$$
L
=
L_{\mathrm{state}}
+
L_{\mathrm{joint\ trajectory}}
+
\lambda_{\mathrm{cf}}L_{\mathrm{matched\ intervention}}
$$

---

## 10. 数据安排

| 数据集         | 主要用途                 |
| ----------- | -------------------- |
| EmotionTalk | 中文多模态统一表征和双人动力学主训练   |
| IEMOCAP     | 英文双人情感与跨数据集验证        |
| K-EmoCon    | 连续 V/A 轨迹和方向性耦合验证    |
| DualTalk    | Avatar 说话、倾听、表情和头动生成 |

第一篇工作暂不把精确历史事件检索作为核心任务。长时情感主要指持续状态、积累、衰减和跨多轮 partner influence。具体事件记忆可以作为后续扩展。

---

## 11. 评测设计

### 统一情感状态

* Emotion F1、UAR；
* VAD CCC；
* 缺失模态鲁棒性；
* Speaker-disjoint 测试；
* Identity/Session Leakage Probe；
* 跨数据集迁移。

### 耦合动力学

* Self-only 与 Dyadic Prediction；
* 删除 partner；
* Random Partner；
* Matched Partner Intervention；
* 固定耦合与动态耦合；
* 对称耦合与方向性耦合；
* Cross-Attention 与结构化耦合算子。

定义：

$$
\mathrm{PartnerGain}_{q\rightarrow p}
=
D(\hat Z_{\mathrm{self}},Z_{\mathrm{target}})
-
D(\hat Z_{\mathrm{dyad}},Z_{\mathrm{target}})
$$

用于衡量 partner history 的额外预测价值。

### 长时情感变化

* 预测未来 1/2/4/8/16/32 轮；
* 每轮重置状态与完整保留状态；
* 去掉长时间尺度；
* 连续负面刺激的积累；
* 无新刺激时的自然衰减；
* Avatar 安慰后的恢复轨迹；
* 不同 Avatar 候选行为造成的预测差异。

### Avatar 生成

将 \(z_t^A\) 通过 AdaLN 或 FiLM 注入 DualTalk 的 Expressive Synthesis Module，评估：

* 情感一致性；
* 多轮表情连续性；
* lip-sync；
* listening behavior；
* empathy、naturalness 和 user satisfaction。

---

## 12. 与近邻工作的区别

AffectVerse 主要根据单个 clip 内的 Audio–Video 历史预测未来模态 latent，学习短期跨模态情感动态。

AffectLoop 维护 speaker 和 robot listener 的两条情感流，并用于条件化 LLM 回复和机器人行为。

本方法进一步研究：

$$
\boxed{
\text{结构化、持续、方向性、可干预验证的双人情感状态转移}
}
$$

核心区别是：

* 统一情感表征本身就是持续动力状态；
* 使用对角块建模个人多时间尺度动力；
* 使用非对角块建模双方非对称影响；
* 使用预测—观测校正持续更新状态；
* 使用 matched intervention 检验 partner behavior 的预测作用；
* 预测双方长期联合情感轨迹，而不是只跟踪两条情感标签。

---

## 13. 预期贡献

1. 提出一个面向长期人–Avatar交互的统一双人情感状态空间，将多模态情感表征和持续动力状态统一为同一个 latent state。

2. 提出动态、非对称的耦合状态转移算子，显式分解个人持续/衰减和双方增强/抑制/转移作用。

3. 提出基于匹配行为干预的耦合学习与评测方法，检验 partner behavior 对未来情感轨迹的额外预测价值。

4. 将学习到的 Avatar 情感状态接入 DualTalk，实现具有长期情感连续性的说话、倾听和非语言反馈。

整个方法可以概括为：

$$
\boxed{
\begin{aligned}
\bar Z_{t+1}^{-}
&=
\mathcal T_t\bar Z_t+B_aA_t\\

Z_{t+1}^-
&=B_Z+\bar Z_{t+1}^- \\

Z_{t+1}
&=
Z_{t+1}^{-}
+
K_{t+1}
\odot
(O_{t+1}-HZ_{t+1}^{-})\\
\hat Z_{t+1:t+H}
&=
F_\theta^{(1:H)}(Z_t)
\end{aligned}
}
$$

研究重点不是增加更多情感模块，而是学习一个能够统一感知、持续更新、解释双方影响并预测未来的双人情感动力系统。
