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

双人交互过程表示为一个结构化情感状态：

$$
Z_t=
\left[
z_t^U,\,
z_t^A,\,
r_t
\right]
$$

其中：

* $z_t^U$：User 在第 $t$ 轮的统一情感状态；
* $z_t^A$：Avatar 在第 $t$ 轮的统一情感状态；
* $r_t$：双方当前的关系与耦合状态。

统一情感状态 $z_t^p$ 同时承担三个功能：

1. 统一 Audio、Face、Text 中的情感证据；
2. 保存前面多轮交互形成的持续情感状态；
3. 作为预测未来情感轨迹的动力学状态。

因此，统一情感表征不是耦合模型前面的独立特征模块，而是耦合动力系统持续演化的内部状态。

---

## 3. 多模态情感观测

对人物 $p\in\{U,A\}$，第 $t$ 轮原始多模态输入记为：

$$
X_t^p=
\left\{
X_t^{p,audio},
X_t^{p,video},
X_t^{p,text}
\right\}
$$

多模态观测编码器将原始输入分解为：

$$
O_t^p=
\left[
o_t^{p,aff},
o_t^{p,event},
o_t^{p,action}
\right]
$$

其中：

* $o_t^{p,aff}$：当前可直接观察到的情感证据，例如 VAD、情感强度、AU、语音韵律和变化趋势；
* $o_t^{p,event}$：当前事件和语义内容，例如 Appraisal、事件类型和语义上下文；
* $o_t^{p,action}$：人物 $p$ 通过文本、语气、表情和动作向 partner 实施的交互行为。

这里将三类信息分开，是为了明确它们在动力学中的职责：

$$
\boxed{
\begin{aligned}
o^{aff},o^{event}&:\quad \text{描述人物自身当前受到的情感证据和事件刺激}\\
o^{action}&:\quad \text{描述人物向 partner 发出的交互行为}
\end{aligned}
}
$$

人物身份信息单独表示为 $d_p$，只用于估计个人情感基线和衰减时间尺度：

$$
b_p=f_b(d_p)
$$

$$
\tau_p=f_\tau(d_p)
$$

其中 $b_p$ 表示人物 $p$ 的个人情感基线，$\tau_p$ 控制不同 latent 维度的自然衰减速度。人物身份不直接进入当前情感识别和 partner coupling。训练中通过跨人物情感对比、条件身份对抗和 Speaker-disjoint 测试，减少统一情感状态中的身份、背景和录音环境泄漏。

---

## 4. 结构化双人情感动力学

### 4.1 相对于个人基线的状态

为显式建模情感状态向个人基线的自然恢复，定义：

$$
\boxed{
\bar z_t^p=z_t^p-b_p
}
$$

其中 $\bar z_t^p$ 表示人物 $p$ 当前状态相对于个人正常情感基线的偏移。

因此，后续动力学主要描述：

$$
\boxed{
\text{情感偏移如何持续、衰减、积累，以及如何受到 partner 影响}
}
$$

最终完整情感状态由基线和状态偏移组成：

$$
\boxed{
z_t^p=b_p+\bar z_t^p
}
$$

### 4.2 自身动力学：持续与衰减

人物自身的状态偏移通过 $D_t^p$ 演化：

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

在没有新的情感刺激和 partner influence 时：

$$
\boxed{
\bar z_{t+1}^p=D_t^p\bar z_t^p
}
$$

因为：

$$
0<e^{-\Delta t/\tau_{p,i}}<1
$$

所以：

$$
\bar z_t^p\rightarrow 0
$$

等价于：

$$
z_t^p\rightarrow b_p
$$

$\tau_{p,i}$ 越小，对应状态维度衰减越快；$\tau_{p,i}$ 越大，对应状态维度保留越久。通过多个可学习时间尺度，同一个统一情感状态可以同时保留短时情感反应、中期 emotion 和长期 mood 偏移。

### 4.3 自身当前刺激

人物自身当前观察到的情感证据和事件定义为：

$$
u_t^p=
\left[
o_t^{p,aff};
o_t^{p,event}
\right]
$$

其中 $[\,;\,]$ 表示向量拼接。

自身当前刺激通过 $B_p$ 注入状态：

$$
B_pu_t^p
$$

连续多轮刺激会通过递归动力学产生积累。例如：

$$
\bar z_{t+1}^p=D_t^p\bar z_t^p+B_pu_t^p
$$

$$
\bar z_{t+2}^p
=(D_t^p)^2\bar z_t^p
+D_t^pB_pu_t^p
+B_pu_{t+1}^p
$$

因此，较慢时间尺度中的早期刺激不会立即消失，新刺激又会继续注入，从而形成跨多轮积累。

### 4.4 发送方情感影响信号

对于施加影响的人物 $q$，先将其当前情感状态偏移和本轮交互行为融合为一个**发送方影响信号**：

$$
\boxed{
s_t^q
=
\phi_q
\left(
\bar z_t^q,
o_t^{q,action}
\right)
}
$$

其中：

* $s_t^q$：人物 $q$ 在第 $t$ 轮向 partner 发出的情感影响信号；
* $\phi_q$：可学习的融合函数，可以使用 MLP 实现；
* $\bar z_t^q$：发送方当前相对于自身基线的情感状态；
* $o_t^{q,action}$：发送方当前实施的文本、语气、表情和动作。

一个直接的实现为：

$$
\phi_q
\left(
\bar z_t^q,
o_t^{q,action}
\right)
=
\operatorname{MLP}_{\phi,q}
\left(
[\bar z_t^q;o_t^{q,action}]
\right)
$$

这里的 $\phi_q$ 只回答一个问题：

$$
\boxed{
\text{发送方这一轮“发出了什么样的情感影响”}
}
$$

例如，Avatar 当前处于 calm 状态，并采用安慰语句、柔和语气和支持性表情，则 $s_t^A$ 表示这一整组多模态行为共同形成的“安慰/平复”影响信号。

### 4.5 接收方动态耦合算子

发送方发出同样的信号，对不同接收方、不同关系阶段可能产生不同作用。因此再定义接收方相关的动态耦合算子：

$$
\boxed{
C_t^{q\rightarrow p}
=
P_p
\operatorname{diag}
\left(
g^{q\rightarrow p}
(r_t,\bar z_t^p)
\right)
Q_q^\top
}
$$

其中：

* $r_t$：双方当前关系与耦合状态；
* $\bar z_t^p$：接收方当前相对于自身基线的情感状态；
* $g^{q\rightarrow p}$：根据关系和接收方状态动态生成的耦合门；
* $Q_q^\top$：将发送方影响信号投影到若干共享的 influence channels；
* $P_p$：将 influence channels 映射回接收方的情感状态空间。

若设置 $k$ 个 influence channels，则可以取：

$$
Q_q^\top\in\mathbb R^{k\times d_s},
\qquad
g^{q\rightarrow p}\in\mathbb R^k,
\qquad
P_p\in\mathbb R^{d\times k}
$$

因此：

$$
C_t^{q\rightarrow p}\in\mathbb R^{d\times d_s}
$$

$g^{q\rightarrow p}$ 的元素可以取正值或负值，用于表示不同 influence channel 对接收方状态的增强、抑制或反向作用。

这里 $C_t^{q\rightarrow p}$ 只回答一个问题：

$$
\boxed{
\text{接收方在当前状态和关系下，会怎样响应这个影响信号}
}
$$

两个方向独立建模：

$$
C_t^{A\rightarrow U}
\neq
C_t^{U\rightarrow A}
$$

因此可以表达非对称影响，例如：

* Avatar calm $\rightarrow$ User arousal 降低；
* User angry $\rightarrow$ Avatar concern 上升；
* Avatar dismissive $\rightarrow$ User negative valence 增强；
* 相同安慰行为在不同关系状态下产生不同效果。

### 4.6 Partner influence

发送方影响信号经过接收方耦合算子后，得到对接收方的实际状态增量：

$$
\boxed{
\Delta_{q\rightarrow p,t}
=
C_t^{q\rightarrow p}s_t^q
}
$$

将前面的定义展开：

$$
\boxed{
\Delta_{q\rightarrow p,t}
=
P_p
\operatorname{diag}
\left(
g^{q\rightarrow p}(r_t,\bar z_t^p)
\right)
Q_q^\top
\phi_q
\left(
\bar z_t^q,o_t^{q,action}
\right)
}
$$

因此整个 partner influence 被明确分成两步：

$$
\boxed{
\underbrace{\phi_q}_{\text{发送方发出了什么}}
\quad\longrightarrow\quad
\underbrace{C_t^{q\rightarrow p}}_{\text{接收方如何响应}}
\quad\longrightarrow\quad
\underbrace{\Delta_{q\rightarrow p,t}}_{\text{最终状态变化}}
}
$$

### 4.7 完整双人状态转移

对于 $p\in\{U,A\}$、$q\neq p$，统一写为：

$$
\boxed{
\bar z_{t+1}^{p,-}
=
D_t^p\bar z_t^p
+
B_pu_t^p
+
\Delta_{q\rightarrow p,t}
}
$$

展开到 User 和 Avatar：

$$
\boxed{
\begin{aligned}
\bar z_{t+1}^{U,-}
&=
D_t^U\bar z_t^U
+B_Uu_t^U
+\Delta_{A\rightarrow U,t}
\\
\bar z_{t+1}^{A,-}
&=
D_t^A\bar z_t^A
+B_Au_t^A
+\Delta_{U\rightarrow A,t}
\end{aligned}
}
$$

然后恢复完整状态：

$$
\boxed{
z_{t+1}^{p,-}=b_p+\bar z_{t+1}^{p,-}
}
$$

上标 $-$ 表示该状态是在尚未看到第 $t+1$ 轮真实多模态反馈时得到的先验预测状态。

这个公式可以直接理解为：

$$
\boxed{
\text{下一轮状态偏移}
=
\text{自身历史残留}
+
\text{自身当前刺激}
+
\text{partner influence}
}
$$

### 4.8 关系状态更新

关系状态 $r_t$ 也需要随交互持续更新，否则无法描述“同一句安慰在不同关系阶段产生不同作用”。定义：

$$
\boxed{
r_{t+1}
=
F_r
\left(
r_t,
\bar z_t^U,
\bar z_t^A,
o_t^{U,action},
o_t^{A,action}
\right)
}
$$

一个具体实现可以采用 GRU：

$$
\boxed{
r_{t+1}
=
\operatorname{GRU}_r
\left(
r_t,
[
\bar z_t^U;
\bar z_t^A;
o_t^{U,action};
o_t^{A,action}
]
\right)
}
$$

$r_{t+1}$ 将参与后续轮次的 $g^{q\rightarrow p}$ 和 $C_t^{q\rightarrow p}$ 生成，因此连续的支持、冲突、忽视或修复行为会逐步改变之后的情感耦合强度。

---

## 5. 多模态观测校正

前面的动力学首先根据历史状态、当前事件和双方行为产生第 $t+1$ 轮的预测状态：

$$
Z_{t+1}^{-}=F_\theta(Z_t,O_t)
$$

当第 $t+1$ 轮真实 Audio、Face、Text 到达后，不直接用完整的 $O_{t+1}$ 校正状态，而只提取与情感状态直接对应的 affective evidence：

$$
\boxed{
y_{t+1}^p
=
E_{aff}(X_{t+1}^p)
}
$$

其中 $y_{t+1}^p$ 可以包含经过统一投影后的 VAD、emotion embedding、AU 情感证据和语音情感证据，并映射到与状态校正兼容的维度。

模型根据预测状态、新情感证据和模态可靠性生成校正门：

$$
\boxed{
k_{t+1}^p
=
\sigma
\left(
G_p[
z_{t+1}^{p,-};
y_{t+1}^p;
M_{t+1}^{p,modality}
]
\right)
}
$$

其中：

$$
k_{t+1}^p\in[0,1]^d
$$

$H_p$ 将情感状态投影到 affective evidence 空间：

$$
\hat y_{t+1}^{p,-}=H_pz_{t+1}^{p,-}
$$

定义观测残差：

$$
e_{t+1}^p
=
y_{t+1}^p-H_pz_{t+1}^{p,-}
$$

通过一个可学习映射 $R_p$ 将观测残差映射回 latent state 空间，并完成校正：

$$
\boxed{
z_{t+1}^p
=
z_{t+1}^{p,-}
+
k_{t+1}^p
\odot
R_p
\left(
y_{t+1}^p-H_pz_{t+1}^{p,-}
\right)
}
$$

这里：

* 当音视频证据清晰、模态完整时，$k_{t+1}^p$ 较大，模型更多利用当前真实观测修正状态；
* 当视频遮挡、音频噪声较大或文本缺失时，$k_{t+1}^p$ 较小，模型更多依赖历史状态和动力学预测。

因此系统形成持续的：

$$
\boxed{
\text{预测}
\rightarrow
\text{真实情感观测}
\rightarrow
\text{校正}
\rightarrow
\text{再次预测}
}
$$

循环。

这里将 $o^{event}$ 和 $o^{action}$ 排除在 observation residual 之外，因为事件内容和交互行为负责推动状态变化，而不是作为“当前情感状态本身”的直接测量值。

---

## 6. 情感变化机制

模型中的不同情感现象具有明确对应关系：

| 情感现象 | 模型中的实现 |
| --- | --- |
| 持续 | $D_t^p\bar z_t^p$ 中较大的时间常数使相对于个人基线的情感偏移跨多轮保留 |
| 衰减 | $e^{-\Delta t/\tau}$ 逐步缩小 $\bar z_t^p=z_t^p-b_p$，使 $z_t^p$ 自然回归个人基线 $b_p$ |
| 积累 | 连续多轮 $B_pu_t^p$ 注入慢时间尺度状态，使情感偏移逐步累积 |
| 增强 | $\Delta_{q\rightarrow p,t}$ 与接收方当前情感偏移方向相同 |
| 抑制 | $\Delta_{q\rightarrow p,t}$ 与接收方当前情感偏移方向相反 |
| 转移 | $C_t^{q\rightarrow p}s_t^q$ 将发送方影响信号转化为接收方状态变化 |
| 关系变化 | $r_t\rightarrow r_{t+1}$，并进一步改变后续 $g^{q\rightarrow p}$ 和耦合算子 |
| 延迟影响 | 慢时间尺度状态跨多轮保留早期事件与 partner influence |

增强可以定义为：

$$
\boxed{
\cos
\left(
\bar z_t^p,
\Delta_{q\rightarrow p,t}
\right)>0
}
$$

抑制可以定义为：

$$
\boxed{
\cos
\left(
\bar z_t^p,
\Delta_{q\rightarrow p,t}
\right)<0
}
$$

其中：

$$
\Delta_{q\rightarrow p,t}=C_t^{q\rightarrow p}s_t^q
$$

该分析最好在经过 VAD、情感类别或其他 affective supervision 约束后的情感状态子空间中进行，以保证 latent 方向具有可解释的情感含义。

---

## 7. 未来联合情感轨迹预测

### 7.1 单步预测

模型首先进行单步状态预测：

$$
\boxed{
\hat Z_{t+1}=F_\theta(Z_t,O_t)
}
$$

其中 $F_\theta$ 就是第 4 节定义的同一套自身动力、当前刺激、方向性耦合和关系状态更新。

### 7.2 多步动力学展开

为了检验同一个状态转移函数能否在较长跨度保持稳定，训练时进行多步 rollout：

$$
\boxed{
\hat Z_{t+h}
=
F_\theta^{(h)}
\left(
Z_t;
O_{t:t+h-1}
\right)
}
$$

这里显式写出 $O_{t:t+h-1}$，因为多轮交互中的未来事件和双方行为本身属于外生输入，不能在公式中无条件消失。

训练阶段可以使用真实序列中的中间交互输入进行 teacher-forced rollout：

$$
\hat Z_{t+2}
=
F_\theta
\left(
\hat Z_{t+1},O_{t+1}
\right)
$$

并对多个跨度同时监督：

$$
h\in\{1,2,4,8,16,32\}
$$

对应预测：

$$
\left(
\hat z_{t+h}^U,
\hat z_{t+h}^A
\right)
$$

这里的目标是验证**同一个情感动力系统在长时间尺度上的状态转移一致性**，而不是额外训练一个独立的“32 轮预测器”。

### 7.3 候选 Avatar 行为的条件轨迹预测

在 Avatar 交互阶段，未来的 User 事件本身不可提前知道，因此更适合把长期预测定义成**条件轨迹预测**：给定不同候选 Avatar 行为，比较其可能造成的后续情感变化。

对第 $i$ 个 Avatar 候选行为 $a_{t,i}^A$：

$$
\boxed{
\hat Z_{t+1}^{(i)}
=
F_\theta
\left(
Z_t,
O_t^{U},
a_{t,i}^A
\right)
}
$$

如果给定一个候选 Avatar 行为序列 $a_{t:t+H-1,i}^A$，则可进行条件 rollout：

$$
\boxed{
\hat Z_{t+1:t+H}^{(i)}
=
F_\theta^{(1:H)}
\left(
Z_t;
a_{t:t+H-1,i}^A
\right)
}
$$

用于比较安慰、鼓励、中性回应、转移话题等策略对 User 情感轨迹的不同影响。

第一阶段工作中，应明确区分：

* **训练时多步动力学展开**：可以使用真实中间交互输入，检验 transition 的长期稳定性；
* **在线决策时的候选行为预测**：条件于给定 Avatar 候选行为，不假设模型提前知道未来真实事件。

---

## 8. 训练方法

整个方法是一套模型，使用两个训练阶段优化。

### Phase A：自身情感动力预训练

暂时关闭 partner influence：

$$
\Delta_{A\rightarrow U,t}
=
\Delta_{U\rightarrow A,t}
=
0
$$

等价地可以令：

$$
C_t^{A\rightarrow U}
=
C_t^{U\rightarrow A}
=
0
$$

训练内容包括：

* 多模态统一情感状态；
* 缺失模态恢复；
* 当前情感识别；
* 自身未来情感预测；
* 多时间尺度持续和衰减；
* 身份和背景信息去除。

这一阶段训练最终模型中的 Observation Encoder、统一状态 $z_t^p$、个人基线 $b_p$、时间尺度 $\tau_p$ 和自身动力块 $D_t^p$，不是额外增加一个独立上游模块。

### Phase B：双人耦合联合训练

打开：

$$
\Delta_{A\rightarrow U,t},
\qquad
\Delta_{U\rightarrow A,t}
$$

联合训练：

* 双方未来联合情感轨迹；
* 发送方影响信号 $s_t^q$；
* 动态方向性耦合 $C_t^{q\rightarrow p}$；
* 关系状态 $r_t$；
* 多跨度状态 rollout；
* Matched Partner Intervention。

Phase A 参数不完全冻结，而是使用较小学习率继续联合优化，使统一情感状态逐步适应双人动力学。

---

## 9. 耦合识别目标

仅设计 $C_t^{q\rightarrow p}$ 并不能保证模型真正使用 partner information。模型可能依赖接收方自身历史完成预测，因此需要显式的 partner intervention 训练和评测。

对于真实 Avatar 行为：

$$
a_t^A
$$

构造匹配替代行为：

$$
\tilde a_t^A
$$

替代行为来自相似 User 状态、相似话题和相似对话阶段，但采用不同回应策略的样本。

定义：

$$
D_{real}
=
D
\left(
F(Z_t,a_t^A),
Z_{t+1}^{target}
\right)
$$

$$
D_{cf}
=
D
\left(
F(Z_t,\tilde a_t^A),
Z_{t+1}^{target}
\right)
$$

要求真实行为产生的预测更接近真实下一状态：

$$
\boxed{
D_{real}<D_{cf}
}
$$

可以使用 margin ranking loss：

$$
\boxed{
L_{cf}
=
\max
\left(
0,
m+D_{real}-D_{cf}
\right)
}
$$

其中 $m>0$ 为 margin。

该目标迫使模型学习：

$$
\boxed{
\text{不同 partner 行为}
\rightarrow
\text{不同 influence signal}
\rightarrow
\text{不同未来情感变化}
}
$$

最终损失可以写为：

$$
\boxed{
L
=
L_{state}
+
\lambda_{traj}L_{joint\ trajectory}
+
\lambda_{cf}L_{cf}
+
\lambda_{aux}L_{aux}
}
$$

其中 $L_{aux}$ 可以包含缺失模态恢复、身份去除和其他辅助训练目标。

---

## 10. 数据安排

| 数据集 | 主要用途 |
| --- | --- |
| EmotionTalk | 中文多模态统一情感状态和双人动力学主训练 |
| IEMOCAP | 英文双人情感与跨数据集验证 |
| K-EmoCon | 连续 V/A 轨迹和方向性耦合验证 |
| DualTalk | Avatar 说话、倾听、表情和头动生成 |

第一篇工作暂不把精确历史事件检索作为核心任务。这里的“长时情感”主要指持续状态、积累、衰减、关系变化和跨多轮 partner influence。具体历史事件记忆和检索可以作为后续扩展。

---

## 11. 评测设计

### 11.1 统一情感状态

* Emotion F1、UAR；
* VAD CCC；
* 缺失模态鲁棒性；
* Speaker-disjoint 测试；
* Identity/Session Leakage Probe；
* 跨数据集迁移。

### 11.2 耦合动力学

* Self-only 与 Dyadic Prediction；
* 删除 partner influence；
* Random Partner；
* Matched Partner Intervention；
* 固定耦合与动态耦合；
* 对称耦合与方向性耦合；
* Cross-Attention 与结构化耦合算子。

定义：

$$
\boxed{
\mathrm{PartnerGain}_{q\rightarrow p}
=
D(\hat Z_{self},Z_{target})
-
D(\hat Z_{dyad},Z_{target})
}
$$

若：

$$
\mathrm{PartnerGain}_{q\rightarrow p}>0
$$

说明加入 partner history 和 partner behavior 后，对未来状态的预测误差下降，partner information 提供了额外预测价值。

### 11.3 长时情感变化

* 预测未来 1/2/4/8/16/32 轮；
* 每轮重置状态与完整保留状态；
* 去掉长时间尺度；
* 连续负面刺激的积累；
* 无新刺激时向个人基线的自然衰减；
* Avatar 安慰后的恢复轨迹；
* 不同 Avatar 候选行为造成的预测差异；
* 固定 $r_t$ 与动态更新 $r_t$ 的对比。

### 11.4 Avatar 生成

将 $z_t^A$ 通过 AdaLN 或 FiLM 注入 DualTalk 的 Expressive Synthesis Module，评估：

* 情感一致性；
* 多轮表情连续性；
* lip-sync；
* listening behavior；
* empathy；
* naturalness；
* user satisfaction。

---

## 12. 与近邻工作的区别

AffectVerse 主要根据单个 clip 内的 Audio–Video 历史预测未来模态 latent，学习短期跨模态情感动态。

AffectLoop 维护 speaker 和 robot listener 的两条情感流，并用于条件化 LLM 回复和机器人行为。

本方法研究：

$$
\boxed{
\text{结构化、持续、方向性、可干预验证的双人情感状态转移}
}
$$

核心区别包括：

* 统一情感表征本身就是持续动力状态；
* 使用个人基线 $b_p$ 和多时间尺度 $D_t^p$ 建模持续与自然衰减；
* 将发送方影响信号 $s_t^q$ 与接收方耦合算子 $C_t^{q\rightarrow p}$ 分离；
* 使用两个方向独立的耦合算子建模非对称 partner influence；
* 使用动态关系状态 $r_t$ 调节后续耦合；
* 使用预测—观测校正持续更新情感状态；
* 使用 matched intervention 检验 partner behavior 的额外预测作用；
* 使用同一状态转移函数进行多跨度联合情感轨迹 rollout，而不是只跟踪两条情感标签。

---

## 13. 预期贡献

1. 提出一个面向长期人–Avatar 交互的统一双人情感状态空间，将多模态情感表征和持续动力状态统一为同一个 latent state。

2. 提出具有个人基线和多时间尺度衰减的自身情感动力学，使短时反应、中期情绪和长期 mood 可以在同一状态空间中持续和自然恢复。

3. 提出动态、非对称的双人耦合机制，将“发送方发出的影响信号”和“接收方如何响应”显式分离，并通过关系状态动态调节影响强度。

4. 提出基于 Matched Partner Intervention 的耦合学习与评测方法，检验 partner behavior 对未来情感轨迹的额外预测价值。

5. 将学习到的 Avatar 情感状态接入 DualTalk，实现具有长期情感连续性的说话、倾听和非语言反馈。

整个方法的核心状态转移可以概括为：

$$
\boxed{
\begin{aligned}
\bar z_t^p
&=z_t^p-b_p
\\
s_t^q
&=\phi_q(\bar z_t^q,o_t^{q,action})
\\
C_t^{q\rightarrow p}
&=P_p\operatorname{diag}
\left(g^{q\rightarrow p}(r_t,\bar z_t^p)\right)Q_q^\top
\\
\Delta_{q\rightarrow p,t}
&=C_t^{q\rightarrow p}s_t^q
\\
\bar z_{t+1}^{p,-}
&=D_t^p\bar z_t^p+B_pu_t^p+\Delta_{q\rightarrow p,t}
\\
z_{t+1}^{p,-}
&=b_p+\bar z_{t+1}^{p,-}
\\
r_{t+1}
&=F_r(r_t,\bar z_t^U,\bar z_t^A,o_t^{U,action},o_t^{A,action})
\\
z_{t+1}^p
&=z_{t+1}^{p,-}
+k_{t+1}^p\odot
R_p(y_{t+1}^p-H_pz_{t+1}^{p,-})
\end{aligned}
}
$$

其中最核心的逻辑是：

$$
\boxed{
\text{下一轮情感}
=
\text{个人基线}
+
\text{历史情感残留}
+
\text{当前自身刺激}
+
\text{partner influence}
+
\text{真实观测校正}
}
$$

研究重点不是增加更多彼此独立的情感模块，而是学习一个能够统一感知、持续更新、分解双方影响、适应关系变化并预测长期联合情感轨迹的双人情感动力系统。
