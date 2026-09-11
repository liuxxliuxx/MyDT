"""Build the reviewed diagnostic report from the saved read-only GPU audit."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/v3_2_1_staged_20260910_gpu23/training_curves'
data = json.loads((OUT/'shrink_gap_diagnostics.json').read_text(encoding='utf-8-sig'))
train = data['protocols']['train']['methods']
val = data['protocols']['fixed_val']['methods']
title = '动力学为何落后于均值收缩：9750步权重诊断'
source_id = 'readonly_gpu_audit'

sections = [
('summary', '已定位到系统性预测偏差、目标冲突和采样失衡', '''最佳动力学的固定起点宏平均 MSE 为 **0.00187841**，均值收缩为 **0.00167040**，动力学高 **12.45%**。在本轮训练起点上，动力学也高 **10.57%**。因此，问题已经出现在训练分布，不能只用验证集过拟合解释。

本次只读诊断显示：训练数据拟合的中心偏移校正可消除约 **46.29%** 的验证差距；4～32秒的传播增量仍偏大；四个有标签训练批次中三个出现标签梯度与向量 MSE 梯度相反。每域对话数量相同，但 DualTalk 只占训练起点的 **5.18%**。这些证据共同指向预测校准、训练目标与采样方式。

关闭预测期间的 partner 耦合使 MSE 升至 **0.00210232**。耦合有条件预测价值，但这一消融不能证明学会了安慰、长期关系或真实因果影响。'''),
('scope', '比较使用相同起点、目标和时间，校准参数仅拟合训练数据', '''检查的是2026年9月10日训练完成后的 best checkpoint（9750步）。训练起点来自最后一个训练块：192段对话、6431个起点、47590个有效角色预测查询；验证使用原来固定的24段对话、777个起点、5863个角色查询，即750464个128维坐标。

所有方法预测未来1、2、4、8、16、32秒。各数据域内部先按误差平方和／有效坐标数计算 MSE，每个时距对有效数据域等权，再对时距等权。32秒没有 DualTalk 有效目标。下文分域指标则在该域内汇总所有时距的有效坐标，两种聚合口径不可混算。

均值、收缩系数、传播幅度系数、中心偏移系数和向量偏置均只使用保存的训练起点缓存拟合。预测期间不读取未来音频、文本、视觉、事件或行为。最佳权重与原始四种预测器的验证指标已复现，差异小于2×10⁻⁸；诊断前后权重摘要相同。'''),
('comparison', '简单校准能降低误差，但还没有超过均值收缩', '''下图使用完全相同的固定验证起点。仅作输出校准，不更新网络参数。“中心偏移校正”每个时距只有一个从训练数据拟合的标量；“向量偏置校正”每个时距有一个128维训练残差均值。

中心偏移校正使 MSE 从0.00187841降至0.00178212，降低5.13%，消除原有差距的46.29%。向量偏置校正达到0.00177914，收益接近。传播幅度校正为0.00181181。这些是同一已选 checkpoint 的验证诊断，尚未做独立测试；各项改善存在重叠，不能相加。'''),
('center', '基础衰减中心与教师分布中心存在失配', '''训练教师向量都经过单位化；训练均值 μ 的范数为 **0.78310**，动力学锁定的 baseline 向量 b 的范数为 **0.41630**。二者方向余弦为0.96645，逐坐标 RMS 差为0.03496。b沿用了上游权重，在全部分阶段训练中冻结。

本次校准为 `预测 + β_h × (μ − b)`，β_h仅用训练残差最小二乘拟合，六个时距约为0.142、0.205、0.267、0.326、0.370、0.390。如此简单的校准便能消除近一半宏平均差距，说明预测存在可修正的系统性偏移。

这支持检查基础衰减中心、快慢偏移分解和输出校准。它不能单独证明“把baseline改成训练均值就能修好”：连续反馈可以维持非零偏移，完整动力学不保证收敛到baseline；输出补偿与修改内部状态方程也不等价。尤其同一全局中心修正改善EmotionTalk和IEMOCAP，却使DualTalk从0.00074249变为0.00078692。必须保留分域验证。'''),
('shrink', '均值收缩同时保留状态并压低不可靠的偏移', '''当前基线为 `μ + a_h × (s_t − μ)`。μ由训练集中34916个有效packet-role教师观测计算，每个观测只计一次；a_h按训练目标最小二乘拟合，并限制在[0,1]。六个时距的a_h依次是 **0.763、0.702、0.671、0.627、0.594、0.533**。

1秒时仍保留76%的当前偏移，32秒保留53%，所以它并没有退化成固定均值。直接预测均值的验证MSE为0.00298703，明显更差。

在平方误差下，最优点预测是给定可用信息的条件均值。未知未来事件与测量波动会让保守的偏移幅度占优。这是预测不确定性的结果，不能解释为“人的情感本来就是简单衰减”。当前训练目标在预测查询分布中的均值范数约0.786，协方差有效秩约5.92；验证有效秩约6.29。表征有较强公共分量和低维变化，但这些统计不能直接判定常量坍缩，更不能据此强行要求128个维度全部独立。

此外，每个时距独立拟合的a_h不必满足连续动力学的时间组合约束。例如a_2约0.702，a_1²约0.582；它同时吸收了状态估计与预测校准误差。动力学则需要用一套连续传播规则覆盖所有时距，未必天然包含这个基线作为可复现的特例。'''),
('amplitude', '中长时距的传播幅度仍需校准', '''固定起点，用训练数据拟合 `s_t + α_h × (预测 − s_t)`，并将α限制在[0,1]。1、2秒取1，4、8、16、32秒分别约为 **0.756、0.690、0.678、0.765**。

应用到验证后，宏平均MSE下降3.55%，仍比均值收缩高8.47%左右。4～32秒传播幅度是有证据的误差来源，但短时距的差距不能靠统一减小幅度解决。简单将所有预测归一化到单位球也不成立：本次验证MSE升至0.00198064，比原预测差5.44%。'''),
('objective', '动力学和收缩基线实际优化的目标不同', '''收缩基线只优化教师情感向量的平方误差。动力学的`prediction_loss`先计算每个有效角色的128维平方误差之和，再平均各时距，并叠加权重为1的真实标签损失：情感分类交叉熵、有效VAD误差和intensity误差。训练日志中的`future_loss`已经包含标签项，不能直接当成向量MSE。

在最佳权重上抽查六个固定训练小批次，每域两批、每批16个起点。EmotionTalk的标签梯度范数为向量梯度的1.74～2.29倍，两批梯度余弦为−0.311、+0.268；IEMOCAP为0.58～0.77倍，余弦−0.249、−0.140；DualTalk没有有效标签项。

这证实局部目标存在冲突，但样本量不足以量化它对整轮训练的贡献，也不能声称总梯度一直使MSE上升。需要受控比较“仅向量预测”和“加入标签约束”，同时报告独立情感指标。不能为了降低MSE就默认删除语义监督。'''),
('sampling', '对话数量平衡没有转化成训练起点平衡', '''最后一块每域都选64段对话。随后从所有起点均匀抽样，得到EmotionTalk **1825个（28.38%）**、IEMOCAP **4273个（66.44%）**、DualTalk **333个（5.18%）**。这是由对话长度、有效目标和预测时距共同决定的。

该域的各时距汇总MSE如下：EmotionTalk动力学0.00283516、收缩0.00266805（高6.26%）；IEMOCAP动力学0.00158977、收缩0.00132300（高20.16%）；DualTalk动力学0.00074249、收缩0.00060439（高22.85%）。DualTalk保持状态为0.00062994，动力学也未超过保持。

共享动力学主要接收较长对话的梯度；DualTalk又缺少真实情感标签，其他域的标签目标可能通过共享参数影响它。这个机制与结果一致，仍需用每域起点配额和分域损失归一化实验确认。'''),
('partner', '持续耦合有收益，收益的语义来源还没有被识别', '''保留完整原始起点，仅关闭预测期间partner条件化、relation条件化和反馈，MSE由0.00187841升至 **0.00210232**。因此，在这一冻结起点协议下，启用预测耦合使误差降低约10.65%。

它不是从头训练的无partner模型，起点本身已包含双方历史，且关闭开关同时移除了多个耦合路径。结果不能证明具体安慰策略或因果影响；部分收益也可能是在补偿公共分量、预测范数或训练中心偏差。应在中心校准后，继续比较正确partner历史、错配partner历史、仅自身状态，并保持自身起点不变。'''),
('limits', '当前证据不足以把问题归结为数据无规律或单一优化器问题', '''训练起点上的差距已经存在，故不能只归因于过拟合。旧的固定起点留出实验中，线性当前状态预测器和线性历史预测器曾超过均值收缩，支持“表征中可能还有未利用的增量”。但旧实验使用不同权重和起点，不能拿旧线性预测器的数字充当本轮比较。

代码也确实在每个1000步训练块重建AdamW，丢弃跨块动量，且没有学习率调度器；这可能影响收敛，但本次没有重训消融来证明它是主要原因。日志中记录到的梯度范数未超过裁剪阈值5，当前没有证据将差距归因于长期梯度裁剪。

本次数据包括选best使用的24段验证对话，有重复起点与时距查询；不能把750464个坐标视作独立统计样本。没有新的多种子、独立test/OOD或Avatar生成结果。'''),
('next', '先补齐可预测的基础变化，再检验耦合的额外增益', '''1. 冻结observer、状态形成和起点，做纯向量目标对照；保持同一数据预算，并单独记录标签与向量梯度。把保持、收缩、线性当前状态、线性历史与动力学同时列为门槛。
2. 校准基础预测中心与不确定性读出，使模型能够表达经过校准的简单回归，再学习其残差。持续记忆仍按真实秒数演化，预测读出留在同一affect坐标；不要把均值收缩直接塞入持续状态或将训练均值强行解释为中性情感。
3. 每个有效训练批次明确分配数据域配额，按有效目标归一化分域损失；保持原始分布指标和宏平均指标并列。
4. 保留持续双人反馈，检验它在中心校准后、相对强单人预测器的额外收益。只有正确partner历史稳定优于错配历史，才能进一步支持其输入依赖性；因果解释还需要更严格设计。
5. 同一阶段连续保留优化器状态，再单独比较恒定学习率与衰减调度，避免一次改动多个因素后无法归因。

本轮最需要回答的问题是：在相同起点与纯向量目标下，校准后的连续动力学能否复现简单回归，并稳定利用双方信息预测其残差。完整实验之前，不能承诺以上修改一定超过均值收缩。''')]

names = {'learned':'当前动力学','hold':'保持状态','shrink':'均值收缩',
         'learned_alpha':'训练拟合幅度校准','learned_center_shift':'训练拟合中心校准',
         'learned_train_bias':'训练拟合向量偏置','without_future_partner':'关闭预测耦合'}
comparison = [dict(method=label, key=method, mse=val[method]['macro_mse'],
    mse_x1000=1000*val[method]['macro_mse'], elements=val[method]['elements'],
    train_mse=train.get(method, {}).get('macro_mse'), step=9750,
    protocol='固定起点验证', fitted_on='仅训练缓存') for method,label in names.items()]
domain_rows = []
for method in ('learned','hold','shrink'):
    for domain, label in enumerate(('EmotionTalk','IEMOCAP','DualTalk')):
        rows = [r for r in val[method]['rows'] if r['domain']==domain]
        queries = sum(r['queries'] for r in rows)
        mse = sum(r['mse']*r['queries'] for r in rows)/queries
        domain_rows.append(dict(domain=label, method=names[method], mse=mse,
            mse_x1000=1000*mse, queries=queries, elements=128*queries,
            train_origin_count=data['protocols']['train']['origins_per_domain'][domain]))
blocks = [dict(id='title', type='markdown', body='# '+title)]
for key,heading,body in sections:
    blocks.append(dict(id=key, type='markdown', body='## '+heading+'\n\n'+body))
    if key=='comparison':
        blocks.append(dict(id='comparison_chart', type='chart', chartId='method_comparison'))
    if key=='sampling':
        blocks.append(dict(id='domain_chart', type='chart', chartId='domain_comparison'))

source = dict(id=source_id, label='9750步最佳权重只读GPU诊断',
    path=str(OUT/'shrink_gap_diagnostics.json'),
    query=dict(engine='Python/PyTorch', language='python', executed_at=data['completed_at'],
        description='scripts/diagnose_shrink_gap.py读取最佳权重及最后一块训练起点，训练集拟合校准参数后评测固定验证起点。',
        filters=['best step 9750','24 fixed validation dialogues','future queries 1/2/4/8/16/32 seconds','valid target coordinates only'],
        metric_definitions=['macro_mse = mean_h(mean_available_domain(SSE / valid coordinates))',
                            'domain mse = sum_h(SSE) / sum_h(valid coordinates)',
                            'chart mse_x1000 = raw mse * 1000']))
# Actual executed SQLite extraction keeps a runnable, file-backed query for
# the report widget. The GPU audit remains the primary source of measurements.
chart_sql = '''WITH methods AS (
  SELECT key AS method_key, value AS payload
  FROM json_each(:audit_json, '$.protocols.fixed_val.methods')
), domain_horizon AS (
  SELECT method_key,
         json_extract(r.value, '$.seconds') AS seconds,
         json_extract(r.value, '$.domain') AS domain,
         json_extract(r.value, '$.mse') AS mse,
         json_extract(r.value, '$.queries') AS queries
  FROM methods, json_each(methods.payload, '$.rows') r
), horizon_macro AS (
  SELECT method_key, seconds, avg(mse) AS mse
  FROM domain_horizon GROUP BY method_key, seconds
)
SELECT method_key, avg(mse) AS macro_mse
FROM horizon_macro GROUP BY method_key'''
domain_sql = '''WITH methods AS (
  SELECT key AS method_key, value AS payload
  FROM json_each(:audit_json, '$.protocols.fixed_val.methods')
), domain_horizon AS (
  SELECT method_key, json_extract(r.value, '$.domain') AS domain,
         json_extract(r.value, '$.mse') AS mse,
         json_extract(r.value, '$.queries') AS queries
  FROM methods, json_each(methods.payload, '$.rows') r
)
SELECT method_key, domain, sum(mse * queries) / sum(queries) AS mse,
       sum(queries) AS queries
FROM domain_horizon GROUP BY method_key, domain'''
connection = sqlite3.connect(':memory:')
bindings = {'audit_json': json.dumps(data)}
sql_overall = dict(connection.execute(chart_sql, bindings).fetchall())
sql_domain = {(m,d):(mse,n) for m,d,mse,n in connection.execute(domain_sql, bindings)}
for row in comparison:
    assert abs(row['mse']-sql_overall[row['key']]) < 1e-12
    row['mse'] = sql_overall[row['key']]
    row['mse_x1000'] = 1000*row['mse']
for row in domain_rows:
    key = next(k for k,v in names.items() if v==row['method'])
    domain = ('EmotionTalk','IEMOCAP','DualTalk').index(row['domain'])
    mse,n = sql_domain[key,domain]
    assert abs(row['mse']-mse) < 1e-12 and row['queries']==n
    row['mse'], row['mse_x1000'] = mse, 1000*mse
connection.close()
chart_source = {**source, 'query': {**source['query'], 'engine':'SQLite JSON1',
    'language':'sql', 'sql':chart_sql,
    'description':'参数audit_json为shrink_gap_diagnostics.json完整内容。原始测量由只读GPU诊断产生；SQL从分域分时距误差复算宏平均。'}}
domain_source = {**chart_source, 'query': {**chart_source['query'], 'sql':domain_sql}}
artifact = dict(surface='report', manifest=dict(version=1, surface='report', title=title,
    generatedAt=data['completed_at'], blocks=blocks, sources=[source], charts=[
        dict(id='method_comparison',title='固定起点验证：预测器MSE（数值×1000）',
             dataset='comparison',type='bar',source=chart_source,
             encodings=dict(x=dict(field='method',type='nominal'),
                            y=dict(field='mse_x1000',type='quantitative')),
             options=dict(orientation='horizontal')),
        dict(id='domain_comparison',title='各数据集的预测MSE（数值×1000）',
             dataset='domains',type='bar',source=domain_source,
             encodings=dict(x=dict(field='domain',type='nominal'),
                            y=dict(field='mse_x1000',type='quantitative'),
                            color=dict(field='method',type='nominal')),
             options=dict(grouping='grouped'))]),
    snapshot=dict(version=1, status='ready',generatedAt=data['completed_at'],
                  datasets=dict(comparison=comparison, domains=domain_rows)),
    package_info=dict(audience='technical',readonly=True, source_script='scripts/diagnose_shrink_gap.py'))
(OUT/'shrink_gap_report.artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
# Companion source narrative, not a separate HTML/app rendering implementation.
(OUT/'shrink_gap_analysis.md').write_text('# '+title+'\n\n'+'\n\n'.join(
    '## '+heading+'\n\n'+body for _,heading,body in sections),encoding='utf-8')
(OUT/'shrink_gap_report_notes.json').write_text(json.dumps(dict(
    audience='technical',delivery='mcp-app',primary_metric='fixed-origin macro MSE',
    chart_contract=[dict(id='method_comparison',question='校准和耦合消融改变多少MSE',family='bar',
                        rows=len(comparison),y='MSE×1000',palette='neutral with explicit method labels'),
                    dict(id='domain_comparison',question='差距是否集中在某个数据集',family='grouped bar',
                         rows=len(domain_rows),legend='method',y='MSE×1000')],
    limitations=['validation previously used for checkpoint selection','local gradient diagnostic: 6 batches',
                 'postprocessing benefit is not proof of an internal causal mechanism'],
    source_sha256=__import__('hashlib').sha256((OUT/'shrink_gap_diagnostics.json').read_bytes()).hexdigest()),
    ensure_ascii=False,indent=2),encoding='utf-8')
print(str(OUT/'shrink_gap_report.artifact.json'))
