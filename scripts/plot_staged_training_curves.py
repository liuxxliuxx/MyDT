"""Export staged validation histories and read-only decay audits as research figures."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib import font_manager
import numpy as np


METHODS = ['learned', 'hold', 'pure_decay', 'shrink', 'mean']
LABELS = dict(learned='学习动力学', hold='保持状态', pure_decay='纯衰减（可恢复权重）',
              shrink='训练均值收缩', mean='直接预测训练均值')
STYLES = dict(learned=dict(color='#2463A6', linestyle='-', linewidth=2.3),
              hold=dict(color='#4C5057', linestyle='--', linewidth=1.5),
              pure_decay=dict(color='#735329', linestyle='None', marker='^', markersize=6),
              shrink=dict(color='#D17B26', linestyle='-', linewidth=1.7),
              mean=dict(color='#8F939A', linestyle=':', linewidth=1.8))
PROTOCOLS = dict(fixed='固定起点：隔离传播能力', live='部署起点：按当前权重重新回放',
                 gap='缺失输入：每32秒遮挡末尾8秒')
PHASES = dict(initial='初始', calibration='校准', fixed='固定起点', joint='联合', readapt='重新适配')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def summarize_sums(sums):
    values = np.asarray(sums, dtype=np.float64)
    error, count = values[..., 0], values[..., 1]
    per_h = [np.mean(error[:, h][count[:, h] > 0]/count[:, h][count[:, h] > 0])
             for h in range(error.shape[1]) if np.any(count[:, h] > 0)]
    return float(error.sum()/count.sum()), float(np.mean(per_h)), int(count.sum())


def gather(directory):
    snapshot = read_json(directory/'source_snapshot.json')
    data, audits = {}, {}
    for stage in ['diagnostic', 'formal']:
        path = directory/(stage+'_decay_audit.json')
        audit = read_json(path) if path.exists() else {}
        audits[stage] = audit
        events = snapshot['runs'][stage]['events'] + audit.get('events', [])
        vals = {}
        training = {}
        for event in events:
            if event.get('event') == 'validation':
                m = event['metrics']
                if m['step'] in vals and vals[m['step']]['candidate_hash'] != m['candidate_hash']:
                    raise AssertionError('Two candidate weights share a run step')
                vals[m['step']] = m
            elif event.get('event') == 'train':
                training[event['step']] = event
        pure = {}
        for item in audit.get('evaluation', []):
            for step in item['steps']:
                if step not in vals:
                    continue
                if vals[step]['candidate_hash'] != item['candidate_hash']:
                    raise AssertionError('Decay evaluation belongs to another weight snapshot')
                pure[step] = item
        data[stage] = dict(values=[vals[k] for k in sorted(vals)], pure=pure,
            train=[training[k] for k in sorted(training)], config=snapshot['runs'][stage]['config'],
            captured_at=audit.get('completed_at', snapshot['time']))
    return snapshot, data, audits


def export_tables(directory, data):
    overall, detail, semantic = [], [], []
    for stage, dataset in data.items():
        horizons = dataset['config']['train']['forecast_seconds']
        names = dataset['config']['data']['source_names']
        for m in dataset['values']:
            common = dict(run=stage, step=m['step'], phase=m['phase'], block=m['block'],
                          candidate_hash=m['candidate_hash'])
            for protocol in PROTOCOLS:
                methods = dict(m[protocol]['methods'])
                sums = dict(zip(methods, m[protocol]['sums']))
                if m['step'] in dataset['pure']:
                    decay = dataset['pure'][m['step']]['protocols'][protocol]['pure_decay']
                    methods['pure_decay'], sums['pure_decay'] = decay, decay['sums']
                for method, values in methods.items():
                    micro, macro, elements = summarize_sums(sums[method])
                    if not np.isclose(macro, values['macro_mse'], rtol=1e-12, atol=1e-14):
                        raise AssertionError('Macro aggregation differs from source log')
                    if not np.isclose(micro, values['mse'], rtol=1e-12, atol=1e-14) or elements != values['elements']:
                        raise AssertionError('Weighted aggregation differs from source log')
                    overall.append(dict(**common, protocol=protocol, method=method,
                        macro_mse=macro, weighted_mse=micro, rmse_weighted=micro**.5,
                        squared_error_sum=float(np.asarray(sums[method])[..., 0].sum()), valid_elements=elements,
                        gain_over_hold=1-macro/m[protocol]['methods']['hold']['macro_mse'],
                        evidence='retained_checkpoint_replay' if method == 'pure_decay' else 'historical_validation_log',
                        origin_producer=m[protocol]['producer'], train_bank_producer=m['bank_producer']))
                    for d, domain in enumerate(names):
                        for h, seconds in enumerate(horizons):
                            error, count = sums[method][d][h]
                            detail.append(dict(**common, protocol=protocol, method=method, domain=domain,
                                horizon_seconds=seconds, squared_error_sum=error, valid_elements=int(count),
                                mse=error/count if count else None))
            sem = dict(m['live'].get('semantic', {}))
            sem.pop('protocol', None)
            semantic.append(dict(**common, **sem,
                current_posterior_mse=m['live']['current']['posterior_sse']/m['live']['current']['elements'],
                current_prior_mse=m['live']['current']['input_conditioned_prior_sse']/m['live']['current']['elements'],
                live_origin_drift_rms=m['live_origin_drift_from_initial_rms']))
    for name, rows in [('metrics_overall.csv', overall), ('metrics_by_domain_horizon.csv', detail), ('semantic_metrics.csv', semantic)]:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (directory/name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    return overall, detail, semantic


def figure_header(fig, title, subtitle):
    fig.suptitle(title, x=.055, y=.98, ha='left', fontsize=17, fontweight='bold', color='#20242A')
    fig.text(.055, .90, subtitle, ha='left', fontsize=9.5, color='#4C5057')


def tidy(ax):
    ax.grid(axis='y', color='#E1E3E6', linewidth=.65)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#ADB1B8')
    ax.tick_params(labelsize=9)


def stage_regions(ax, dataset, labels=True):
    maximum = dataset['values'][-1]['step']
    s = dataset['config']['staged']
    calibration, fixed = s['calibration_steps'], s['fixed_steps']
    limits = [(0, calibration, '校准'), (calibration, calibration+fixed, '固定起点'),
              (calibration+fixed, maximum, '联合 / 重新适配')]
    for a, b, label in limits:
        b = min(b, maximum)
        if b <= a:
            continue
        if labels:
            ax.text((a+b)/2, 1.018, label, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=8, color='#62666D')
        if a > 0:
            ax.axvline(a, color='#A8ACB3', linewidth=.8, linestyle=':')
    cursor = calibration+fixed
    for _ in range(s['joint_rounds']):
        joint_end = cursor+s['joint_steps_per_round']
        if cursor < maximum:
            ax.axvspan(cursor, min(joint_end, maximum), color='#E3E5E8', alpha=.45, zorder=0)
        cursor = joint_end+s['readapt_steps_per_round']
    ax.set_xlim(-maximum*.015, maximum*1.018)
    ax.set_xlabel('训练步数', fontsize=10)


def best_point(dataset):
    eligible = [m for m in dataset['values'] if m['phase'] in ('fixed', 'readapt') and m.get('acceptance', {}).get('passed')]
    return min(eligible, key=lambda m: m['acceptance']['score']) if eligible else None


def curve_values(dataset, protocol, method, metric='macro_mse'):
    xs, ys = [], []
    for m in dataset['values']:
        if method == 'pure_decay':
            if m['step'] not in dataset['pure']:
                continue
            value = dataset['pure'][m['step']]['protocols'][protocol]['pure_decay'][metric]
        else:
            value = m[protocol]['methods'][method][metric]
        xs.append(m['step']); ys.append(value)
    return xs, ys


def draw_overview(dataset, stage, gain=False):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.7), sharey=True)
    title = ('正式训练' if stage == 'formal' else '1000步诊断') + ('：相对保持状态的预测增益' if gain else '：各预测器验证 MSE')
    figure_header(fig, title, '24个验证对话｜预测1、2、4、8、16、32秒｜每时距内按数据域等权，再按时距等权；没有平滑曲线')
    handles = []
    best = best_point(dataset)
    for ax, protocol in zip(axes, PROTOCOLS):
        for method in METHODS:
            xs, ys = curve_values(dataset, protocol, method)
            if gain:
                holds = {m['step']: m[protocol]['methods']['hold']['macro_mse'] for m in dataset['values']}
                ys = [100*(1-y/holds[x]) for x, y in zip(xs, ys)]
            else:
                ys = np.asarray(ys)*1000
            line, = ax.plot(xs, ys, label=LABELS[method], **STYLES[method])
            if protocol == 'fixed':
                handles.append(line)
        if best:
            point = best[protocol]['methods']['learned']['macro_mse']
            y = 100*(1-point/best[protocol]['methods']['hold']['macro_mse']) if gain else point*1000
            ax.plot(best['step'], y, marker='*', color='#2463A6', markeredgecolor='white', markersize=12, zorder=7)
        ax.set_title(PROTOCOLS[protocol], fontsize=11, pad=28)
        stage_regions(ax, dataset)
        tidy(ax)
        if gain:
            ax.axhline(0, color='#30343A', linewidth=.8)
    axes[0].set_ylabel('相对保持的 MSE 降低（%），越高越好' if gain else r'情感向量宏平均 MSE（$\times10^{-3}$），越低越好', fontsize=10)
    fig.legend(handles, [LABELS[k] for k in METHODS], loc='upper center', bbox_to_anchor=(.5, .885), ncol=5, frameon=False, fontsize=9)
    cutoff = dataset['captured_at'][:19].replace('T', ' ')
    fig.text(.055, .075, f'记录截至 {cutoff}；已完成验证至 {dataset["values"][-1]["step"]} 步。星号：按训练验收规则选出的 best（{best["step"] if best else "无"}步）。', fontsize=9)
    fig.text(.055, .04, '三角形只表示可恢复权重的纯衰减实测值，缺失权重不补点。灰色带：联合阶段。均值 / 收缩参数随训练对话批次更新。', fontsize=9, color='#4C5057')
    fig.subplots_adjust(left=.065, right=.985, bottom=.20, top=.73, wspace=.14)
    return fig


def draw_horizons(dataset):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9), sharex=True)
    horizons = dataset['config']['train']['forecast_seconds']
    figure_header(fig, '正式训练：固定起点在各预测时距的 MSE', r'相同初始起点与目标；每个时距对有有效样本的数据域等权｜纵轴 $\times10^{-3}$，越低越好')
    for ax, h, seconds in zip(axes.flat, range(len(horizons)), horizons):
        for method in METHODS:
            xs, ys = [], []
            for m in dataset['values']:
                if method == 'pure_decay':
                    if m['step'] not in dataset['pure']:
                        continue
                    sums = dataset['pure'][m['step']]['protocols']['fixed']['pure_decay']['sums']
                else:
                    sums = m['fixed']['sums'][list(m['fixed']['methods']).index(method)]
                a = np.asarray(sums)[:, h]
                mask = a[:, 1] > 0
                xs.append(m['step']); ys.append(np.mean(a[mask, 0]/a[mask, 1])*1000)
            ax.plot(xs, ys, **STYLES[method])
        ax.set_title(f'预测 {seconds} 秒后', fontsize=11)
        stage_regions(ax, dataset, labels=False); tidy(ax)
        ax.set_ylabel(r'MSE $\times10^{-3}$')
    fig.legend([Line2D([], [], **STYLES[k]) for k in METHODS], [LABELS[k] for k in METHODS],
               loc='upper center', bbox_to_anchor=(.5, .905), ncol=5, frameon=False, fontsize=9)
    fig.text(.055, .027, '32秒时距的 DualTalk 有效元素数为0，因此该时距宏平均只包含另外两个数据域。', fontsize=9, color='#4C5057')
    fig.subplots_adjust(left=.065, right=.985, bottom=.10, top=.82, hspace=.34, wspace=.23)
    return fig


def draw_domains(dataset):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.5))
    names = dataset['config']['data']['source_names']
    figure_header(fig, '正式训练：固定起点按数据集拆分', r'每个数据集按误差总和 / 有效元素数汇总全部时距｜纵轴 $\times10^{-3}$，越低越好；三个面板量程不同')
    for ax, d, name in zip(axes, range(3), names):
        for method in METHODS:
            xs, ys = [], []
            for m in dataset['values']:
                if method == 'pure_decay':
                    if m['step'] not in dataset['pure']:
                        continue
                    sums = dataset['pure'][m['step']]['protocols']['fixed']['pure_decay']['sums']
                else:
                    sums = m['fixed']['sums'][list(m['fixed']['methods']).index(method)]
                a = np.asarray(sums)[d]
                xs.append(m['step']); ys.append(a[:, 0].sum()/a[:, 1].sum()*1000)
            ax.plot(xs, ys, **STYLES[method])
        ax.set_title(name, fontsize=12, pad=28)
        stage_regions(ax, dataset); tidy(ax)
        ax.set_ylabel(r'MSE $\times10^{-3}$')
    fig.legend([Line2D([], [], **STYLES[k]) for k in METHODS], [LABELS[k] for k in METHODS],
               loc='upper center', bbox_to_anchor=(.5, .885), ncol=5, frameon=False, fontsize=9)
    fig.text(.055, .065, '这里只比较相同的固定起点，避免把起点变化计入传播改善；纯衰减只显示可恢复权重的实测点。', fontsize=9, color='#4C5057')
    fig.subplots_adjust(left=.065, right=.985, bottom=.19, top=.73, wspace=.25)
    return fig


def draw_semantic(dataset):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    specs = [('emotion_accuracy', '真实情感标签准确率（%）↑', 100), ('emotion_ce', '情感分类交叉熵 ↓', 1),
             ('vad_mse', '真实 VAD 标签 MSE ↓', 1), ('intensity_mse', '真实强度标签 MSE ↓', 1)]
    figure_header(fig, '正式训练：动力学预测的独立标签指标', '按真实话语结束时刻评测，使用冻结解码器；这些曲线只属于学习动力学，未为其他预测器补造历史标签指标')
    xs = [m['step'] for m in dataset['values']]
    for ax, (key, label, scale) in zip(axes.flat, specs):
        ys = [m['live']['semantic'][key]*scale for m in dataset['values']]
        ax.plot(xs, ys, color='#2463A6', linewidth=1.9)
        ax.axhline(ys[0], color='#8F939A', linestyle='--', linewidth=1, label='正式训练初始值')
        ax.set_title(label, fontsize=11)
        stage_regions(ax, dataset, labels=False); tidy(ax)
        ax.legend(frameon=False, fontsize=8)
    sem = dataset['values'][-1]['live']['semantic']
    fig.text(.055, .035, f'有效分类查询 {int(sem["emotion_query_count"])}；有效 VAD 坐标 {int(sem["vad_coordinate_count"])}；有效强度查询 {int(sem["intensity_query_count"])}。同一标签可能对应多个起点/时距查询。', fontsize=9, color='#4C5057')
    fig.subplots_adjust(left=.075, right=.975, bottom=.12, top=.84, hspace=.4, wspace=.25)
    return fig


def write_report(directory, data, audits):
    formal, diagnostic = data['formal'], data['diagnostic']
    best, latest = best_point(formal), formal['values'][-1]
    lines = [
        '# 本轮动力学训练曲线与预测器对比', '',
        f'正式训练记录截至 **{formal["captured_at"][:19].replace("T", " ")}**，共{len(formal["values"])}个验证点，覆盖0—{latest["step"]}步。',
        f'1000步诊断阶段共{len(diagnostic["values"])}个验证点，单独展示。图表使用所有已记录的验证结果，没有平滑、挑选有利区间或填补缺失评测值。', '',
        f'在这份快照中，训练选择规则选出的最佳权重是 **{best["step"]}步**。最新{latest["step"]}步尚未超过该综合选择分数。均值收缩仍优于学习动力学。', '',
        f'![正式训练 MSE]({(directory.resolve()/"formal_mse.png").as_posix()})', '',
        '## 同一起点下的预测器比较', '',
        f'下表使用{best["step"]}步权重。数值是情感向量宏平均 MSE，越低越好；没有混入 FLAME 生成误差。', '',
        '| 预测器 | 固定起点 | 当前权重重新形成起点 | 周期性缺失输入 |',
        '|---|---:|---:|---:|',
    ]
    for method in METHODS:
        values = []
        for protocol in PROTOCOLS:
            if method == 'pure_decay':
                record = formal['pure'].get(best['step'])
                value = record['protocols'][protocol]['pure_decay']['macro_mse'] if record else None
            else:
                value = best[protocol]['methods'][method]['macro_mse']
            values.append(f'{value:.9f}' if value is not None else '权重已覆盖，未补测')
        lines.append('| '+LABELS[method]+' | '+' | '.join(values)+' |')
    lines += ['', '## 曲线反映的变化', '']
    for protocol in PROTOCOLS:
        initial = formal['values'][0][protocol]['methods']['learned']['macro_mse']
        value = best[protocol]['methods']['learned']['macro_mse']
        shrink = best[protocol]['methods']['shrink']['macro_mse']
        hold = best[protocol]['methods']['hold']['macro_mse']
        lines.append(f'- {PROTOCOLS[protocol]}：best相对正式初始权重的MSE降低{(1-value/initial)*100:.2f}%，比保持降低{(1-value/hold)*100:.2f}%，但比均值收缩高{(value/shrink-1)*100:.2f}%。')
    lines += ['', f'按数据集拆分的固定起点结果（{best["step"]}步，误差总和/有效元素数）：', '',
              '| 数据集 | 学习动力学 MSE | 保持 MSE | 动力学相对保持的误差变化 |',
              '|---|---:|---:|---:|']
    for d, domain in enumerate(formal['config']['data']['source_names']):
        a = np.asarray(best['fixed']['sums'][0][d])
        b = np.asarray(best['fixed']['sums'][1][d])
        learned, hold = a[:, 0].sum()/a[:, 1].sum(), b[:, 0].sum()/b[:, 1].sum()
        lines.append(f'| {domain} | {learned:.9f} | {hold:.9f} | {(learned/hold-1)*100:+.2f}% |')
    lines += ['', 'DualTalk上仍然差于保持，三数据集的总体平均改善不能代替DualTalk单域上的有效性。', '']
    lines += [
        '- 约3500—4000步，重新形成起点和缺失输入的误差出现明显尖峰；固定起点没有对应幅度的尖峰。这支持进一步检查状态形成和部署起点分布，单凭曲线不能判定为过拟合。',
        '- 宏平均MSE、分类准确率、VAD误差和强度误差没有一致单调改善。独立标签指标见第5张图，不能用向量MSE的改善代替情感语义改善。',
        '- `best`要求固定起点改善、优于保持以及部署起点安全检查；选择规则没有要求超过均值收缩，也没有要求全部独立情感指标改善。', '',
        '## 指标和比较协议', '',
        '- 每次验证使用同样的24个对话，EmotionTalk、IEMOCAP、DualTalk各8个。它们属于验证集；本报告没有新增独立test/OOD或多种子实验。',
        '- 预测时距为1、2、4、8、16、32秒。每个时距内先对各有效数据域的MSE等权平均，再对时距等权平均。CSV同时保存误差总和/有效元素数得到的加权MSE、RMSE及分域分时距结果。',
        '- 每个协议的向量误差包含750464个有效坐标。32秒时距没有DualTalk有效目标，因此这个时距只平均另两个域；缺失数据没有填成零误差。',
        '- 固定起点始终使用正式训练初始化权重形成的相同状态。部署起点每次由对应权重重新顺序回放完整对话。缺失协议在回放中每32秒遮挡末尾8秒输入；指标统计整个回放的有效预测起点，并非仅统计缺失片段。',
        '- 所有未来预测都不读取未来观测、事件或行为。训练均值和收缩系数仅由当前训练起点库拟合；训练库随训练块轮换，所以它们的曲线会小幅变化。',
        '- 保持：预测值等于当前状态。直接均值：预测值等于训练教师向量均值。均值收缩：`mu + a_h*(z_t-mu)`，每个时距一个0—1系数。',
        '- 纯衰减调用对应权重的 `core.decay_only`：`baseline + fast*exp(-h/tau_fast) + slow*exp(-h/tau_slow)`。保留该权重的快慢时间常数，关闭自主混合和双人反馈；这是传播消融，没有另训一个模型。',
        '- 分类/VAD/强度曲线来自冻结解码器和真实话语结束标签，当前历史日志只记录了学习动力学的这些指标。有效分类查询646，VAD坐标2490，强度查询339；同一标签可对应不同起点/时距，因此不能把它们当作同等数量的独立对话。', '',
        '## 权重可恢复范围与补测', '',
        '训练程序反复覆盖 `last.pt` 和 `best.pt`，没有为每个验证步保存完整文件。完整历史曲线来自当时的验证日志。不能宣称把已经覆盖的每个历史权重重新评测了一遍。', '',
    ]
    max_delta = 0.
    for stage, dataset in data.items():
        steps = sorted(dataset['pure'])
        lines.append(f'- {stage}可恢复并补测纯衰减的步数：'+', '.join(map(str, steps))+'。包括完整checkpoint以及其中保存的初始/起点生产模型。')
        for item in audits[stage].get('evaluation', []):
            max_delta = max(max_delta, *(v['hold_log_abs_difference'] for v in item['protocols'].values()))
    lines += [
        '- 图中的纯衰减只画实测三角形，历史缺失点不补齐、不连成假想轨迹。',
        f'- 补测重新形成的起点经过保持MSE与原日志比对，最大绝对差{max_delta:.3g}，有效元素数一致；全部宏平均/加权MSE也从误差总和与计数重新计算核对。',
        '- 补测使用服务器1的GPU2、3，没有停止训练，没有修改服务器训练代码、权重、日志或缓存。诊断与正式训练使用不同的固定参考起点，故不把两者拼成一条连续训练曲线。', '',
        '## 文件', '',
        '- `training_comparison.pdf`：6页完整图表。',
        '- `formal_mse.png/svg`、`formal_gain.png/svg`：正式训练总览与相对保持增益。',
        '- `formal_horizons.png/svg`、`formal_domains.png/svg`：按预测时距、数据集拆分。',
        '- `formal_semantic.png/svg`：独立情感标签指标。',
        '- `diagnostic_mse.png/svg`：1000步诊断阶段。',
        '- `metrics_overall.csv`、`metrics_by_domain_horizon.csv`、`semantic_metrics.csv`：全部原始数值与分母。',
        '- `source_snapshot.json`、`formal_decay_audit.json`、`diagnostic_decay_audit.json`：日志快照、权重来源、补测结果及日志摘要哈希。',
        '- 复现绘图：`python scripts/plot_staged_training_curves.py --directory runs/v3_2_1_staged_20260910_gpu23/training_curves`。', '',
    ]
    (directory/'comparison_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args(); directory = args.directory
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams.update({'axes.unicode_minus': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    snapshot, data, audits = gather(directory)
    overall, detail, semantic = export_tables(directory, data)
    write_report(directory, data, audits)
    figures = [('formal_mse', draw_overview(data['formal'], 'formal')),
               ('formal_gain', draw_overview(data['formal'], 'formal', gain=True)),
               ('formal_horizons', draw_horizons(data['formal'])),
               ('formal_domains', draw_domains(data['formal'])),
               ('formal_semantic', draw_semantic(data['formal'])),
               ('diagnostic_mse', draw_overview(data['diagnostic'], 'diagnostic'))]
    with PdfPages(directory/'training_comparison.pdf') as pdf:
        for name, fig in figures:
            fig.savefig(directory/(name+'.png'), dpi=180)
            fig.savefig(directory/(name+'.svg'))
            pdf.savefig(fig)
            plt.close(fig)
        assert pdf.get_pagecount() == len(figures)
    summary = {}
    for stage, dataset in data.items():
        best, latest = best_point(dataset), dataset['values'][-1]
        summary[stage] = dict(validation_points=len(dataset['values']), steps=[m['step'] for m in dataset['values']],
            pure_decay_steps=sorted(dataset['pure']), best_step=best['step'] if best else None,
            latest_step=latest['step'], captured_at=dataset['captured_at'],
            best=best, latest=latest, initial=dataset['values'][0],
            pure_decay={str(step): v['protocols'] for step, v in dataset['pure'].items()})
    (directory/'comparison_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    contract = dict(question='How do dynamics and simple predictors compare throughout staged training?',
        renderer='Matplotlib scientific standalone PNG/SVG and PDF', family='line with sparse measured checkpoint markers',
        palette_policy='two non-neutral roots (blue/orange) plus neutrals', smoothing=False,
        missing_checkpoint_policy='No interpolation or invented decay metrics; actual markers only',
        source_root=snapshot['root'], metric='macro domain-within-horizon, then mean over horizons',
        rows=len(overall), detail_rows=len(detail), semantic_rows=len(semantic),
        source_hashes={stage: audits[stage].get('log_sha256', snapshot['runs'][stage]['log_sha256']) for stage in data})
    (directory/'chart_contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    print(json.dumps({stage:{k:v for k,v in value.items() if k in ['validation_points','pure_decay_steps','best_step','latest_step','captured_at']} for stage,value in summary.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
