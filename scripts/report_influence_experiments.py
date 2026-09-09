"""Create auditable tables, source-cluster intervals and standalone research plots."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from emotion_ssm.evaluate_influence import dump, weighted, paired_summary


VARIANTS = ("none", "affect", "self", "dyadic")
COLORS = {"none": "#929292", "affect": "#555555", "self": "#2364AA", "dyadic": "#BA7C16"}


def scalar_ci(rows, getter, repetitions=2000):
    groups = {}
    for r in rows:
        value = getter(r)
        if value is not None and np.isfinite(value):
            groups.setdefault(r["source"], []).append(float(value))
    if not groups:
        return {"mean": None, "source_cluster_95ci": None, "count": 0}
    groups = list(groups.values())
    rng = np.random.default_rng(6666)
    samples = [np.mean([v for i in rng.integers(len(groups), size=len(groups)) for v in groups[i]])
               for _ in range(repetitions)]
    return {"mean": float(np.mean([v for g in groups for v in g])),
            "source_cluster_95ci": np.quantile(samples, [.025, .975]).tolist(),
            "count": sum(map(len, groups)), "source_clusters": len(groups)}


def analyze(root):
    values = {v: json.loads((root / (v+".json")).read_text(encoding="utf-8")) for v in VARIANTS}
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    result = {"manifest_digest": manifest["digest"], "within_checkpoint": {}, "event_trajectory_vs_none": {},
              "event_pulse_by_role": {}, "event_retention": {}, "comparison_scale": {}, "speaking_counts": {}}
    for variant, value in values.items():
        rows = value["records"]
        result["within_checkpoint"][variant] = {}
        for experiment, mode in (("history", "reset"), ("event", "removed"),
                                 ("partner", "shuffled"), ("partner", "coupling_off")):
            actual = {r["name"]: r for r in rows if r[experiment] is not None}
            altered = {name: {**r, experiment: {**r[experiment], "natural": r[experiment][mode]}}
                       for name, r in actual.items()}
            # Here the labels mean full and ablation, never independently retrained controls.
            interval = paired_summary(actual, altered, lambda r: r[experiment]["natural"], "mean_expression_mse")
            interval["full"] = interval.pop("none")
            interval["ablated"] = interval.pop("condition")
            interval["direction"] = "ablated minus full; positive supports retaining this mechanism"
            result["within_checkpoint"][variant][experiment+"_"+mode] = interval
        event_rows = [r for r in rows if r["event"] is not None]
        baseline = {r["name"]: r for r in values["none"]["records"] if r["event"] is not None}
        result["event_trajectory_vs_none"][variant] = {
            field: scalar_ci(event_rows, lambda r, f=field: r["event"]["trajectory"][f]-baseline[r["name"]]["event"]["trajectory"][f])
            for field in ("centered_trajectory_mse", "response_amplitude_mae", "peak_lag_error_seconds",
                          "late_response_error", "response_area_error")}
        result["event_pulse_by_role"][variant] = {}
        for origin in ("target", "partner"):
            selected = [r for r in event_rows if (r["name"].rsplit("_", 1)[1] == r["event"]["anchor"]["role"]) == (origin == "target")]
            result["event_pulse_by_role"][variant][origin] = {"count": len(selected),
                **{field: np.mean([r["event"][field]["per_second_rms"] for r in selected], 0).tolist()
                    if selected else None for field in ("single_effect", "repeated_effect")}}
        # Ratios only when the single-event output change exceeds the declared 1e-6 floor.
        active = [r for r in event_rows if r["event"]["single_effect"]["per_second_rms"][0] > 1e-6]
        result["event_retention"][variant] = {"eligible": len(active), "total": len(event_rows),
            "threshold": 1e-6,
            "retention_at_4s": scalar_ci(active, lambda r: r["event"]["single_effect"]["per_second_rms"][4] / r["event"]["single_effect"]["per_second_rms"][0]),
            "retention_at_8s": scalar_ci(active, lambda r: r["event"]["single_effect"]["per_second_rms"][8] / r["event"]["single_effect"]["per_second_rms"][0]),
            "repeat_minus_single_at_4s": scalar_ci(event_rows, lambda r: r["event"]["repeated_effect"]["per_second_rms"][4] - r["event"]["single_effect"]["per_second_rms"][4])}
        hist = [r["history"] for r in rows if r["history"] is not None]
        scale = math.sqrt(weighted([r["natural"] for r in hist], "expression_mse"))
        result["comparison_scale"][variant] = {"natural_expression_rmse": scale,
            "history_effect_fraction_of_prediction_rmse": summary["mechanisms"]["history"][variant]["mean_output_rms"]/scale,
            "partner_effect_fraction_of_prediction_rmse": summary["mechanisms"]["partner"][variant]["shuffle_output_rms"]/scale}
        result["speaking_counts"][variant] = {k: summary["mechanisms"]["partner"][variant][k+"_valid_chunks"] for k in ("listening", "speaking")}
    dump(root/"extended_summary.json", result)
    return values, manifest, summary, result


def figures(root, summary, extended):
    os.environ.setdefault("MPLCONFIGDIR", str(root/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for v, marker in (("self", "o"), ("dyadic", "s")):
        axes[0].plot(np.arange(4, 12), summary["mechanisms"]["history"][v]["mean_per_second_rms"],
                     marker=marker, color=COLORS[v], label=v)
    axes[0].axhline(0, color=COLORS["none"], ls="--", lw=1, label="none / affect = 0")
    axes[0].set(title="A. History reset: output difference", xlabel="Seconds since reset", ylabel="Expression RMS difference")
    axes[0].legend(fontsize=8)
    for v, marker in (("self", "o"), ("dyadic", "s")):
        for field, linestyle in (("single_effect", "-"), ("repeated_effect", "--")):
            axes[1].plot(range(9), extended["event_pulse_by_role"][v]["target"][field],
                color=COLORS[v], marker=marker, ls=linestyle, label=v+(" once" if linestyle == "-" else " 3 pulses"))
    axes[1].set(title="B. Target event: output response", xlabel="Seconds after first event", ylabel="RMS difference from zero-event branch")
    axes[1].legend(fontsize=8)
    for second in (0, 2, 4):
        axes[1].axvline(second, color="#dddddd", lw=.8, zorder=0)
    x = np.arange(4)
    axes[2].bar(x-.17, [summary["mechanisms"]["partner"][v]["shuffle_output_rms"] for v in VARIANTS],
                width=.32, color="#2364AA", label="Partner state shuffled")
    axes[2].bar(x+.17, [summary["mechanisms"]["partner"][v]["coupling_output_rms"] for v in VARIANTS],
                width=.32, color="white", edgecolor="#2364AA", hatch="//", label="Coupling disabled")
    axes[2].set(xticks=x, xticklabels=VARIANTS, title="C. Partner: output difference", ylabel="Expression RMS difference")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax.grid(axis="y", color="#ededed", lw=.6)
        ax.set_axisbelow(True)
    fig.suptitle("State interventions in existing 1,000-step checkpoints", fontsize=15)
    history_count = summary["mechanisms"]["history"]["none"]["count"]
    target_event_count = extended["event_pulse_by_role"]["none"]["target"]["count"]
    event_count = summary["mechanisms"]["event"]["none"]["count"]
    fig.supxlabel(f"Test: {history_count} directed history/partner clips; {target_event_count} target-event clips. Raw AV and fast affect held fixed. Larger responses do not imply better emotion.", fontsize=9)
    fig.savefig(root/"mechanisms.png", dpi=170)
    fig.savefig(root/"mechanisms.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, cohort, title in zip(axes, ("history", "event"), (f"History / partner cohort ({history_count} clips)", f"Event cohort ({event_count} clips)")):
        for i, v in enumerate(("affect", "self", "dyadic")):
            r = summary["quality"][cohort][v]["mean_expression_mse"]
            value = r["relative_percent"]
            lo, hi = [100*t/r["none"] for t in r["source_cluster_95ci"]]
            ax.plot([lo, hi], [i, i], color=COLORS[v], lw=2)
            ax.plot(value, i, "o", color=COLORS[v], ms=7)
            ax.annotate(f"{value:+.3f}%", (value, i), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=9)
        ax.axvline(0, color="#777777", ls="--", lw=1)
        ax.set(yticks=range(3), yticklabels=["affect", "self", "dyadic"], ylim=(-.65, 2.65),
               xlabel="Mean expression MSE change versus none (%)", title=title)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#ededed", lw=.6)
    fig.suptitle("Natural-target reconstruction quality", fontsize=15)
    fig.supxlabel("Negative values indicate lower error. 95% paired source-cluster bootstrap intervals; one training seed. Metric is not a VA/emotion accuracy score.", fontsize=9)
    fig.savefig(root/"quality.png", dpi=170)
    fig.savefig(root/"quality.pdf")
    plt.close(fig)


def fmt(value, digits=6):
    return "缺失" if value is None else f"{value:.{digits}g}"


def markdown(root, values, manifest, summary, extended):
    # This narrative is reviewed for the frozen September 8 pilot, not a generic claim generator.
    if manifest["digest"] != "4d4a3ff577fc64920263199a5c2d736d82ee4a0a41acfeea65b3d2ba15a2d9c4":
        raise ValueError("Review the narrative for the new selection before exporting this pilot report")
    base_trajectory = summary["mechanisms"]["event"]["none"]["centered_trajectory_mse"]
    dyadic_trajectory = summary["mechanisms"]["event"]["dyadic"]["centered_trajectory_mse"]
    trajectory_ci = extended["event_trajectory_vs_none"]["dyadic"]["centered_trajectory_mse"]["source_cluster_95ci"]
    boundary = summary["quality"]["event"]["dyadic"]["boundary_velocity_mse"]
    lines = ["# 历史、事件响应和 partner 影响实验结果", "", "运行日期：2026-09-08。",
        "", "历史、事件和 partner 状态通路均会改变 dyadic 输出，但绝对影响较小。主要的一秒平均 expression MSE 对照区间均跨零；保留历史、保留单次事件和保留耦合的预测收益也未在本批片段得到稳定支持。",
        "", f"事件片段存在两个正向信号：dyadic 的去基线表情轨迹 MSE 从 {base_trajectory:.6f} 降到 {dyadic_trajectory:.6f}，下降 {100*(1-dyadic_trajectory/base_trajectory):.2f}%；模型减 none 的来源聚类 95% 区间为 [{trajectory_ci[0]:+.6g}, {trajectory_ci[1]:+.6g}]。跨块边界速度误差下降 {-boundary['relative_percent']:.2f}%。这些是预先定义的次要指标，来自单个训练种子、19 对事件片段，尚未进行多重比较修正，不足以推出整体情感质量优于 none。",
        "", "使用四组独立训练的 1000 步 checkpoint，种子 6666；每次输出 25 帧，最多读取当前一秒与此前三秒输入。比较对象为同一流式协议下的 `none`。本轮没有追加训练。",
        "", f"选样摘要：`{manifest['digest']}`。历史与 partner 为 32 对 / 64 个有方向片段；事件为 19 对 / 38 个有方向片段；合并后 100 个片段，覆盖 {len(set(r['source'] for r in manifest['rows']))} 个原始视频。",
        "", "事件候选来自固定英文情感词表及 CTC 对齐，未人工确认语境、否定或真实情感。测试集中 533 对片段只有 200 对含可用对齐文本，最终 19 对满足本实验窗口与置信度条件。",
        "", "## 1. 自然输入下与 none 的质量比较", "",
        "主要指标为每秒平均 50 维 expression 与真实 FLAME 的 MSE。历史与 partner 使用完全相同的评分窗口，只列一行；它们不是两份独立质量证据。", "",
        "| 片段组 | none | affect | self | dyadic |", "|---|---:|---:|---:|---:|"]
    for cohort, label in (("history", "历史 / partner"), ("event", "事件后 0–8 秒")):
        q = summary["quality"][cohort]
        lines.append(f"| {label} | {fmt(q['affect']['mean_expression_mse']['none'])} | " +
            " | ".join(f"{fmt(q[v]['mean_expression_mse']['condition'])} ({q[v]['mean_expression_mse']['relative_percent']:+.3f}%)" for v in ("affect", "self", "dyadic")) + " |")
    lines += ["", "括号为相对 none 的误差变化，负数表示降低。", "", f"![自然目标质量比较]({(root/'quality.png').resolve().as_posix()})", "",
        "| 片段组 | 模型 | MSE 差值：模型 − none | 配对来源 bootstrap 95% 区间 |", "|---|---|---:|---|"]
    for cohort in ("history", "event"):
        for v in ("affect", "self", "dyadic"):
            q = summary["quality"][cohort][v]["mean_expression_mse"]
            lines.append(f"| {cohort} | {v} | {q['delta']:+.8g} | [{q['source_cluster_95ci'][0]:+.8g}, {q['source_cluster_95ci'][1]:+.8g}] |")
    lines += ["", "区间只反映原始视频抽样差异，不包含训练种子的方差；没有进行多重比较校正。", "", "## 2. 历史状态是否起作用", "",
        "处理第九秒输入前重置状态；保持后续音视频、当前 affect、缓存文本不变；推进三秒后评分。以下误差差值为重置 − 正常，正值才支持保留真实历史。", "",
        "| 模型 | 输出 expression RMS 变化 | 正常 MSE | 重置 MSE | 重置 − 正常 | 95% 区间 |", "|---|---:|---:|---:|---:|---|"]
    for v in VARIANTS:
        h = summary["mechanisms"]["history"][v]
        r = extended["within_checkpoint"][v]["history_reset"]
        lines.append(f"| {v} | {fmt(h['mean_output_rms'])} | {fmt(r['full'])} | {fmt(r['ablated'])} | {r['delta']:+.8g} | {r['source_cluster_95ci']} |")
    lines += ["", "## 3. 事件后的持续、衰减与积累", "",
        "自然事件轨迹以各自事件前两秒的平均 expression 为基线，统计事件所在秒及后八秒。去基线误差会忽略恒定偏移，因此必须与上一节绝对 MSE 同时看。以下指标均为真实 FLAME 轨迹误差，越低越好。", "",
        "| 模型 | 去基线轨迹 MSE | 幅度 MAE | 峰值时间误差（秒） | 后四秒幅度误差 | 响应面积误差 |", "|---|---:|---:|---:|---:|---:|"]
    fields = ("centered_trajectory_mse", "response_amplitude_mae", "peak_lag_error_seconds", "late_response_error", "response_area_error")
    for v in VARIANTS:
        e = summary["mechanisms"]["event"][v]
        lines.append(f"| {v} | " + " | ".join(fmt(e[f]) for f in fields) + " |")
    lines += ["", "| 模型 | 去基线轨迹 MSE：模型 − none | 95% 来源聚类区间 |", "|---|---:|---|"]
    for v in ("affect", "self", "dyadic"):
        t = extended["event_trajectory_vs_none"][v]["centered_trajectory_mse"]
        lines.append(f"| {v} | {t['mean']:+.8g} | {t['source_cluster_95ci']} |")
    lines += ["", "受控分支保留同一自然后缀，在相对 0 秒注入一次事件，或在 0、2、4 秒重复注入。它没有独立反事实真实标签，下面只记录机制响应。", "",
        "| 模型 | 单次事件输出变化：0s | 4s | 8s | 三次事件输出变化：4s | 三次 − 单次：4s |", "|---|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        p = extended["event_pulse_by_role"][v]["target"]
        a, b = p["single_effect"], p["repeated_effect"]
        lines.append(f"| {v} | {fmt(a[0])} | {fmt(a[4])} | {fmt(a[8])} | {fmt(b[4])} | {b[4]-a[4]:+.6g} |")
    lines += ["", "此表仅含事件发起方为预测目标的 19 个片段，避免将自身事件与 partner 事件混合。全部 38 个片段和角色细分曲线存于 JSON。`none/affect` 不读取持续状态，事件状态干预的零输出响应是结构预期。", "",
        "## 4. partner 状态与耦合", "",
        "置换只改变 SSM 的 partner 观测，保持生成器原始 partner 音视频、双方当前 affect 和文本一致；关闭耦合还停止 relation 更新。该实验检验额外的情感状态通路，baseline 本身仍有原始 partner 输入。", "",
        "| 模型 | partner 置换输出 RMS | 关闭耦合输出 RMS | 正常 MSE | 置换 MSE | 关闭耦合 MSE |", "|---|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        p = summary["mechanisms"]["partner"][v]
        lines.append(f"| {v} | {fmt(p['shuffle_output_rms'])} | {fmt(p['coupling_output_rms'])} | {fmt(p['natural_mean_expression_mse'])} | {fmt(p['shuffled_mean_expression_mse'])} | {fmt(p['coupling_off_mean_expression_mse'])} |")
    lines += ["", "| dyadic 消融 | 消融 − 正常 MSE | 95% 来源聚类区间 |", "|---|---:|---|"]
    for label, key in (("删除锚点事件", "event_removed"), ("置换 partner", "partner_shuffled"), ("关闭耦合", "partner_coupling_off")):
        e = extended["within_checkpoint"]["dyadic"][key]
        lines.append(f"| {label} | {e['delta']:+.8g} | {e['source_cluster_95ci']} |")
    lines += ["", f"![三类机制响应]({(root/'mechanisms.png').resolve().as_posix()})", "", "## 5. 全部重建指标", "",
        "下表全部为模型相对 none 的误差变化百分比，负值更低。来源聚类区间完整保存在 summary.json。", "",
        "| 片段组 | 模型 | 逐帧 expression | jaw | neck | velocity | boundary velocity |", "|---|---|---:|---:|---:|---:|---:|"]
    for cohort in ("history", "event"):
        for v in ("affect", "self", "dyadic"):
            q = summary["quality"][cohort][v]
            lines.append(f"| {cohort} | {v} | " + " | ".join(f"{q[k+'_mse']['relative_percent']:+.3f}%"
                for k in ("expression", "jaw", "neck", "velocity", "boundary_velocity")) + " |")
    lines += ["", "事件片段中 self 和 dyadic 的边界速度误差分别降低约 0.77% 和 0.78%，未校正的区间均在零以下；历史片段中 dyadic 的 neck 误差上升约 0.36%，未校正区间在零以上。其余这些重建指标的区间跨零。幅度、峰值时机和后期持续幅度指标没有随去基线轨迹 MSE 一同改善。",
        "", "## 6. 数据与执行校验", "",
        "- 每个片段均比较正常重放与真实流式前向；只缓存完全冻结模型的 FiLM 前激活，不跨模型复用。",
        "- 所有实验按有效帧和元素数聚合；双方方向与同源片段在 bootstrap 中一起抽样。",
        "- 事件选样依赖输入文本，不依赖模型输出；历史与 partner 按固定来源抽样。没有因结果不好而更换片段。",
        "- 这些都是同一 checkpoint 的推理消融，不等价于将该结构重新训练一次。",
        "- FLAME expression 误差和响应幅度不能当成独立 VA/情感分类指标。较大响应不保证更恰当。",
        "", "| 模型 | 条件重放最大绝对误差 | 输出重放最大绝对误差 | 当前音量阈值判定的 listening 块数 |", "|---|---:|---:|---:|"]
    for v in VARIANTS:
        p = values[v]["parity"]
        lines.append(f"| {v} | {fmt(p['context_max_absolute_error'])} | {fmt(p['prediction_max_absolute_error'])} | {extended['speaking_counts'][v]['listening']} |")
    lines += ["", "音量阈值不是人工说话人轮次标签。若 listening 块数不足，不据此给出聆听场景结论。",
        "", "运行目录：`/home/s21_yhr/lzh/MyDualTalk/runs/v2_influence_20260908/`。日志为 `launcher.log` 及 `none.log / affect.log / self.log / dyadic.log`。",
        "", "本地同目录保存 manifest、四组逐片段 JSON、summary、extended_summary 和 PNG/PDF 图。更多逐帧及分部位指标见 summary.json；所有配对原始统计可由四组 JSON 复算。"]
    (root/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    values, manifest, summary, extended = analyze(args.root)
    dump(args.root/"chart_contract.json", {"surface": "standalone PNG/PDF research figures",
        "source": "manifest plus complete per-clip JSON", "palette": COLORS,
        "figures": {"mechanisms": "history line (8 times), event line (9 times), partner grouped bars; sensitivity only",
                    "quality": "two panels of paired model-minus-none dot-and-95%-intervals; mean expression MSE"},
        "comparison_units": "source-cluster bootstrap, same valid elements", "grayscale": "markers, dashes, hatched/open bars",
        "selection": "fixed before model inference; all four controls reported"})
    figures(args.root, summary, extended)
    markdown(args.root, values, manifest, summary, extended)


if __name__ == "__main__":
    main()
