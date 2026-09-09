"""Additive prediction statistics; CCC is computed after global aggregation."""
import torch
import torch.distributed as dist
import torch.nn.functional as F

from emotion_ssm.metrics import classification_metrics


class PredictionStatistics:
    def __init__(self):
        self.values = {}

    @torch.no_grad()
    def update(self, name, prediction, target, valid, class_weights=None):
        valid = valid.detach().bool().cpu().reshape(-1)
        p = {k: v.detach().double().cpu() for k, v in prediction.items() if k in ("aff", "emotion", "intensity", "vad")}
        t = {k: v.detach().cpu() for k, v in target.items()}
        stats = {"matrix": torch.zeros(7, 7, dtype=torch.float64), "scalar": torch.zeros(7, dtype=torch.float64),
                 "vad": torch.zeros(6, 3, dtype=torch.float64)}
        if "aff" in t:
            mask = valid & (t["aff"].abs().sum(-1) > 0)
            error = 1-F.cosine_similarity(p["aff"], t["aff"].double(), dim=-1)
            error += F.smooth_l1_loss(p["aff"], t["aff"].double(), reduction="none").mean(-1)
            stats["scalar"][0], stats["scalar"][1] = error[mask].sum(), mask.sum()
        labeled = valid & (t["emotion"] >= 0)
        if labeled.any():
            labels = t["emotion"][labeled].long()
            logits = p["emotion"][labeled]
            indices = labels * 7 + logits.argmax(-1)
            stats["matrix"] = torch.bincount(indices, minlength=49).reshape(7, 7).double()
            weights = None if class_weights is None else class_weights.detach().double().cpu()
            stats["scalar"][2] = F.cross_entropy(logits, labels, weight=weights, reduction="sum")
            stats["scalar"][3] = len(labels) if weights is None else weights[labels].sum()
        intensity_valid = valid & t.get("intensity_mask", torch.ones_like(valid)).bool()
        difference = p["intensity"] - t["intensity"].double()
        stats["scalar"][4] = difference[intensity_valid].square().sum()
        stats["scalar"][5] = intensity_valid.sum()
        stats["scalar"][6] = difference[intensity_valid].abs().sum()
        mask = valid[:, None] & t["vad_mask"].bool()
        x, y = p["vad"], t["vad"].double()
        stats["vad"] = torch.stack([mask.sum(0), (x*mask).sum(0), (y*mask).sum(0),
            (x.square()*mask).sum(0), (y.square()*mask).sum(0), (x*y*mask).sum(0)])
        self._add(name, stats)

    def _add(self, name, stats):
        if name not in self.values:
            self.values[name] = {k: v.clone() for k, v in stats.items()}
        else:
            for key in stats:
                self.values[name][key] += stats[key]

    def merge(self, other, prefix=""):
        for key, value in other.values.items():
            self._add(prefix+key, value)

    def metrics(self):
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, self.values)
            self.values = {}
            for values in gathered:
                for key, stats in values.items():
                    self._add(key, stats)
        result = {}
        for name, stats in self.values.items():
            s, v = stats["scalar"], stats["vad"]
            n, sx, sy, sx2, sy2, sxy = v
            count = n.clamp_min(1)
            mx, my = sx/count, sy/count
            denominator = sx2/count-mx.square()+sy2/count-my.square()+(mx-my).square()
            ccc = 2*(sxy/count-mx*my)/denominator.clamp_min(1e-8)
            valid_ccc = n >= 2
            mse = (sx2+sy2-2*sxy).clamp_min(0).sum()/n.sum().clamp_min(1)
            mean_ccc = ccc[valid_ccc].mean() if valid_ccc.any() else ccc.new_zeros(())
            classification = classification_metrics(stats["matrix"])
            result.update({f"{name}_affect_loss": float(s[0]/s[1].clamp_min(1)),
                f"{name}_affect_count": float(s[1]), f"{name}_emotion_count": float(stats["matrix"].sum()),
                f"{name}_emotion_ce": float(s[2]/s[3].clamp_min(1)),
                f"{name}_macro_f1": classification["macro_f1"], f"{name}_uar": classification["uar"],
                f"{name}_intensity_mse": float(s[4]/s[5].clamp_min(1)),
                f"{name}_intensity_mae": float(s[6]/s[5].clamp_min(1)),
                f"{name}_intensity_count": float(s[5]), f"{name}_vad_mse": float(mse),
                f"{name}_vad_ccc": float(mean_ccc), f"{name}_vad_count": float(n.sum()),
                f"{name}_vad_loss": float(mse + (1-mean_ccc if valid_ccc.any() else 0.))})
        return result

    def observation(self, output, predictions, batch, subset_names):
        domains = batch["dataset_id"]
        for domain in domains.unique():
            for index, name in enumerate(subset_names):
                valid = (domains == domain) & output.valid_subsets[:, index]
                if valid.any():
                    self.update(f"domain{int(domain)}_{name}", {k: v[:, index] for k, v in predictions.items()},
                        {k: batch[k] for k in ("emotion", "intensity", "vad", "vad_mask", "intensity_mask") if k in batch}, valid)
