"""Count-weighted metrics, including missing counterfactual candidates."""
import re
import torch
import torch.distributed as dist


class WeightedStatistics:
    def __init__(self):
        self.sums, self.counts = {}, {}

    def add(self, name, value, count=1):
        self.sums[name] = self.sums.get(name, 0.) + float(value) * float(count)
        self.counts[name] = self.counts.get(name, 0.) + float(count)

    def dynamics(self, values, prefix=""):
        events = float(values.get("valid_events", 1))
        for name, value in values.items():
            if name.endswith("_count") or name in ("valid_events", "cf_anchors", "cf_valid_anchors", "cf_pairs", "cf_pair_correct", "cf_margin_pair_correct"):
                continue
            horizon = re.search(r"h(\d+)$", name)
            count = float(values.get(f"h{horizon.group(1)}_count", 0)) if horizon else events
            if name in ("counterfactual", "cf_ranking_accuracy"):
                count = float(values.get("cf_valid_anchors", 0))
            self.add(prefix + name, value, count)
        self.add(prefix + "cf_candidate_coverage", float(values.get("cf_valid_anchors", 0)) /
                 max(float(values.get("cf_anchors", 0)), 1), values.get("cf_anchors", 0))
        self.add(prefix + "cf_pair_accuracy", float(values.get("cf_pair_correct", 0)) /
                 max(float(values.get("cf_pairs", 0)), 1), values.get("cf_pairs", 0))
        self.add(prefix + "cf_margin_success_rate", float(values.get("cf_margin_pair_correct", 0)) /
                 max(float(values.get("cf_pairs", 0)), 1), values.get("cf_pairs", 0))

    def finalize(self):
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, (self.sums, self.counts))
            self.sums, self.counts = {}, {}
            for sums, counts in gathered:
                for name in sums:
                    self.sums[name] = self.sums.get(name, 0.) + sums[name]
                    self.counts[name] = self.counts.get(name, 0.) + counts[name]
        result = {name: self.sums[name] / max(self.counts[name], 1) for name in self.sums}
        result.update({name+"_count": count for name, count in self.counts.items()
                       if "cf_" in name})
        return result


class RepresentationTotals:
    def __init__(self):
        self.values = {}

    def update(self, output, domains, subset_names):
        for domain in domains.unique():
            for index, name in enumerate(subset_names):
                mask = (domains == domain) & output.valid_subsets[:, index]
                value = output.aff[mask, index].detach().double().cpu()
                if not len(value):
                    continue
                key = f"domain{int(domain)}_{name}"
                n, total, square = self.values.get(key, (0, torch.zeros(value.shape[-1]), torch.zeros(value.shape[-1])))
                self.values[key] = (n+len(value), total+value.sum(0), square+value.square().sum(0))

    def metrics(self):
        if dist.is_available() and dist.is_initialized():
            gathered = [None]*dist.get_world_size()
            dist.all_gather_object(gathered, self.values)
            self.values = {}
            for values in gathered:
                for key, (n, total, square) in values.items():
                    old = self.values.get(key, (0, torch.zeros_like(total), torch.zeros_like(square)))
                    self.values[key] = (old[0]+n, old[1]+total, old[2]+square)
        result = {}
        for key, (n, total, square) in self.values.items():
            result[key+"_samples"] = n
            if n > 1:
                result[key+"_aff_std"] = float((square/n-(total/n).square()).clamp_min(0).sqrt().mean())
                result[key+"_aff_cosine"] = float((total.square().sum()-square.sum())/(n*(n-1)))
        return result
