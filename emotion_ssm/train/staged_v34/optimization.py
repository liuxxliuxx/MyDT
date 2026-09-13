"""Alternating update/propagation blocks; moments persist across the boundary."""
from emotion_ssm.train.staged_v33.optimization import PersistentOptimizers as BaseOptimizers


class PersistentOptimizers(BaseOptimizers):
    def phase(self,phase):
        super().phase(phase)
        if phase=='joint':
            self.active=['update']
            for _,p in self.named['flow']:
                p.requires_grad_(False);p.grad=None
