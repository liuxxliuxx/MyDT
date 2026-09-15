"""Identical target-defined eligible windows for actual/oracle semantic controls."""
import hashlib
from emotion_ssm.utils.generation_losses import loss_settings,visual_window_mask


class SemanticWindowDataset:
    prefetch_rng_neutral=True
    def __init__(self,base,settings):
        self.base=base;self.settings=loss_settings(settings)
        for key in ('names','lengths','split','feature_sources','feature_digest'):
            setattr(self,key,getattr(base,key))
        if hasattr(base,'source_ids'):self.source_ids=base.source_ids
        self.manifest_digest=hashlib.sha256((base.manifest_digest+':semantic-windows-v1:'+str(self.settings['min_visual_frames'])).encode()).hexdigest()
    def __len__(self):return len(self.base)
    def packets(self,index):
        for packet,target,valid in self.base.packets(index):
            eligible=visual_window_mask(target,valid,self.settings)
            yield {**packet,'semantic_window_qualified':bool(eligible.all())},target,valid&eligible[:,None]
