import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm
from wav2vec import Wav2Vec2Model

class DualSpeakerJointEncoder(nn.Module):
    def __init__(self, blendshape_dim=56, feature_dim=256, audio_config=None):
        super(DualSpeakerJointEncoder, self).__init__()
        self.audio_encoder1 = (Wav2Vec2Model(audio_config) if audio_config is not None else
                               Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self"))
        self.audio_encoder1.feature_extractor._freeze_parameters()
        self.audio_encoder2 = (Wav2Vec2Model(audio_config) if audio_config is not None else
                               Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self"))
        self.audio_encoder2.feature_extractor._freeze_parameters()
        
        self.audio_projection = nn.Linear(self.audio_encoder1.config.hidden_size, feature_dim)
        
        self.blendshape_encoder = nn.Sequential(
            nn.Linear(blendshape_dim, int(feature_dim // 2)),
            nn.ReLU(),
            nn.Linear(int(feature_dim // 2), feature_dim),
            nn.ReLU()
        )
    
    @torch.no_grad()
    def extract_audio_features(self, audio1, audio2, batch_size=16):
        """Extract both frozen speaker streams in bounded micro-batches."""
        self.require_frozen_audio_cache()
        if audio1.shape != audio2.shape:
            raise ValueError("paired audio tensors must have the same shape")
        batch_size = max(int(batch_size), 1)
        fps = 25
        frame_num = math.ceil(audio1.shape[1] / 16000 * fps)
        features1 = []
        features2 = []
        for start in range(0, audio1.shape[0], batch_size):
            first = audio1[start : start + batch_size]
            second = audio2[start : start + batch_size]
            features1.append(
                self.audio_projection(
                    self.audio_encoder1(first, frame_num=frame_num).last_hidden_state
                )
            )
            features2.append(
                self.audio_projection(
                    self.audio_encoder2(second, frame_num=frame_num).last_hidden_state
                )
            )
        if not features1:
            empty = audio1.new_zeros((0, frame_num, self.audio_projection.out_features))
            return empty, empty.clone()
        return torch.cat(features1, dim=0), torch.cat(features2, dim=0)

    def require_frozen_audio_cache(self):
        if any(p.requires_grad for module in (self.audio_encoder1, self.audio_encoder2, self.audio_projection)
               for p in module.parameters()):
            raise ValueError("Audio caching requires frozen acoustic backbones and projection")

    def forward_from_audio_features(self, audio_feat1, audio_feat2, blendshape):
        blendshape_feat = self.blendshape_encoder(blendshape)
        return audio_feat1, audio_feat2, blendshape_feat

    def forward(self, audio1, audio2, blendshape, audio_features=None):
        fps = 25
        if audio_features is None:
            frame_num = math.ceil(audio1.shape[1] / 16000 * fps)
            audio_feat1 = self.audio_projection(
                self.audio_encoder1(audio1, frame_num=frame_num).last_hidden_state
            )
            audio_feat2 = self.audio_projection(
                self.audio_encoder2(audio2, frame_num=frame_num).last_hidden_state
            )
            return self.forward_from_audio_features(
                audio_feat1, audio_feat2, blendshape
            )
        self.require_frozen_audio_cache()
        return self.forward_from_audio_features(
            audio_features[0], audio_features[1], blendshape
        )
    
class CrossModalTemporalEnhancer(nn.Module):
    def __init__(self, feature_dim=256):
        super(CrossModalTemporalEnhancer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.temporal_lstm = nn.LSTM(feature_dim, feature_dim // 2, num_layers=2, bidirectional=True, batch_first=True)

    
    def forward(self, audio_feat, blendshape_feat):
        cross_modal_feat, _ = self.cross_attention(audio_feat, blendshape_feat, blendshape_feat)
        temporal_feat, _ = self.temporal_lstm(cross_modal_feat)

        return temporal_feat

class DualSpeakerInteractionModule(nn.Module):
    def __init__(self, feature_dim=256):
        super(DualSpeakerInteractionModule, self).__init__()
        self.interaction_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim*2, nhead=4, batch_first=True), 
            num_layers=3
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=feature_dim*2, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, audio_feat1, temporal_feat):
        combined_feat = torch.cat([audio_feat1, temporal_feat], dim=-1)
        
        encoded_feat = self.interaction_encoder(combined_feat)
        
        enhanced_feat, _ = self.self_attention(encoded_feat, encoded_feat, encoded_feat)
        return self.dropout(enhanced_feat)

class ExpressiveSynthesisModule(nn.Module):
    def __init__(self, feature_dim=256, blendshape_dim=56, mod_factor=0.1):
        super(ExpressiveSynthesisModule, self).__init__()
        self.synthesis_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=feature_dim, nhead=4, batch_first=True), 
            num_layers=1
        )
        self.blendshape_predictor = nn.Linear(feature_dim, blendshape_dim)
        
        # Adaptive Expression Modulation integrated
        self.modulation_factor = mod_factor
        self.modulation_layer = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, interaction_feat):
        decoded_feat = self.synthesis_decoder(interaction_feat, interaction_feat)
        
        # Modulate the expressions adaptively
        modulation = self.modulation_layer(decoded_feat)
        modulated_feat = decoded_feat + self.modulation_factor * modulation
        
        blendshape_output = self.blendshape_predictor(modulated_feat)
        return blendshape_output

class DualTalkModel(nn.Module):
    def __init__(self, args):
        super(DualTalkModel, self).__init__()
        blendshape_dim = args.blendshape_dim
        feature_dim = args.feature_dim 
        
        self.joint_encoder = DualSpeakerJointEncoder(blendshape_dim, feature_dim, getattr(args, "audio_config", None))
        self.temporal_enhancer = CrossModalTemporalEnhancer(feature_dim)
        self.interaction_module = DualSpeakerInteractionModule(feature_dim)
        self.synthesis_module = ExpressiveSynthesisModule(feature_dim*2, blendshape_dim)
        
    def forward(self, audio1, audio2, blendshape):
        audio_feat1, audio_feat2, blendshape_feat = self.joint_encoder(audio1, audio2, blendshape)
        temporal_feat = self.temporal_enhancer(audio_feat2, blendshape_feat)
        interaction_feat = self.interaction_module(audio_feat1, temporal_feat)
        blendshape_output = self.synthesis_module(interaction_feat)
        
        return blendshape_output
