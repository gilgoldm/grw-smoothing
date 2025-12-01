from typing import Tuple, Dict, TypeVar

import torch
from torch import nn, Tensor

from grw_smoothing_models.model.attention import TransformerParams, SimpleViT
from grw_smoothing_models.model.video_backbone import VideoBackbone

Encoder = TypeVar('Encoder', bound=VideoBackbone)


class ActionRecognitionModel(nn.Module):
    def __init__(self,
                 video_backbone: Encoder,
                 T: int,
                 transformer_params: TransformerParams,
                 dropout: float = 0.,
                 stochastic_depth: float = 0.):
        super(ActionRecognitionModel, self).__init__()
        assert transformer_params.embed_dim == video_backbone.embed_dim
        embed_dim = transformer_params.embed_dim
        self.T: int = T
        self.embed_dim: int = embed_dim
        self.video_backbone: VideoBackbone = video_backbone
        self.transformer_params: TransformerParams = transformer_params
        self.num_classes: int = transformer_params.num_classes
        self.ln = nn.LayerNorm(normalized_shape=self.embed_dim)
        self.attention = SimpleViT(
            transformer_params=transformer_params,
            dropout=dropout,
            stochastic_depth= stochastic_depth
        )

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param X: (B,N,C,H,W) tensor
        :return: (B,num_classes) logits tensor, where B is the batch size,
                 and (B,n_clips,T,K) Z-embedding tensor, where n_clips=N//T
        """
        B, N, C, H, W = X.shape
        assert N % self.T == 0
        n_clips = N // self.T
        Z = self.video_backbone(X)
        assert Z.shape == (B, N, self.embed_dim)
        Z_ = self.ln(Z)
        logits = self.attention(Z_)
        Z = Z.reshape(B,n_clips,self.T,self.embed_dim)
        return logits, Z

    def save(self, model_path: str, num_samples: int, ema_weights = None):
        snapshot = {'model_state': self.state_dict(),
                    'ema_weights': ema_weights,
                    'num_samples': num_samples,
                    'transformer_params': self.transformer_params,
                    'backbone': self.video_backbone.__class__,
                    'T': self.T,
                    'version': 10}
        torch.save(snapshot, model_path)
        print(f"Saved snapshot to {model_path}, num_samples = {num_samples}")

    @staticmethod
    def load(model_path: str, video_backbone: VideoBackbone, dropout: float = 0., stochastic_depth: float = 0.,
             load_ema: bool = False, patch=None) \
            -> Tuple['ActionRecognitionModel', int]:
        snapshot: Dict = torch.load(model_path, map_location='cpu', pickle_module=patch)
        num_samples = snapshot['num_samples']
        transformer_params = snapshot['transformer_params']
        if snapshot['version'] == 10:
            T = snapshot['T']
            model: ActionRecognitionModel = ActionRecognitionModel(video_backbone=video_backbone,
                                                                   transformer_params=transformer_params,
                                                                   T=T,
                                                                   dropout=dropout,
                                                                   stochastic_depth= stochastic_depth)
            if load_ema:
                print(f'Loading EMA weights')
                from torch.optim.swa_utils import AveragedModel
                ema_model: AveragedModel = AveragedModel(model=model,
                                                         multi_avg_fn=lambda x: x,
                                                         use_buffers=True)
                ema_model.load_state_dict(snapshot['ema_weights'])
                model.load_state_dict(ema_model.module.state_dict())
            else:
                model.load_state_dict(snapshot['model_state'])
        else:
            raise Exception(f'Model version {snapshot["version"]} is no longer supported. Please upgrade manually to version 10.')
        print(f'Loaded action recognition model {model_path} with num_samples={num_samples}')
        return model, num_samples

    @staticmethod
    def replace_head(model_path: str,  # only supports replacing head for a V10 model. See from_v9
                     new_model_path: str,
                     video_backbone: Encoder,
                     transformer_params: TransformerParams) -> 'ActionRecognitionModel':
        old_model, num_samples = ActionRecognitionModel.load(model_path, video_backbone)
        video_backbone = old_model.video_backbone
        video_backbone.embed_dim = transformer_params.embed_dim
        new_model: ActionRecognitionModel = ActionRecognitionModel(
            video_backbone=old_model.video_backbone,
            transformer_params=transformer_params,
            T=old_model.T,
            dropout=old_model.attention.dropout,
            stochastic_depth= old_model.attention.stochastic_depth)
        new_model.save(new_model_path, num_samples)
        return new_model


