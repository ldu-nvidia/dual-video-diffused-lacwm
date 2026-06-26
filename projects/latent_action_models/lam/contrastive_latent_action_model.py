import logging
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lam.loss_scheduler import CustomLossScheduler
from omegaconf import DictConfig
from torch.distributed.nn import all_gather

from robot_wm.modeling.modules.positional_embedding import PositionalEmbedding
from robot_wm.modeling.networks.st_forward_model import STForwardModel
from robot_wm.modeling.networks.st_inverse_model import STInverseModel
from robot_wm.modeling.tokenizers.rgb.base import RGBTokenizer
from robot_wm.modeling.tokenizers.action.base import ActionTokenizer
from robot_wm.utils.model import log_model_size
from robot_wm.modeling.tokenizers.action.simple import SequenceActionTokenizer

from .latent_action_model import LatentActionModel

logger = logging.getLogger(__name__)

class ContrastiveLatentActionModel(LatentActionModel):
    def __init__(
        self,
        rgb_tokenizer: RGBTokenizer,
        rgb_pos_embed: PositionalEmbedding,
        inverse_model: STInverseModel,
        quantizer: nn.Module,
        action_decoder: nn.Module,
        action_pos_embed: PositionalEmbedding,
        forward_model: STForwardModel,
        loss_scheduler: CustomLossScheduler,
        config: DictConfig,
    ):

        super().__init__(
            rgb_tokenizer,
            rgb_pos_embed,
            inverse_model,
            quantizer,
            action_decoder,
            action_pos_embed,
            forward_model,
            loss_scheduler,
            config,
        )
        self.contrastive_loss_weight = config.get('contrastive_loss_weight', 0)
        # TODO: input_dim=157 hardcoded for EgoDex
        clip_dim = config.get('clip_dim', 512)
        trf_kwargs = dict(hidden_dim=clip_dim, num_heads=8, num_layers=2, rope_end=1920, rope_theta=50000)
        self.gt_action_encoder = SequenceActionTokenizer(input_dim=157, output_dim=clip_dim, **trf_kwargs)
        self.pred_action_encoder = SequenceActionTokenizer(input_dim=quantizer.dim, output_dim=clip_dim, **trf_kwargs)
        self.gt_action_encoder.init_weights()
        self.pred_action_encoder.init_weights()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

    # CLIP loss, but for action latents vs. gt action embeddings
    def clip_loss(self, image_features, text_features, mask=None):
        all_image_features = torch.cat(all_gather(image_features), dim=0)
        all_text_features = torch.cat(all_gather(text_features), dim=0)

        all_image_features = F.normalize(all_image_features, dim=-1)
        all_text_features = F.normalize(all_text_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * all_image_features @ all_text_features.T
        logits_per_text = logits_per_image.T

        num_logits = logits_per_image.shape[0]
        device = image_features.device
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        if mask is not None:
            mask = torch.cat(all_gather(mask), dim=0)
            labels[mask] = -100

        loss = (
            F.cross_entropy(logits_per_image, labels, ignore_index=-100)
            + F.cross_entropy(logits_per_text, labels, ignore_index=-100)
        ) / 2

        return loss

    def forward(self, rgb: torch.Tensor, actions, **kwargs) -> torch.Tensor:
        """Returns prediction loss.
        Args:
            rgb: torch.Tensor [N, T, C, H, W]
        Returns:
            torch.Tensor []
        """

        self.aux_losses = {}

        selected_loss = "teacher-forcing"
        if self.training:
            selected_loss = self.loss_scheduler.get_loss()
            self.loss_scheduler.step()
        assert selected_loss == "teacher-forcing"

        x = self._forward_tokenizer_encode(rgb=rgb)
        z = self._forward_inverse_model(x)  # (B, T, atok=8 * 3, D)
        vq_out = self.quantizer(z)

        xhat = self._forward_forward_model(x[:, :-1], vq_out["zhat"][:, 1:])
        recon_loss = self._forward_loss(xhat, x[:, 1:])

        self.aux_losses.update(
            {"recon_loss": recon_loss.detach().clone(), **vq_out["logs"]}
        )

        total_loss = recon_loss + vq_out["loss"]

        # Extra contrastive loss
        if self.training and self.contrastive_loss_weight > 0:

            actions = rearrange(actions[:, :-1], 'b t chunk d -> b (t chunk) d') # Flatten actions across chunks
            gt_action_emb = self.gt_action_encoder(actions)  # (B, 1 + (T-1), d))
            pred_actions = rearrange(vq_out["zhat"][:, 1:], "b tm1 atok d -> b (tm1 atok) d")
            pred_action_emb = self.pred_action_encoder(pred_actions)  # (B, 1 + (T-1) * atok, D)

            contrastive_loss_action = self.clip_loss(gt_action_emb[:, 0], pred_action_emb[:, 0])
            total_loss = total_loss + self.contrastive_loss_weight * contrastive_loss_action
            self.aux_losses.update({
                "contrastive_loss_action": contrastive_loss_action.detach().clone(),
            })

        return total_loss

