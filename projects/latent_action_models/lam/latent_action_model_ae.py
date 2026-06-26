# References:
#     Genie: Generative Interactive Environments (https://arxiv.org/abs/2402.15391)
#     Latent Action Pretraining from Videos (https://arxiv.org/abs/2410.11758)
#     https://github.com/pytorch/torchtitan/blob/6cb13c7a97eed890f7df872261c9bf1523959dee/torchtitan/models/llama/model.py

import logging
import warnings
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lam.loss_scheduler import CustomLossScheduler
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model

from robot_wm.modeling.modules.positional_embedding import PositionalEmbedding
from robot_wm.modeling.networks.st_forward_model import STForwardModel
from robot_wm.modeling.networks.st_inverse_model import STInverseModel
from robot_wm.modeling.tokenizers.rgb.base import RGBTokenizer
from robot_wm.utils.model import log_model_size

logger = logging.getLogger(__name__)


class AELatentActionModel(nn.Module):
    def __init__(
        self,
        rgb_tokenizer: RGBTokenizer,
        rgb_pos_embed: PositionalEmbedding,
        action_encoder: nn.Module,
        quantizer: nn.Module,
        action_decoder: nn.Module,
        action_pos_embed: PositionalEmbedding,
        forward_model: STForwardModel,
        loss_scheduler: CustomLossScheduler,
        config: DictConfig,
        pre_training: bool = False,
        lora_finetune: bool = False,
        lora_rank: int = 8,
        unfreeze_first_layer: bool = False,
    ):
        super().__init__()
        self.rgb_tokenizer = rgb_tokenizer
        for p in self.rgb_tokenizer.parameters():
            p.requires_grad = False
        self.action_encoder = action_encoder
        self.rgb_pos_embed = rgb_pos_embed
        self.quantizer = quantizer
        self.action_decoder = action_decoder
        self.action_pos_embed = action_pos_embed
        self.forward_model = forward_model
        self.loss_scheduler = loss_scheduler
        self.config = config
        self.ae_only = True
        self.init_weights()

        if pre_training:
            for p in self.forward_model.parameters():
                p.requires_grad = False
            for p in self.action_pos_embed.parameters():
                p.requires_grad = False
            for p in self.rgb_pos_embed.parameters():
                p.requires_grad = False

            if unfreeze_first_layer:
                for p in self.forward_model.context_proj.parameters():
                    p.require_grad = True
                for p in self.forward_model.layers["0:query_attn"].parameters():
                    p.requires_grad = True
                for p in self.forward_model.layers["0:context_attn"].parameters():
                    p.requires_grad = True
                for p in self.forward_model.layers["0:cross_attn"].parameters():
                    p.requires_grad = True

        self.lora_config = LoraConfig(
            r=8,  # rank
            lora_alpha=32,
            target_modules=["linear"],  # or specific layer names
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",  # or appropriate type
        )
        if lora_finetune:
            self.forward_model = get_peft_model(self.forward_model, self.lora_config)
            logger.info("Enabled LoRA finetuning.")

        log_model_size(self, logger.info)

    def init_weights(self):
        self.action_encoder.init_weights(init_std=0.02)
        self.quantizer.init_weights()
        self.action_pos_embed.init_weights()
        self.forward_model.init_weights()
        if self.action_decoder is not None:
            self.action_decoder.init_weights(init_std=0.02)

    @classmethod
    def from_snapshot(cls, snapshot_path: str, *args, **kwargs):
        model = cls(*args, **kwargs)
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        state_dict = snapshot["model"]
        # remove incompatible keys from state_dict
        model_state_dict = model.state_dict()
        for k in model_state_dict:
            if k in state_dict and state_dict[k].shape != model_state_dict[k].shape:
                del state_dict[k]
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Initialized from: {snapshot_path}; {missing_keys = }")
        return model

    @torch.no_grad()
    def _forward_tokenizer_encode(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.rgb_tokenizer.encode(rgb)
        x = rearrange(x, "N T d h w -> N T h w d")
        return x.detach()  # N T h w d

    def _forward_action_encode(
        self,
        action: torch.Tensor,
        morphology_index: torch.Tensor,
        ee_action_dim,
        rgb_tokens,
    ) -> torch.Tensor:
        z = self.action_encoder(
            action,
            morphology_index=morphology_index,
            ee_action_dim=ee_action_dim,
            rgb_tokens=self.rgb_pos_embed(rgb_tokens),
        )
        if isinstance(z, dict):
            B, T, A, D = action.shape
            all_indices = []
            actions_concat = []

            for d in z.values():
                mask = d["mask"]
                if not mask.any():
                    continue
                indices = mask.nonzero(as_tuple=True)[0]
                all_indices.append(indices)
                actions_concat.append(d["actions"])

            all_indices = torch.cat(all_indices, dim=0).to(
                action.device
            )  # shape [B, T-1, A]
            actions_all = torch.cat(actions_concat, dim=0).to(
                action.device
            )  # shape [B, D]
            inverse_perm = torch.argsort(all_indices)
            z_out = actions_all[inverse_perm]
        else:
            z_out = z
        return z_out  # [N, T-1, A, D]

    @torch.no_grad()
    def _forward_tokenizer_decode(
        self, x: torch.Tensor, out_shape: Sequence[int]
    ) -> torch.Tensor:
        assert x.shape[1] == out_shape[1]
        x = rearrange(x, "N T h w d -> N T d h w")
        rgb_hat = self.rgb_tokenizer.decode(x, out_shape)
        return rgb_hat  # N T C H W

    def _forward_forward_model(
        self, x: torch.Tensor, zhat: torch.Tensor
    ) -> torch.Tensor:
        x = self.rgb_pos_embed(x)  # add positional embedding
        zhat = self.action_pos_embed(zhat)  # add positional embedding
        xhat = self.forward_model(x=x, context=zhat)
        return xhat  # n t h w dx

    def _forward_action_decoder(
        self,
        z: torch.Tensor,
        zhat: torch.Tensor,
        decode_zhat: bool = False,
        morphology_index: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decodes latent actions into ground truth actions.
        Args:
            z: torch.Tensor [N, T-1, A, D]
            zhat: torch.Tensor [N, T-1, A, D]
            tokenized_rgb: torch.Tensor [N, T, C, H, W]
        Returns:
            torch.Tensor [N, T-1, A, D]
        """
        action_z = zhat if decode_zhat else z
        if not self.config["backprog_action_decoding"]:
            action_z = action_z.detach()
        a = self.action_decoder(action_z, morphology_index=morphology_index)
        return a

    def _forward_loss(self, xhat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(input=xhat, target=x)
        return loss

    def _multi_action_decoding_loss(
        self,
        decoded_actions: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
        loss_type: str = "l2",
    ):
        """Calculates the action decoding loss for multiple morphologies.
        Args:
            decoded_actions: dict[str, torch.Tensor] {morphology_id: [N, T-1, 1, A*D]}
            actions: dict[str, torch.Tensor] {morphology_id: [N, T-1, A, D]}
        Returns:
            torch.Tensor []
        """
        total_loss = 0.0
        for morph_id in decoded_actions.keys():
            mask = decoded_actions[morph_id]["mask"]
            if not mask.any():
                continue
            decoded_action = decoded_actions[morph_id]["actions"]
            action = actions[mask]
            B, _, _, morphology_dim = decoded_action.shape
            loss = self._action_decoding_loss(
                decoded_actions=decoded_action,
                actions=action[..., :morphology_dim],
                loss_type=loss_type,
            )
            total_loss += loss * B
        total_loss /= len(actions)  # average over batch
        return total_loss

    def _action_decoding_loss(
        self,
        decoded_actions: torch.Tensor,
        actions: torch.Tensor,
        loss_type: str = "l2",
    ) -> torch.Tensor:
        """Calculates the action decoding loss.
        Args:
            decoded_actions: torch.Tensor [N, T-1, 1, A*D]
            actions: torch.Tensor [N, T-1, A, D]
        Returns:
            torch.Tensor []
        """
        if decoded_actions.ndim == 3:
            actions = rearrange(actions, "N T A D -> N T (A D)")
        assert (
            decoded_actions.shape == actions.shape
        ), f"{decoded_actions.shape} != {actions.shape}"
        if loss_type == "l2":
            loss = F.mse_loss(input=decoded_actions, target=actions)
        elif loss_type == "l1":
            loss = F.l1_loss(input=decoded_actions, target=actions)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return loss

    @torch.no_grad()
    def _generate_random_action_future(
        self, rgb: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Generate future predictions based on random actions (just for the sake of visualization).
        Args:
            rgb: torch.Tensor [N, T, C, H, W]
        Returns:
            torch.Tensor [N, T-1, A, D]
        """
        # generate latent actions
        x = self._forward_tokenizer_encode(rgb=rgb)
        z = self._forward_action_encode(
            action,
            morphology_index=kwargs.get("morphology_index", None),
            ee_action_dim=kwargs.get("ee_action_dim", None),
            rgb_tokens=x,
        )
        if self.ae_only:
            zhat = z
        else:
            zhat = self.quantizer(z)["zhat"]
        zhat = torch.randn_like(zhat)  # random actions

        # rollout future predictions
        xhat = x.detach().clone()
        T = xhat.shape[1]
        for idx in range(1, T):
            xhat[:, idx] = self._forward_forward_model(
                xhat[:, :idx], zhat[:, 1 : idx + 1]
            )[:, -1]

        # decode
        rgbhat = self._forward_tokenizer_decode(xhat, rgb.shape)
        return rgbhat

    @torch.no_grad()
    def _generate_future(self, rgb: torch.Tensor, action, **kwargs) -> torch.Tensor:
        # generate latent actions
        x = self._forward_tokenizer_encode(rgb=rgb)
        z = self._forward_action_encode(
            action,
            morphology_index=kwargs.get("morphology_index", None),
            ee_action_dim=kwargs.get("ee_action_dim", None),
            rgb_tokens=x,
        )
        if self.ae_only:
            zhat = z
        else:
            zhat = self.quantizer(z)["zhat"]

        # rollout future predictions
        xhat = x.detach().clone()
        T = xhat.shape[1]
        for idx in range(1, T):
            xhat[:, idx] = self._forward_forward_model(
                xhat[:, :idx], zhat[:, 1 : idx + 1]
            )[:, -1]

        # decode
        rgbhat = self._forward_tokenizer_decode(xhat, rgb.shape)
        return rgbhat

    @torch.no_grad()
    def _generate_alternate_future(
        self, rgb: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # generate latent actions
        x = self._forward_tokenizer_encode(rgb=rgb)

        z = self._forward_action_encode(
            action,
            morphology_index=kwargs.get("morphology_index", None),
            ee_action_dim=kwargs.get("ee_action_dim", None),
            rgb_tokens=x,
        )

        if self.ae_only:
            zhat = z
        else:
            zhat = self.quantizer(z)["zhat"]

        # use latent actions from a different rollout
        if zhat.shape[0] == 1:
            warnings.warn("batch size is 1; alternate future is invalid")

        zhat = torch.roll(zhat, shifts=1, dims=0)

        # rollout future predictions
        xhat = x.detach().clone()
        T = xhat.shape[1]
        for idx in range(1, T):
            xhat[:, idx] = self._forward_forward_model(
                xhat[:, :idx], zhat[:, 1 : idx + 1]
            )[:, -1]

        # decode
        rgbhat = self._forward_tokenizer_decode(xhat, rgb.shape)
        return rgbhat

    @torch.no_grad()
    def _generate_one_step_future(
        self, rgb: torch.Tensor, action, **kwargs
    ) -> torch.Tensor:
        x = self._forward_tokenizer_encode(rgb=rgb)
        z = self._forward_action_encode(
            action,
            morphology_index=kwargs.get("morphology_index", None),
            ee_action_dim=kwargs.get("ee_action_dim", None),
            rgb_tokens=x,
        )
        if self.ae_only:
            zhat = z
        else:
            zhat = self.quantizer(z)["zhat"]
        xhat = self._forward_forward_model(x[:, :-1], zhat[:, 1:])
        xhat = torch.cat([x[:, :1], xhat], dim=1)  # add context
        rgbhat = self._forward_tokenizer_decode(xhat, rgb.shape)
        return rgbhat

    @torch.no_grad()
    def _encode_decode(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self._forward_tokenizer_encode(rgb=rgb)  # encode
        rgbhat = self._forward_tokenizer_decode(x, rgb.shape)  # decode
        return rgbhat

    @torch.no_grad()
    def _compose_visualization(self, images):
        # stack the images vertically
        side_by_side = torch.cat(images, dim=3)  # [B, T - 1, C, 2 * H, W]

        # convert rgb range to 0-1
        if self.rgb_tokenizer._input_range == "normal":
            C = side_by_side.shape[-3]
            device = side_by_side.device
            mean = torch.tensor(self.rgb_tokenizer._input_mean, device=device)
            std = torch.tensor(self.rgb_tokenizer._input_std, device=device)
            side_by_side = torch.clamp(
                side_by_side * std.view(C, 1, 1) + mean.view(C, 1, 1), 0, 1
            )
        elif self.rgb_tokenizer._input_range == "[0, 1]":
            side_by_side = torch.clamp(side_by_side, 0, 1)
        else:
            raise ValueError(f"Unknown input range: {self.rgb_tokenizer._input_range}")

        return side_by_side

    @torch.no_grad()
    def visualize(
        self,
        rgb: torch.Tensor,
        actions: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        reconstructions = self._encode_decode(rgb)
        predictions = self._generate_future(rgb, actions, **kwargs)
        alternate_predictions = self._generate_alternate_future(rgb, actions, **kwargs)
        one_step_predictions = self._generate_one_step_future(rgb, actions, **kwargs)
        side_by_side = self._compose_visualization(
            [
                reconstructions,
                one_step_predictions,
                predictions,
                alternate_predictions,
            ]
        )
        return side_by_side

    def custom_to(self, *args, **kwargs):
        # ugly, should not be required, but it is for the JEPA2 tokenizer
        self.rgb_tokenizer = self.rgb_tokenizer.to(*args, **kwargs)
        return self

    def get_sampling_loss(self, x, zhat) -> str:
        # calculate sampling horizon
        n, t, h, w, d = x.shape
        horizon = self.config["sampling_horizon"]
        horizon = t - 1 if horizon == -1 else min(t - 1, horizon)

        # TODO: it would be nice to offset the temporal positional embeddings when
        # horizon < t - 1. Currently, we always start from the first temporal
        # position.

        # sample future predictions within `torch.no_grad()` (inplace)
        with torch.no_grad():
            xp = torch.empty((n, horizon, h, w, d), device=x.device, dtype=x.dtype)
            xp[:, :1] = x[:, :1].detach().clone()
            for idx in range(1, horizon):
                xp[:, idx] = self._forward_forward_model(
                    xp[:, :idx], zhat[:, 1 : idx + 1]
                )[:, -1]

        # When amp is enabled, using `torch.no_grad()` causes an error, so we need
        # to clear the autocast cache. See:
        # https://github.com/pytorch/pytorch/issues/65766#issuecomment-932511164
        if torch.is_autocast_enabled():
            torch.clear_autocast_cache()

        # use future predictions for forward pass
        xhat = self._forward_forward_model(xp[:, :horizon], zhat[:, 1 : horizon + 1])
        recon_loss = self._forward_loss(xhat, x[:, 1 : horizon + 1])

        return recon_loss

    def forward(
        self,
        rgb: torch.Tensor,
        actions: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Returns prediction loss.
        Args:
            rgb: torch.Tensor [N, T, C, H, W]
        Returns:
            torch.Tensor []
        """

        self.aux_losses = {}
        total_loss = 0
        cb_only = False
        action_decode = False
        decode_zhat = self.config.get("decode_zhat", False)

        selected_loss = "teacher-forcing"
        if self.training:
            selected_loss = self.loss_scheduler.get_loss()
            self.loss_scheduler.step()

        if "cb-only-" in selected_loss:
            cb_only = True
            selected_loss = selected_loss.replace("cb-only-", "")
            if "decode-z-" in selected_loss:
                decode_zhat = False
                selected_loss = selected_loss.replace("decode-z-", "")

        if "action-decode-" in selected_loss:
            action_decode = True
            selected_loss = selected_loss.replace("action-decode-", "")

        assert selected_loss in [
            "pretrain-teacher-forcing",
            "pretrain-sampling",
            "teacher-forcing",
            "sampling",
            "action-only-teacher-forcing",
            "action-only-sampling",
            "forward-only-teacher-forcing",
            "forward-only-sampling",
        ]

        x = self._forward_tokenizer_encode(rgb=rgb)
        z = self._forward_action_encode(
            actions,
            morphology_index=kwargs.get("morphology_index", None),
            ee_action_dim=kwargs.get("ee_action_dim", None),
            rgb_tokens=x,
        )
        if self.ae_only:  # autoencoder only, no quantization
            zhat = z
        else:
            vq_out = self.quantizer(z, cb_only=cb_only)
            zhat, qloss = vq_out["zhat"], vq_out["loss"]
            total_loss += qloss
            self.aux_losses.update({**vq_out["logs"]})

        if selected_loss.startswith("pretrain-"):
            zhat = 0 * zhat
            selected_loss = selected_loss.replace("pretrain-", "")

        if selected_loss == "teacher-forcing":
            xhat = self._forward_forward_model(x[:, :-1], zhat[:, 1:])
            recon_loss = self._forward_loss(xhat, x[:, 1:])
        elif selected_loss == "forward-only-teacher-forcing":
            xhat = self._forward_forward_model(
                x[:, :-1], zhat[:, 1:].clone().detach()
            )  # the wm gradient should not backpropagate to the action encoder models
            recon_loss = self._forward_loss(xhat, x[:, 1:])

        if selected_loss == "sampling":
            recon_loss = self.get_sampling_loss(x, zhat)
        elif selected_loss == "forward-only-sampling":
            recon_loss = self.get_sampling_loss(x, zhat.clone().detach())

        if not selected_loss.startswith("action-only-"):
            total_loss += recon_loss
            self.aux_losses.update({"recon_loss": recon_loss.detach().clone()})
        else:
            assert (
                action_decode
            ), "Action decoding should be enabled for action-only losses."

        if action_decode and actions is not None:
            ahat = self._forward_action_decoder(
                z,
                zhat,
                x,
                decode_zhat=decode_zhat,
                morphology_index=kwargs.get("morphology_index", None),
            )
            if isinstance(ahat, dict):
                # multi-action decoding
                act_decoding_loss = self._multi_action_decoding_loss(
                    decoded_actions=ahat,
                    actions=actions,
                    loss_type=self.config.get("action_decoding_loss_type", "l2"),
                )
            else:
                act_decoding_loss = self._action_decoding_loss(
                    decoded_actions=ahat,
                    actions=actions,
                    loss_type=self.config.get("action_decoding_loss_type", "l2"),
                )
            action_decoding_weight = self.config["action_decoding_weight"]
            total_loss += action_decoding_weight * act_decoding_loss
            self.aux_losses.update(
                {
                    "act_decoding_loss": act_decoding_loss.detach().clone(),
                }
            )

        return total_loss

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["loss_scheduler"] = self.loss_scheduler.state_dict()
        return state_dict

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        if "loss_scheduler" in state_dict:
            self.loss_scheduler.load_state_dict(state_dict["loss_scheduler"])
            del state_dict["loss_scheduler"]
        return super().load_state_dict(
            state_dict=state_dict, strict=strict, assign=assign
        )


if __name__ == "__main__":
    from robot_wm.modeling.modules.finite_scalar_quantization import FSQ
    from robot_wm.modeling.modules.vector_quantization import VectorQuantizer
    from robot_wm.modeling.tokenizers.rgb.cosmos import CosmosImageTokenizer

    logging.basicConfig(
        format="%(levelname)-8s %(name)s %(asctime)s %(message)s", level=logging.INFO
    )

    # data
    N, T, C, H, W = 7, 9, 3, 180, 320
    rgb = torch.randn(N, T, C, H, W).to("cuda").to(torch.bfloat16)
    logger.info(f"{rgb.shape = }")

    # quantizer
    use_fsq = False
    if use_fsq:
        quantizer_levels = [8, 5, 5, 5]
        latent_action_dim = len(quantizer_levels)
        quantizer = FSQ(levels=quantizer_levels)
    else:
        latent_action_dim, latent_action_codebook_size = 256, 8
        quantizer = VectorQuantizer(latent_action_dim, latent_action_codebook_size)

    # model
    dim, rgb_tokenizer_dim = 256, 16
    num_layers, num_heads, num_output_layers, num_actions = 8, 16, 1, 1
    rope_end, rope_theta = 2 * max(T // 8, H // -8 * W // -8), 50_000
    latent_action_model = AELatentActionModel(
        rgb_tokenizer=CosmosImageTokenizer(
            tokenizer_name="Cosmos-0.1-Tokenizer-CI8x8",
            tokenizer_type="both",
            output_dim=rgb_tokenizer_dim,
        ),
        rgb_pos_embed=PositionalEmbedding(
            T, -(H // -8), -(W // -8), dim=rgb_tokenizer_dim, use_sincos=True
        ),
        inverse_model=STInverseModel(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_output_layers=num_output_layers,
            num_actions=num_actions,
            input_dim=rgb_tokenizer_dim,
            output_dim=latent_action_dim,
            rope_end=rope_end,
            rope_theta=rope_theta,
        ),
        quantizer=quantizer,
        action_pos_embed=PositionalEmbedding(
            T, num_actions, dim=latent_action_dim, use_sincos=True
        ),
        forward_model=STForwardModel(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            x_dim=rgb_tokenizer_dim,
            context_dim=latent_action_dim,
            output_dim=rgb_tokenizer_dim,
            rope_end=rope_end,
            rope_theta=rope_theta,
            use_video_cross_attention=True,
        ),
        loss_scheduler=CustomLossScheduler(
            pretrain_steps=10,
            warmup_steps=10,
            total_steps=100,
            losses=["sampling"],
            seed=0,
        ),
        config=DictConfig({"sampling_horizon": 4}),
    )
    latent_action_model = latent_action_model.to("cuda").to(torch.bfloat16)

    # generate latent actions
    # actions = latent_action_model.generate(rgb)
    # logger.info(f"{actions.shape = }")

    # generate future
    # future = latent_action_model.generate(rgb, return_future=True)
    # logger.info(f"{future.shape = }")

    # calculate loss
    loss = latent_action_model(rgb)
    logger.info(f"{loss.item() = :0.3f}")

    # print gpu usage
    gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    logger.info(f"GPU memory (allocated): {gpu_allocated:0.1f} GB")
    gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
    logger.info(f"GPU memory ( reserved): {gpu_reserved:0.1f} GB")
