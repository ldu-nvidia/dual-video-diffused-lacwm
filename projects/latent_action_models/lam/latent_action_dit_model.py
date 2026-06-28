# Latent-action world model with a Wan2.1-Fun-1.3B-Control diffusion forward model.
#
# Same skeleton as LatentActionModel (lam/latent_action_model.py) but the forward
# (world) model is a conditional video-diffusion model instead of STForwardModel:
#
#   rgb --WanVAE(per-frame)--> per-frame latents --inverse_model--> latent action z
#   history frames --WanVAE(temporal)--> reference latent  ─┐
#   latent action z --ActionToControl--> control latent     ┼─ y=[control|ref] (32ch)
#   future frames  --WanVAE(temporal)+noise--> noisy (16ch) ─┘
#       --> Wan DiT (frozen + LoRA) --flow-matching denoise--> future frames
#   action_decoder(z) --> GT robot action (L1 supervision, unchanged from LAM)
#
# Loss = flow_matching_mse(future) + action_decoding_weight * action_decoding_l1.

import logging
import warnings
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from einops import rearrange
from omegaconf import DictConfig

from lam.loss_scheduler import CustomLossScheduler
from robot_wm.modeling.modules.positional_embedding import PositionalEmbedding
from robot_wm.modeling.networks.st_inverse_model import STInverseModel
from robot_wm.modeling.networks.wan_forward_model import WanForwardModel
from robot_wm.modeling.tokenizers.rgb.wan_vae import WanVAETokenizer
from robot_wm.utils.model import log_model_size

logger = logging.getLogger(__name__)


class LatentActionDiTModel(nn.Module):
    def __init__(
        self,
        rgb_tokenizer: WanVAETokenizer,
        forward_model: WanForwardModel,
        loss_scheduler: CustomLossScheduler,
        config: DictConfig,
        rgb_pos_embed: PositionalEmbedding = None,
        inverse_model: STInverseModel = None,
        action_decoder: nn.Module = None,
        action_pos_embed: PositionalEmbedding = None,
        morphology_tokens: nn.Module = None,
        num_history_frames: int = 5,
        num_future_frames: int = 8,
        num_views: int = 3,
        latent_action_dim: int = 64,
        scheduler_config_path: str = "/scr/ravenh/VideoX-Fun/config/wan2.1/wan_civitai.yaml",
        null_prompt_path: str = "/scr/ravenh/wan_fun_1.3b_control/null_prompt_umt5.pt",
        text_dim: int = 4096,
        clip_seq_len: int = 257,
        clip_dim: int = 1280,
        # flow-matching timestep sampling
        weighting_scheme: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        viz_num_steps: int = 20,
        # quantizer/action_encoder kept for config compatibility (unused)
        quantizer: nn.Module = None,
        action_encoder: nn.Module = None,
    ):
        super().__init__()
        self.latent_action_dim = latent_action_dim
        self.num_history_frames = num_history_frames
        self.num_future_frames = num_future_frames
        self.num_views = num_views

        self.rgb_tokenizer = rgb_tokenizer  # frozen Wan VAE
        for p in self.rgb_tokenizer.parameters():
            p.requires_grad = False

        self.rgb_pos_embed = rgb_pos_embed
        self.inverse_model = inverse_model
        self.action_decoder = action_decoder
        self.action_pos_embed = action_pos_embed
        self.forward_model = forward_model
        self.loss_scheduler = loss_scheduler
        self.config = config
        self.morphology_tokens = morphology_tokens
        self.quantizer = None  # no VQ

        # number of *latent* frames that are treated as known history (reference)
        self.num_history_latent = self.rgb_tokenizer.latent_temporal_len(num_history_frames)

        # learnable temporal pooling: the temporal_ratio future actions falling in each
        # future latent frame are concatenated (temporal order) and combined by an MLP
        # into one control feature -- replaces uniform average pooling.
        tr = self.rgb_tokenizer.temporal_ratio
        self.action_pool = nn.Sequential(
            nn.Linear(tr * latent_action_dim, 2 * tr * latent_action_dim),
            nn.SiLU(),
            nn.Linear(2 * tr * latent_action_dim, latent_action_dim),
        )

        # flow-matching scheduler (for sigmas / timesteps)
        from omegaconf import OmegaConf

        sched_cfg = OmegaConf.to_container(OmegaConf.load(scheduler_config_path)["scheduler_kwargs"])
        sched_cfg.pop("scheduler_subpath", None)
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(**sched_cfg)
        # separate scheduler instance for sampling/visualization so that set_timesteps()
        # there does not clobber the training scheduler's 1000-step timeline.
        self.sample_scheduler = FlowMatchEulerDiscreteScheduler(**sched_cfg)
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.viz_num_steps = viz_num_steps

        self.text_dim, self.clip_seq_len, self.clip_dim = text_dim, clip_seq_len, clip_dim
        # cached null-prompt umT5 embedding -> [L, text_dim]
        try:
            null = torch.load(null_prompt_path, map_location="cpu").float()
            if null.ndim == 3:
                null = null[0]
            logger.info(f"loaded null prompt embedding {tuple(null.shape)} from {null_prompt_path}")
        except Exception as e:  # noqa
            warnings.warn(f"null prompt not found ({e}); using a single zero token")
            null = torch.zeros(1, text_dim)
        self.register_buffer("null_prompt", null, persistent=False)

        self.aux_losses = {}
        self.init_weights()
        log_model_size(self, logger.info)

    def init_weights(self):
        if self.rgb_pos_embed is not None:
            self.rgb_pos_embed.init_weights()
        if self.inverse_model is not None:
            self.inverse_model.init_weights()
        if self.action_pos_embed is not None:
            self.action_pos_embed.init_weights()
        self.forward_model.init_weights()
        self.loss_scheduler.reset()
        if self.action_decoder is not None:
            self.action_decoder.init_weights(init_std=0.02)

    # ------------------------------------------------------------- VAE helpers
    @torch.no_grad()
    def _encode_per_frame(self, rgb: torch.Tensor) -> torch.Tensor:
        z = self.rgb_tokenizer.encode_per_frame(rgb)  # [N,T,d,h,w]
        return rearrange(z, "n t d h w -> n t h w d").detach()

    @torch.no_grad()
    def _encode_clip(self, rgb: torch.Tensor) -> torch.Tensor:
        video = rearrange(rgb, "n t c h w -> n c t h w")
        return self.rgb_tokenizer.encode_temporal(video).detach()  # [N,d,Fp,h,w]

    # ----------------------------------------------------- latent-action paths
    def _forward_inverse_model(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rgb_pos_embed(x)
        return self.inverse_model(x=x)  # [N, T, A, D]

    def _add_morphology(self, z: torch.Tensor, morphology_index: torch.Tensor) -> torch.Tensor:
        if self.morphology_tokens is not None and morphology_index is not None:
            z = self.morphology_tokens(morphology_index).unsqueeze(1).unsqueeze(1) + z
        return z

    def _latent_actions(self, rgb: torch.Tensor, actions, morphology_index, Fp: int, K: int):
        """Generate ONLY the num_future latent actions (one per future transition).
        `actions` is unused here (the LAM derives the latent from frames); the explicit
        action world model overrides this method to encode `actions` directly.

        The inverse model is run over a (num_future + 1)-frame window = [last history
        frame | all future frames]; the boundary frame anchors the first action, so the
        window yields exactly num_future transition actions. We do not generate or use
        the history-region actions (the history is supplied to the world model via the
        reference channel, not as actions).

        Returns:
            z_future:      [N, num_future, A, D]  raw (for action-decoding supervision)
            z_control:     [N, Fp, D]             per-latent-frame control (history rows = 0)
            future_tokens: [N, num_future, h, w, d] per-frame latents of the future frames
        """
        win = max(self.num_history_frames - 1, 0)        # last history frame anchors action 1
        rgb_win = rgb[:, win:]                            # [N, num_future + 1, C, H, W]
        x_pf_win = self._encode_per_frame(rgb_win)        # [N, num_future + 1, h, w, d]
        z_win = self._forward_inverse_model(x_pf_win)     # [N, num_future + 1, A, D]
        z_future = z_win[:, 1:]                            # [N, num_future, A, D]
        future_tokens = x_pf_win[:, 1:]                    # [N, num_future, h, w, d]
        z_morph = self._add_morphology(z_future, morphology_index).mean(dim=2)  # [N, num_future, D]
        z_control = self._future_control(z_morph, Fp, K)  # [N, Fp, D]
        return z_future, z_control, future_tokens

    def _future_control(self, z_future: torch.Tensor, Fp: int, K: int) -> torch.Tensor:
        """Map the num_future latent actions onto the future latent frames [K:Fp] with a
        LEARNABLE MLP (not average pooling): the temporal_ratio actions that fall in each
        future latent frame are concatenated in temporal order and combined by self.action_pool.
        History latent frames [0:K] get zero control."""
        N, Fnum, D = z_future.shape
        tr = self.rgb_tokenizer.temporal_ratio
        n_future_lat = Fp - K
        assert Fnum == n_future_lat * tr, (
            f"learnable pool expects num_future ({Fnum}) == (Fp-K)*temporal_ratio "
            f"({n_future_lat}*{tr}); adjust num_history/num_future."
        )
        grouped = z_future.reshape(N, n_future_lat, tr * D)  # consecutive tr actions per future latent frame
        pooled = self.action_pool(grouped)                   # [N, n_future_lat, D]  (learnable)
        hist = z_future.new_zeros(N, K, D)                   # history latent frames -> zero control
        return torch.cat([hist, pooled], dim=1)              # [N, Fp, D]

    # ---------------------------------------------------- diffusion conditioning
    def _build_context(self, batch_size: int, device, dtype):
        null = self.null_prompt.to(device=device, dtype=dtype)
        return [null for _ in range(batch_size)]

    def _build_clip(self, batch_size: int, device, dtype):
        return torch.zeros(batch_size, self.clip_seq_len, self.clip_dim, device=device, dtype=dtype)

    def _get_sigmas(self, timesteps, n_dim, dtype, device):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_t = self.noise_scheduler.timesteps.to(device)
        step_idx = [(schedule_t == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_idx].flatten()
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _sample_timesteps(self, bsz, device):
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=1.29,
        )
        idx = (u * self.noise_scheduler.config.num_train_timesteps).long()
        idx = idx.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)
        return self.noise_scheduler.timesteps.to(device)[idx].to(device)

    # ------------------------------------------------------- action decoding (LAM)
    def _forward_action_decoder(self, z, tokenized_rgb=None, morphology_index=None):
        zhat = self._add_morphology(z, morphology_index)
        action_z = rearrange(zhat, "N T A D -> N T (A D)")
        return self.action_decoder(action_z, morphology_index, tokenized_rgb)

    def _action_decoding_loss(self, decoded_actions, actions, loss_type="l1", loss_mask=None):
        if decoded_actions.ndim == 3:
            actions = rearrange(actions, "N T A D -> N T (A D)")
        assert decoded_actions.shape == actions.shape, f"{decoded_actions.shape} != {actions.shape}"
        fn = F.l1_loss if loss_type == "l1" else F.mse_loss
        loss = fn(input=decoded_actions, target=actions, reduction="none")
        if loss_mask is not None:
            loss = loss * loss_mask.view(*loss_mask.shape, 1)
        return loss.mean()

    def _multi_action_decoding_loss(self, decoded_actions, actions, loss_type="l1", loss_mask=None):
        total = 0.0
        for morph_id in decoded_actions.keys():
            mask = decoded_actions[morph_id]["mask"]
            if not mask.any():
                continue
            decoded_action = decoded_actions[morph_id]["actions"]
            action = actions[mask]
            B, T, A, D = action.shape
            _, _, AE = decoded_action.shape
            morphology_dim = AE // A
            total += self._action_decoding_loss(
                decoded_actions=decoded_action,
                actions=action[..., :morphology_dim],
                loss_type=loss_type,
                loss_mask=loss_mask[mask],
            )
        return total

    def _build_loss_mask(self, rgb: torch.Tensor, mask: torch.Tensor, latents_shape) -> torch.Tensor:
        """Latent-space validity mask [N,1,Fp,1,w] for the flow loss, combining:
          - per-VIEW spatial validity: a width-stacked view that is a constant (black/padded)
            image is excluded -> the model is not trained to predict masked/missing views;
          - per-FRAME temporal validity: zero-padded frames (short trajectories, batch `mask`)
            are excluded.
        Masked regions remain as model INPUTS (history/control) but are not supervised."""
        N, _, Fp, h, w = latents_shape
        device = rgb.device
        nv = self.num_views

        # per-view: a view is "real" if its pixels vary (masked/padded views are constant)
        if nv > 1 and rgb.shape[-1] % nv == 0 and w % nv == 0:
            rv = rearrange(rgb, "n t c hh (v wv) -> n v (t c hh wv)", v=nv)
            view_valid = (rv.float().std(dim=-1) > 1e-3).float()        # [N, nv]
            width_mask = view_valid.repeat_interleave(w // nv, dim=1)   # [N, w]
        else:
            width_mask = torch.ones(N, w, device=device)

        # per-frame -> per-latent-frame (a latent frame is valid only if all its pixel frames are)
        if mask is not None:
            tr = self.rgb_tokenizer.temporal_ratio
            T = mask.shape[1]
            lat_valid = torch.ones(N, Fp, device=device)
            for li in range(Fp):
                frames = [t for t in range(T) if (0 if t == 0 else (t - 1) // tr + 1) == li]
                if frames:
                    lat_valid[:, li] = mask[:, frames].float().amin(dim=1)
            time_mask = lat_valid
        else:
            time_mask = torch.ones(N, Fp, device=device)

        return (time_mask[:, None, :, None, None] * width_mask[:, None, None, None, :]).float()

    # ------------------------------------------------------------------ forward
    def forward(self, rgb: torch.Tensor, actions: torch.Tensor = None,
                mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """rgb: [N, T, C, H, W] in [-1, 1]; T = num_history + num_future."""
        self.aux_losses = {}
        morphology_index = kwargs.get("morphology_index", None)
        device = rgb.device

        # gated localization of a CUDA illegal-memory-access: a synchronize after each stage
        # makes an async fault surface AT that stage's checkpoint (near-normal speed, unlike
        # full CUDA_LAUNCH_BLOCKING). The traceback line then names the offending stage.
        import os as _os
        _sync = bool(_os.environ.get("DIT_SYNC"))
        def _ck(stage):
            if _sync:
                torch.cuda.synchronize()  # if this raises IMA, `stage` (just produced) was the culprit

        # --- diffusion target: full-clip temporal latent ---
        latents = self._encode_clip(rgb).to(rgb.dtype)   # [N,16,Fp,h,w]
        _ck("encode_clip")
        N, Cl, Fp, h, w = latents.shape

        # reference (history) latent frames, rest zeroed -> conditioning channel
        ref = torch.zeros_like(latents)
        K = min(self.num_history_latent, Fp)
        ref[:, :, :K] = latents[:, :, :K]

        # --- exactly num_future latent actions (one per future transition) ---
        z_future, z_control, future_tokens = self._latent_actions(rgb, actions, morphology_index, Fp, K)
        z_control = z_control.to(rgb.dtype)
        _ck("latent_actions")

        # flow matching
        noise = torch.randn_like(latents)
        timesteps = self._sample_timesteps(N, device)
        sigmas = self._get_sigmas(timesteps, latents.ndim, latents.dtype, device)
        noisy = (1.0 - sigmas) * latents + sigmas * noise
        target = noise - latents
        _ck("flow_matching_setup")

        context = self._build_context(N, device, rgb.dtype)
        clip_fea = self._build_clip(N, device, rgb.dtype)
        v_pred = self.forward_model(noisy, timesteps, z_control, ref, context, clip_fea)
        _ck("forward_model_DiT")

        # masked flow loss: exclude masked/missing views and zero-padded frames from supervision
        loss_mask = self._build_loss_mask(rgb, mask, latents.shape)  # [N,1,Fp,1,w]
        if self.config.get("flow_loss_on_future_only", True) and Fp > K:
            fl_pred, fl_tgt, m = v_pred[:, :, K:], target[:, :, K:], loss_mask[:, :, K:]
        else:
            fl_pred, fl_tgt, m = v_pred, target, loss_mask
        se = (fl_pred.float() - fl_tgt.float()) ** 2
        # m is [N,1,Fp,1,w] and broadcasts over channels+height; expand_as so the denominator
        # counts the actual number of supervised scalar elements (not 1/(C*h) of them).
        m = m.expand_as(se)
        flow_loss = (se * m).sum() / m.sum().clamp(min=1.0)
        total_loss = flow_loss
        self.aux_losses["flow_loss"] = flow_loss.detach().clone()
        _ck("flow_loss")

        # --- action decoding: supervise the num_future latent actions against the
        #     future GT action chunks (actions[:, num_history:]) ---
        if actions is not None and self.action_decoder is not None:
            fut = self.num_history_frames
            gt_actions = actions[:, fut:]
            gt_mask = mask[:, fut:] if mask is not None else None
            ahat = self._forward_action_decoder(
                z_future, tokenized_rgb=self.rgb_pos_embed(future_tokens),
                morphology_index=morphology_index,
            )
            _ck("action_decoder_forward")
            if isinstance(ahat, dict):
                act_loss = self._multi_action_decoding_loss(
                    ahat, gt_actions, loss_type=self.config.get("action_decoding_loss_type", "l1"),
                    loss_mask=gt_mask,
                )
            else:
                act_loss = self._action_decoding_loss(
                    ahat, gt_actions, loss_type=self.config.get("action_decoding_loss_type", "l1"),
                    loss_mask=gt_mask,
                )
            total_loss = total_loss + self.config.get("action_decoding_weight", 0.5) * act_loss
            self.aux_losses["act_decoding_loss"] = act_loss.detach().clone()

        # diagnostic: log magnitudes of each intermediate to find what blows up (gated by DIT_DIAG)
        import os as _os
        if _os.environ.get("DIT_DIAG") and _os.environ.get("RANK", "0") == "0":
            self._fwd_count = getattr(self, "_fwd_count", 0) + 1
            _mx = lambda t: (float(t.abs().max()) if t is not None and t.numel() else -1.0)
            _act = self.aux_losses.get("act_decoding_loss", None)
            if self._fwd_count % 10 == 0 or not torch.isfinite(total_loss):
                print(f"DIAG fwd{self._fwd_count} morph={morphology_index.tolist() if morphology_index is not None else None} "
                      f"|lat|={_mx(latents):.2e} |ref|={_mx(ref):.2e} |zctrl|={_mx(z_control):.2e} "
                      f"|vpred|={_mx(v_pred):.2e} flow={float(flow_loss):.3e} finflow={bool(torch.isfinite(flow_loss))} "
                      f"act={(float(_act) if _act is not None else -1):.3e}", flush=True)

        # non-finite DETECTOR: surface exactly WHERE a NaN/Inf first originates (rgb input /
        # VAE latent / action path / DiT output) + which morphology, so the real bug can be
        # fixed at the source rather than masked. The zero-grad skip only avoids corrupting the
        # weights so we can keep collecting these diagnostics across batches.
        if not torch.isfinite(total_loss):
            import os as _os
            _fin = lambda t: bool(torch.isfinite(t).all())
            _act = self.aux_losses.get("act_decoding_loss", None)
            self._nonfinite_count = getattr(self, "_nonfinite_count", 0) + 1
            print(
                f"[NONFINITE #{self._nonfinite_count}] rank={_os.environ.get('RANK', '0')} "
                f"morph={morphology_index.tolist() if morphology_index is not None else None} "
                f"rgb_nonfinite={int((~torch.isfinite(rgb)).sum())} "
                f"rgb_range=[{float(rgb.min()):.3f},{float(rgb.max()):.3f}] | finite: "
                f"latents={_fin(latents)} z_control={_fin(z_control)} v_pred={_fin(v_pred)} "
                f"flow={_fin(flow_loss)} act={(_fin(_act) if _act is not None else 'NA')}",
                flush=True,
            )
            anchor = next(self.forward_model.action_to_control.parameters())
            total_loss = anchor.sum() * 0.0
        self.aux_losses = {k: torch.nan_to_num(v) for k, v in self.aux_losses.items()}

        return total_loss

    # ----------------------------------------------------------------- sampling
    @torch.no_grad()
    def _sample_future(self, rgb, actions=None, morphology_index=None):
        latents = self._encode_clip(rgb).to(rgb.dtype)
        N, Cl, Fp, h, w = latents.shape
        K = min(self.num_history_latent, Fp)
        ref = torch.zeros_like(latents)
        ref[:, :, :K] = latents[:, :, :K]
        _, z_control, _ = self._latent_actions(rgb, actions, morphology_index, Fp, K)
        z_control = z_control.to(rgb.dtype)
        context = self._build_context(N, rgb.device, rgb.dtype)
        clip_fea = self._build_clip(N, rgb.device, rgb.dtype)

        self.sample_scheduler.set_timesteps(self.viz_num_steps, device=rgb.device)
        x = torch.randn_like(latents)  # flow matching: start from standard normal noise
        for t in self.sample_scheduler.timesteps:
            tt = t.expand(N).to(rgb.device)
            v = self.forward_model(x, tt, z_control, ref, context, clip_fea)
            x = self.sample_scheduler.step(v.float(), t, x.float()).prev_sample.to(rgb.dtype)
        pred_pix = self.rgb_tokenizer.decode_temporal(x, out_hw=(rgb.shape[-2], rgb.shape[-1]))
        gt_pix = self.rgb_tokenizer.decode_temporal(latents, out_hw=(rgb.shape[-2], rgb.shape[-1]))
        return pred_pix, gt_pix  # [N,C,F,H,W] in [-1,1]

    @torch.no_grad()
    def visualize(self, rgb, actions=None, mask=None, **kwargs):
        morphology_index = kwargs.get("morphology_index", None)
        pred_pix, gt_pix = self._sample_future(rgb, actions, morphology_index)
        nf = min(self.num_future_frames, pred_pix.shape[2])
        pred = pred_pix[:, :, -nf:]   # [N,C,nf,H,W]
        gt = gt_pix[:, :, -nf:]
        # side-by-side [gt | pred], to [N, nf, C, H, 2W], range [0,1]
        side = torch.cat([gt, pred], dim=-1)
        side = rearrange(side, "n c f h w -> n f c h w")
        side = torch.clamp(side * 0.5 + 0.5, 0.0, 1.0)
        return side

    # --------------------------------------------------------------- state dict
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd["loss_scheduler"] = self.loss_scheduler.state_dict()
        return sd

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        if "loss_scheduler" in state_dict:
            self.loss_scheduler.load_state_dict(state_dict["loss_scheduler"])
            del state_dict["loss_scheduler"]
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
