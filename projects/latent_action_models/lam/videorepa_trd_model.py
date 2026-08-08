"""VPM continuation with training-only VideoREPA relational supervision.

This is deliberately not a second denoising branch.  The ordinary VPM Wan
forward runs once, a scoped hook observes the output of block 14, and a clean
cached V-JEPA target supervises only the pairwise relations of pooled tokens.
The hook is installed only by ``forward`` (training/validation); deployment
sampling calls the ordinary Wan video trunk directly and has no TRD target,
projection, feature branch, or additional parameter.
"""

from __future__ import annotations

import logging
import math
from typing import Mapping

import torch

from lam.dual_explicit_action_dit_model import DualExplicitActionDiTModel
from robot_wm.modeling.dual_diffusion.token_relation_distillation import (
    pool_per_view_tokens,
    token_relation_loss,
    wan_tokens_to_grid,
)

logger = logging.getLogger(__name__)


class VideoRepaTRDModel(DualExplicitActionDiTModel):
    """Parameter-identical VPM model with an optional training-only TRD loss."""

    def __init__(self, *, token_relation_distillation: Mapping, **kwargs):
        config = dict(token_relation_distillation)
        super().__init__(**kwargs)
        self.trd_mode = str(config.get("mode", "off"))
        self.trd_loss_weight = float(config.get("loss_weight", 0.0))
        self.trd_margin = float(config.get("margin", 0.05))
        self.trd_block_index = int(config.get("block_index", 14))
        self.trd_num_views = int(config.get("num_views", 3))
        self.trd_pool_height = int(config.get("pool_height", 2))
        self.trd_pool_width_per_view = int(
            config.get("pool_width_per_view", 4)
        )
        self.trd_exclude_first_temporal_bin = bool(
            config.get("exclude_first_temporal_bin", True)
        )
        self._trd_forward_hook_installations = 0

        if self.trd_mode not in {"off", "on"}:
            raise ValueError("TRD mode must be 'off' or 'on'")
        if self.trd_mode == "off" and self.trd_loss_weight != 0.0:
            raise ValueError("TRD-OFF requires an exact zero loss weight")
        if self.trd_mode == "on" and self.trd_loss_weight <= 0.0:
            raise ValueError("TRD-ON requires a positive loss weight")
        if not 0.0 <= self.trd_margin < 2.0:
            raise ValueError("TRD margin must lie in [0,2)")
        if self.trd_block_index != 14:
            raise ValueError("the prospective screen freezes Wan block index 14")
        if (
            self.trd_num_views != 3
            or self.trd_pool_height != 2
            or self.trd_pool_width_per_view != 4
            or not self.trd_exclude_first_temporal_bin
        ):
            raise ValueError(
                "the prospective TRD schema is frozen to three views, 2x4 "
                "spatial pooling, and first-temporal-bin exclusion"
            )
        blocks = getattr(self.forward_model.transformer, "blocks", None)
        if blocks is None or len(blocks) != 30:
            raise RuntimeError("TRD screen requires the pinned 30-block Wan model")
        if bool(
            getattr(self.forward_model.transformer, "gradient_checkpointing", False)
        ):
            raise RuntimeError(
                "TRD block capture requires gradient_checkpointing=false so "
                "backward differentiates the exact captured forward"
            )
        # The exact VPM parent keeps the parameter-matched auxiliary schema but
        # makes it a video-path no-op.  TRD must not accidentally reactivate it.
        if (
            not self.parameter_matched_control
            or self.condition_on_tf
            or self.condition_on_tf_clock
            or self.tf_loss_weight != 0.0
        ):
            raise RuntimeError(
                "TRD must continue the exact parameter-matched, video-only VPM path"
            )
        logger.info(
            "VideoRepaTRDModel: mode=%s weight=%.3f margin=%.3f block=%d "
            "pooled_tokens=%d",
            self.trd_mode,
            self.trd_loss_weight,
            self.trd_margin,
            self.trd_block_index,
            self.trd_num_views
            * 3
            * self.trd_pool_height
            * self.trd_pool_width_per_view,
        )

    def _trd_loss(
        self,
        hidden_tokens: torch.Tensor,
        clean_target: torch.Tensor,
    ):
        if clean_target.ndim != 5 or tuple(clean_target.shape[1:]) != (
            64,
            4,
            24,
            120,
        ):
            raise RuntimeError(
                "TRD requires cached V-JEPA targets [B,64,4,24,120]"
            )
        if hidden_tokens.shape[0] != clean_target.shape[0]:
            raise RuntimeError("TRD student/teacher batch sizes differ")
        patch_size = tuple(
            int(value) for value in self.forward_model.transformer.patch_size
        )
        student_grid = wan_tokens_to_grid(
            hidden_tokens,
            target_grid_shape=tuple(int(value) for value in clean_target.shape[2:]),
            patch_size=patch_size,
        )
        student = pool_per_view_tokens(
            student_grid,
            num_views=self.trd_num_views,
            pooled_height=self.trd_pool_height,
            pooled_width_per_view=self.trd_pool_width_per_view,
        )
        teacher = pool_per_view_tokens(
            clean_target.detach(),
            num_views=self.trd_num_views,
            pooled_height=self.trd_pool_height,
            pooled_width_per_view=self.trd_pool_width_per_view,
        )
        # VideoREPA excludes the first 3D-VAE bin because it primarily anchors
        # static semantics; use the remaining three bins for spatial/dynamic
        # relations in both representations.
        student = student[:, :, 1:]
        teacher = teacher[:, :, 1:]
        if student.shape[:4] != teacher.shape[:4]:
            raise RuntimeError(
                "pooled Wan/V-JEPA lattices differ: "
                f"{tuple(student.shape)} vs {tuple(teacher.shape)}"
            )
        relation = token_relation_loss(
            student,
            teacher,
            margin=self.trd_margin,
        )
        return relation, student, teacher

    def forward(
        self,
        rgb: torch.Tensor,
        actions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        clean_target = kwargs.get("auxiliary_target")
        if clean_target is None:
            raise RuntimeError(
                "TRD training/validation requires the offline clean V-JEPA target"
            )
        blocks = self.forward_model.transformer.blocks
        captured: list[torch.Tensor] = []

        def capture_midpoint(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise RuntimeError("Wan block 14 did not return [B,N,D] tokens")
            captured.append(output)

        self._trd_forward_hook_installations += 1
        handle = blocks[self.trd_block_index].register_forward_hook(capture_midpoint)
        try:
            base_loss = super().forward(rgb, actions, mask, **kwargs)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                f"expected one Wan block-14 capture, observed {len(captured)}"
            )

        # The control computes identical telemetry but detaches before the
        # relation calculation, so its returned objective is exactly the VPM
        # video objective and has no zero-weight auxiliary graph.
        student_hidden = captured[0]
        if self.trd_mode == "off":
            with torch.no_grad():
                relation, student, teacher = self._trd_loss(
                    student_hidden.detach(), clean_target
                )
            total_loss = base_loss
        else:
            relation, student, teacher = self._trd_loss(
                student_hidden, clean_target
            )
            total_loss = base_loss + self.trd_loss_weight * relation.total.mean()

        spatial = relation.spatial.mean()
        temporal = relation.temporal.mean()
        trd_total = relation.total.mean()
        self.aux_losses["trd/spatial_relation_l1"] = spatial.detach()
        self.aux_losses["trd/temporal_relation_l1"] = temporal.detach()
        self.aux_losses["trd/relation_l1"] = trd_total.detach()
        self.aux_losses["trd/weighted_loss"] = (
            trd_total.detach() * self.trd_loss_weight
        )
        self.aux_losses["trd/enabled"] = base_loss.new_tensor(
            float(self.trd_mode == "on")
        )
        self.aux_losses["trd/loss_weight"] = base_loss.new_tensor(
            self.trd_loss_weight
        )
        self.aux_losses["trd/margin"] = base_loss.new_tensor(self.trd_margin)
        self.aux_losses["trd/block_index"] = base_loss.new_tensor(
            float(self.trd_block_index)
        )
        self.aux_losses["trd/pooled_token_count"] = base_loss.new_tensor(
            float(student.shape[1] * student.shape[2] * student.shape[3])
        )
        self.aux_losses["trd/student_hidden_rms"] = (
            student.detach().float().square().mean().sqrt()
        )
        self.aux_losses["trd/teacher_target_rms"] = (
            teacher.detach().float().square().mean().sqrt()
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("combined VPM/TRD loss is non-finite")
        return total_loss

    def _video_only_wan(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        z_control: torch.Tensor,
        reference: torch.Tensor,
        context,
        clip_fea,
    ) -> torch.Tensor:
        """Execute only the VPM video trunk; never construct an auxiliary state."""

        forward_model = self.forward_model
        batch, _channels, frames, height, width = noisy_latents.shape
        control = forward_model.action_to_control(
            z_control, height, width
        ).to(noisy_latents.dtype)
        y = torch.cat([control, reference], dim=1)
        patch_size = tuple(int(value) for value in forward_model.patch_size)
        seq_len = int(
            math.ceil(height * width / (patch_size[1] * patch_size[2]))
            * frames
        )
        output = forward_model.transformer(
            x=noisy_latents,
            t=timesteps,
            context=context,
            seq_len=seq_len,
            y=y,
            clip_fea=clip_fea,
        )
        # This is the same call signature used by the ordinary, non-dual
        # WanForwardModel.  The parameter-matched parent instead supplies an
        # exact-zero ``y_camera`` through an Identity adapter; omitting that
        # additive zero is the algebraic video-only path and avoids executing
        # the dormant adapter/clock/head altogether.
        velocity = output[0] if isinstance(output, (list, tuple)) else output
        if velocity.shape != noisy_latents.shape or velocity.shape[0] != batch:
            raise RuntimeError("video-only Wan velocity shape differs")
        return velocity

    @torch.no_grad()
    def sample_future_deployable(
        self,
        history_rgb,
        actions,
        morphology_index=None,
        *,
        collect_artifacts: bool = False,
        sample_ids: torch.Tensor | None = None,
    ):
        """Generate through a true video-only path with zero TRD/V-JEPA branch.

        The exact VPM checkpoint retains dormant parameter-matching modules in
        its state dict.  This deployment override does not execute or read any
        of them: no auxiliary tensor is allocated, no TF/V-JEPA projection or
        velocity head runs, and no block-14 TRD hook is installed.
        """

        if history_rgb.ndim != 5 or history_rgb.shape[1] != self.num_history_frames:
            raise ValueError("deployable TRD sampler accepts exactly observed history")
        if tuple(self.evaluation_condition_sources) != ("off",):
            raise ValueError("video-only TRD sampler requires condition source 'off'")
        if sample_ids is None or sample_ids.reshape(-1).numel() != history_rgb.shape[0]:
            raise ValueError("video-only TRD sampling requires one immutable clip ID")
        if not self.evaluation_nfe_steps or any(
            int(value) < 1 for value in self.evaluation_nfe_steps
        ):
            raise ValueError("evaluation NFE grid must contain positive integers")
        batch = history_rgb.shape[0]
        history_latents = self._encode_clip(history_rgb).to(history_rgb.dtype)
        if history_latents.shape[2] != self.num_history_latent:
            raise RuntimeError("history-only VAE latent length differs")
        latent_frames = self.rgb_tokenizer.latent_temporal_len(
            self.num_history_frames + self.num_future_frames
        )
        history_frames = history_latents.shape[2]
        video_shape = (
            batch,
            history_latents.shape[1],
            latent_frames,
            history_latents.shape[3],
            history_latents.shape[4],
        )
        reference = history_latents.new_zeros(video_shape)
        reference[:, :, :history_frames] = history_latents
        _, z_control, _ = self._latent_actions(
            history_rgb,
            actions,
            morphology_index,
            latent_frames,
            history_frames,
        )
        z_control = z_control.to(history_rgb.dtype)
        context = self._build_context(batch, history_rgb.device, history_rgb.dtype)
        clip_fea = self._build_clip(batch, history_rgb.device, history_rgb.dtype)
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        initial = self._evaluation_noise(
            video_shape,
            device=history_rgb.device,
            dtype=history_rgb.dtype,
            base_seed=self.evaluation_noise_seed,
            sample_ids=sample_ids,
            stream=0,
            rank=rank,
        )
        artifact_limit = getattr(self, "artifact_batch_limit", 1)
        evidence_count = (
            batch if artifact_limit is None else min(batch, int(artifact_limit))
        )
        evidence = slice(0, evidence_count)
        artifacts = None
        if collect_artifacts:
            artifacts = {
                "video_initial_state": initial[evidence].detach().cpu().to(torch.float16),
                "reference_latents": reference[evidence].detach().cpu().to(torch.float16),
                "deployment_mode": torch.tensor([1], dtype=torch.int64),
                "history_latent_frames": torch.tensor([history_frames]),
                "auxiliary_clean_available": torch.tensor([0], dtype=torch.int64),
                "online_teacher_call_count": torch.tensor([0], dtype=torch.int64),
                "trd_inference_branch_call_count": torch.tensor([0], dtype=torch.int64),
                "sample_ids": sample_ids[evidence].detach().cpu().to(torch.int64),
            }
        primary = None
        call_counts = {}
        for nfe in self.evaluation_nfe_steps:
            nfe = int(nfe)
            self.sample_scheduler.set_timesteps(nfe, device=history_rgb.device)
            state = initial.clone()
            sigmas = self.sample_scheduler.sigmas.to(
                device=state.device, dtype=state.dtype
            )
            calls = 0
            for index, timestep in enumerate(self.sample_scheduler.timesteps):
                expanded_timestep = timestep.expand(batch).to(history_rgb.device)
                velocity = self._video_only_wan(
                    state,
                    expanded_timestep,
                    z_control,
                    reference,
                    context,
                    clip_fea,
                )
                state = self.sample_scheduler.step(
                    velocity.float(), timestep, state.float()
                ).prev_sample.to(history_rgb.dtype)
                next_sigma = sigmas[index + 1]
                state[:, :, :history_frames] = (
                    (1.0 - next_sigma) * reference[:, :, :history_frames]
                    + next_sigma * initial[:, :, :history_frames]
                )
                calls += 1
            if calls != nfe:
                raise RuntimeError("video-only sampler Wan-call count differs")
            decoded = self.rgb_tokenizer.decode_temporal(
                state, out_hw=(history_rgb.shape[-2], history_rgb.shape[-1])
            )
            future = decoded[:, :, -self.num_future_frames :]
            call_counts[f"off:nfe_{nfe}"] = calls
            if artifacts is not None:
                artifacts[f"wan_call_count_off_nfe_{nfe}"] = torch.tensor(
                    [calls], dtype=torch.int64
                )
                artifacts[f"video_final_off_nfe_{nfe}"] = (
                    state[evidence].detach().cpu().to(torch.float16)
                )
                artifacts[f"decoded_future_off_nfe_{nfe}"] = (
                    ((future[evidence].float().clamp(-1.0, 1.0) + 1.0) * 127.5)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                )
            if nfe == self.viz_num_steps:
                primary = future
        if primary is None:
            raise RuntimeError("viz_num_steps was not evaluated")
        self._last_sampling_counters = {
            "wan_calls_by_source_nfe": call_counts,
            "wan_calls_total": sum(call_counts.values()),
            "online_teacher_calls": 0,
            "trd_inference_branch_calls": 0,
            "auxiliary_branch_calls": 0,
            "deployment_mode": 1,
        }
        if artifacts is not None:
            self._visualization_artifacts = artifacts
        return primary

    @torch.no_grad()
    def visualize(self, rgb, actions=None, mask=None, **kwargs):
        """Render monitoring videos through the same target-free sampler."""

        del mask
        # Clone guarantees the sampler cannot retain a view backed by future
        # RGB. Ground truth is read only after target-free generation returns.
        history = rgb[:, : self.num_history_frames].clone(
            memory_format=torch.contiguous_format
        )
        predicted = self.sample_future_deployable(
            history,
            actions,
            kwargs.get("morphology_index"),
            collect_artifacts=True,
            sample_ids=kwargs.get("clip_index"),
        )
        target = rgb.permute(0, 2, 1, 3, 4)
        future_frames = min(
            self.num_future_frames,
            predicted.shape[2],
            target.shape[2],
        )
        side_by_side = torch.cat(
            [
                target[:, :, -future_frames:],
                predicted[:, :, -future_frames:],
            ],
            dim=-1,
        )
        return torch.clamp(
            side_by_side.permute(0, 2, 1, 3, 4) * 0.5 + 0.5,
            0.0,
            1.0,
        )
