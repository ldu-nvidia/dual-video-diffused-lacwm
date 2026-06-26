from abc import ABC, abstractmethod

import torch

from robot_wm.inference.actor.base import RobotActionHistory, RobotObsHistory


class WMState:
    def __init__(self, visual_state, proprio_state=None, actions=None):
        self.visual_state = visual_state  # (B, T, P, Z)  (potentially in latents)
        self.proprio_state = proprio_state  # (B, T, Z) -- Optional
        self.actions = actions  # (B, T, A) -- Optional
        self.__post_init__()

    def __post_init__(self):
        if not isinstance(self.visual_state, torch.Tensor):
            raise ValueError("visual_state must be a torch.Tensor instance")
        if len(self.visual_state.shape) != 4:
            raise ValueError(
                "visual_state must have shape (B, T, P, Z) got {}".format(
                    self.visual_state.shape
                )
            )

        if self.proprio_state is not None:
            if not isinstance(self.proprio_state, torch.Tensor):
                raise ValueError("proprio_state must be a torch.Tensor instance")
            if len(self.proprio_state.shape) != 3:
                raise ValueError(
                    "proprio_state must have shape (B, T, Z) got {}".format(
                        self.proprio_state.shape
                    )
                )

            assert (
                self.visual_state.shape[0] == self.proprio_state.shape[0]
            ), "visual_state and proprio_state must have the same batch size"
            assert (
                self.visual_state.shape[1] == self.proprio_state.shape[1]
            ), "visual_state and proprio_state must have the same time dimension"

    def pick_sample(self, idx):
        visual_state = None
        proprio_state = None
        actions = None
        if self.visual_state is not None:
            visual_state = self.visual_state[idx].unsqueeze(0)
        if self.proprio_state is not None:
            proprio_state = self.proprio_state[idx].unsqueeze(0)
        if self.actions is not None:
            actions = self.actions[idx].unsqueeze(0)
        return WMState(visual_state, proprio_state, actions)


class WorldModel(ABC):
    @abstractmethod
    def reset(self):
        pass

    def encode_goal(self, goal) -> WMState:
        encoded_goal = self._encode_goal(goal)
        # assert isinstance(encoded_goal, WMState), "encode_goal must return a WMState instance"
        return encoded_goal

    @abstractmethod
    def _encode_goal(self, goal) -> WMState:
        pass

    def encode_history(
        self, obs_history: RobotObsHistory, action_history: RobotActionHistory
    ) -> WMState:
        # assert isinstance(obs_history, RobotObsHistory)
        # assert isinstance(action_history, RobotActionHistory)
        # assert len(obs_history) - 1 == len(
        #     action_history
        # ), f"We should have 1 less action than obs when encoding history. Got {len(obs_history)} obs and {len(action_history)} actions"
        encoded_history = self._encode_history(obs_history, action_history)
        return encoded_history

    @abstractmethod
    def input_frames_per_prediction(self, input_freq):
        pass

    @abstractmethod
    def _encode_history(
        self, obs_history: RobotObsHistory, action_history: RobotActionHistory
    ) -> WMState:
        pass

    def rollout(
        self, context: WMState, actions_sequence_to_rollout: RobotActionHistory
    ) -> WMState:
        # assert isinstance(context, WMState)
        # assert isinstance(actions_sequence_to_rollout, RobotActionHistory)
        return self._rollout(context, actions_sequence_to_rollout)

    @abstractmethod
    def _rollout(
        self, context: WMState, actions_sequence_to_rollout: RobotActionHistory
    ) -> WMState:
        pass

    def decode_latents(self, state: WMState) -> RobotObsHistory:
        # assert isinstance(state, WMState)
        decoded_latents = self._decode_latents(state)
        # assert isinstance(decoded_latents, RobotObsHistory)
        return decoded_latents

    @abstractmethod
    def _decode_latents(self, state: WMState) -> RobotObsHistory:
        pass
