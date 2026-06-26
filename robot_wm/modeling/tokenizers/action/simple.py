import torch
import torch.nn as nn

from robot_wm.modeling.modules.attention import precompute_freqs_cis
from robot_wm.modeling.modules.blocks import AttentionBlock
from robot_wm.modeling.tokenizers.action.base import ActionTokenizer


class SimpleActionTokenizer(ActionTokenizer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        chunk_decoded_shape: tuple[int, ...] = None,
        chunk_encoded_shape: tuple[int, ...] = None,
    ):
        super().__init__(
            chunk_decoded_shape=chunk_decoded_shape,
            chunk_encoded_shape=chunk_encoded_shape,
        )
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._layer = nn.Linear(input_dim, output_dim)

    @property
    def dim(self) -> int:
        return self._output_dim

    def init_weights(self):
        nn.init.trunc_normal_(self._layer.weight, mean=0.0, std=0.02)

    def forward(self, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            actions: torch.Tensor [N, T, *, D]
        Returns:
            torch.Tensor [N, T, *, d]
        """
        return self._layer(actions)


class SequenceActionTokenizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        num_layers: int,
        rope_end: int,
        rope_theta: int,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        mlp_drop_prob: float = 0.0,
        residual_drop_prob: float = 0.0,
        drop_path_rate: float = 0.0,
        build_norm=nn.RMSNorm,
    ):
        super().__init__()
        self.x_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.proj_dropout = nn.Dropout(p=proj_drop_prob)
        self.layers = nn.ModuleList(
            [
                AttentionBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    attn_drop_prob=attn_drop_prob,
                    proj_drop_prob=proj_drop_prob,
                    mlp_drop_prob=mlp_drop_prob,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = build_norm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        self.register_buffer(
            name="freqs_cis",
            tensor=precompute_freqs_cis(
                dim=hidden_dim // num_heads, end=rope_end, theta=rope_theta
            ),
            persistent=False,
        )

        self.cls_tok = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self._output_dim = output_dim

    @property
    def dim(self) -> int:
        return self._output_dim

    def init_weights(self):
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.x_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.output_proj.weight, mean=0.0, std=0.02)
        for idx, layer in enumerate(self.layers):
            init_std = 0.02 / (2 * (idx + 1)) ** 0.5
            layer.init_weights(init_std=init_std)
        nn.init.uniform_(self.cls_tok, a=-1, b=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [N, T,  D]
        Returns:
            torch.Tensor [N, 1 + T, D]
        """
        x = self.x_proj(x)
        x = torch.cat([self.cls_tok.expand(x.shape[0], -1, -1), x], 1)
        x = self.proj_dropout(x)

        for layer in self.layers:
            x = layer(x, self.freqs_cis)

        x = self.norm(x)
        x = self.output_proj(x)
        x = self.proj_dropout(x)
        return x


if __name__ == "__main__":
    # N, T, A, input_dim = 5, 9, 5, 7
    # actions = torch.randn(N, T, A, input_dim)
    # actions = actions.to("cuda").to(torch.bfloat16)
    # print(f"{actions.shape = }")

    # output_dim = 16
    # tokenizer = SimpleActionTokenizer(input_dim, output_dim)
    # tokenizer = tokenizer.to("cuda").to(torch.bfloat16)
    # tokenizer.init_weights()
    # tokenizer_size = sum(p.numel() for p in tokenizer.parameters())
    # print(f"{tokenizer_size = }")

    # a1 = tokenizer(actions)
    # print(f"{a1.shape = }")

    # write debug code for multi simple action tokenizer
    input_dim = {"Agios": 67, "Droid": 10}
    output_dim = 16
    tokenizer = SimpleActionTokenizer(input_dim, output_dim)
    tokenizer = tokenizer.to("cuda").to(torch.bfloat16)
    tokenizer.init_weights()

    N, T, A, input_dim = 5, 9, 5, 67
    actions = torch.randn(N, T, A, input_dim)
    actions = actions.to("cuda").to(torch.bfloat16)
    print(f"{actions.shape = }")
    a1 = tokenizer(actions)
    print(f"{a1.shape = }")
