from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class AdapterParamCounts:
    a: int
    b: int

    @property
    def total(self) -> int:
        return self.a + self.b


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_linear)}")
        if r <= 0:
            raise ValueError(f"r must be > 0, got {r}")

        self.base = base_linear
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0.0 else nn.Identity()

        d_in = base_linear.in_features
        d_out = base_linear.out_features

        self.lora_A = nn.Parameter(torch.empty((self.r, d_in), dtype=base_linear.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros((d_out, self.r), dtype=base_linear.weight.dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def adapter_param_counts(self) -> AdapterParamCounts:
        a = self.lora_A.numel()
        b = self.lora_B.numel()
        return AdapterParamCounts(a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        x_d = self.dropout(x)
        z = torch.matmul(x_d, self.lora_A.t())
        delta = torch.matmul(z, self.lora_B.t())
        return base_out + (self.scaling * delta)


class GroupLocalLoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        m: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"GroupLocalLoRALinear expects nn.Linear, got {type(base_linear)}")
        if r <= 0:
            raise ValueError(f"r must be > 0, got {r}")
        if m <= 0:
            raise ValueError(f"m must be > 0, got {m}")
        if r % m != 0:
            raise ValueError(f"r % m must be 0, got r={r}, m={m}")

        self.base = base_linear
        self.r = int(r)
        self.m = int(m)
        self.g = self.r // self.m
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0.0 else nn.Identity()

        d_in = base_linear.in_features
        d_out = base_linear.out_features

        if d_out % self.g != 0:
            raise ValueError(f"d_out % g must be 0, got d_out={d_out}, g={self.g} (r={r}, m={m})")
        self.d_block = d_out // self.g

        self.lora_A = nn.Parameter(torch.empty((self.r, d_in), dtype=base_linear.weight.dtype))
        self.lora_B_grouped = nn.Parameter(torch.zeros((self.g, self.d_block, self.m), dtype=base_linear.weight.dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def adapter_param_counts(self) -> AdapterParamCounts:
        a = self.lora_A.numel()
        b = self.lora_B_grouped.numel()
        return AdapterParamCounts(a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        x_d = self.dropout(x)
        z = torch.matmul(x_d, self.lora_A.t())
        z = z.reshape(*z.shape[:-1], self.g, self.m)

        out_blocks = torch.einsum("...gm,gdm->...gd", z, self.lora_B_grouped)
        delta = out_blocks.reshape(*out_blocks.shape[:-2], self.out_features)
        return base_out + (self.scaling * delta)


def iter_adapter_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, GroupLocalLoRALinear)):
            yield name, module


def adapter_trainable_param_count(model: nn.Module) -> int:
    total = 0
    for _, module in iter_adapter_modules(model):
        for p in module.parameters():
            if p.requires_grad:
                total += p.numel()
    return total


def adapter_state_dict(model: nn.Module, device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    keys: set[str] = set()
    for name, module in iter_adapter_modules(model):
        for param_name, _ in module.named_parameters(recurse=False):
            keys.add(f"{name}.{param_name}")

    full = model.state_dict()
    out: Dict[str, torch.Tensor] = {}
    for k in sorted(keys):
        if k not in full:
            continue
        out[k] = full[k].detach().to(device)
    return out


def load_adapter_state_dict(model: nn.Module, state: Dict[str, torch.Tensor], strict: bool = True) -> None:
    expected = set(adapter_state_dict(model).keys())
    provided = set(state.keys())

    missing = expected - provided
    unexpected = provided - expected

    if strict and (missing or unexpected):
        parts = []
        if missing:
            parts.append(f"missing keys: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            parts.append(f"unexpected keys: {sorted(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}")
        raise KeyError("Adapter state_dict mismatch: " + ", ".join(parts))

    model.load_state_dict(state, strict=False)


def count_params(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(int(p.numel()) for p in parameters)

