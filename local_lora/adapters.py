from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Literal, Optional, Tuple

import torch
from torch import nn


ScalingMode = Literal["standard", "rs"]
GroupingMode = Literal["contiguous", "random", "head_aligned"]


@dataclass(frozen=True)
class AdapterParamCounts:
    a: int
    b: int

    @property
    def total(self) -> int:
        return self.a + self.b


def _compute_scaling(alpha: float, r: int, scaling_mode: ScalingMode) -> float:
    if scaling_mode == "standard":
        return float(alpha) / float(r)
    if scaling_mode == "rs":
        # Rank-stabilized LoRA scaling (used when comparing different ranks).
        return float(alpha) / math.sqrt(float(r))
    raise ValueError(f"Unknown scaling_mode: {scaling_mode!r}")


def _make_deterministic_permutation(length: int, *, seed: int, salt: str) -> torch.Tensor:
    payload = f"{seed}:{salt}:{length}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    derived = int.from_bytes(digest[:8], "big", signed=False) % (2**63 - 1)
    gen = torch.Generator()
    gen.manual_seed(int(derived))
    return torch.randperm(int(length), generator=gen)


def _invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(int(perm.numel()), dtype=perm.dtype)
    return inv


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        *,
        scaling_mode: ScalingMode = "standard",
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_linear)}")
        if r <= 0:
            raise ValueError(f"r must be > 0, got {r}")

        self.base = base_linear
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling_mode: ScalingMode = scaling_mode
        self.scaling = _compute_scaling(alpha=self.alpha, r=self.r, scaling_mode=self.scaling_mode)
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
        *,
        scaling_mode: ScalingMode = "standard",
        grouping_mode: GroupingMode = "contiguous",
        perm_seed: Optional[int] = None,
        perm_group_id: Optional[str] = None,
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
        self.scaling_mode: ScalingMode = scaling_mode
        self.scaling = _compute_scaling(alpha=self.alpha, r=self.r, scaling_mode=self.scaling_mode)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0.0 else nn.Identity()
        self.grouping_mode: GroupingMode = grouping_mode

        d_in = base_linear.in_features
        d_out = base_linear.out_features

        if self.grouping_mode == "random":
            if perm_seed is None:
                raise ValueError("perm_seed must be provided when grouping_mode='random'")
            if not perm_group_id:
                raise ValueError("perm_group_id must be provided when grouping_mode='random'")
            perm_out = _make_deterministic_permutation(d_out, seed=int(perm_seed), salt=f"out:{perm_group_id}")
            inv_perm_out = _invert_permutation(perm_out)
        else:
            perm_out = None
            inv_perm_out = None

        self.register_buffer("perm_out", perm_out, persistent=False)
        self.register_buffer("inv_perm_out", inv_perm_out, persistent=False)

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
        if self.inv_perm_out is not None:
            delta = delta.index_select(-1, self.inv_perm_out)
        return base_out + (self.scaling * delta)


class InputLocalLoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        m: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        *,
        scaling_mode: ScalingMode = "standard",
        grouping_mode: GroupingMode = "contiguous",
        perm_seed: Optional[int] = None,
        perm_group_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"InputLocalLoRALinear expects nn.Linear, got {type(base_linear)}")
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
        self.scaling_mode: ScalingMode = scaling_mode
        self.scaling = _compute_scaling(alpha=self.alpha, r=self.r, scaling_mode=self.scaling_mode)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0.0 else nn.Identity()
        self.grouping_mode: GroupingMode = grouping_mode

        d_in = base_linear.in_features
        d_out = base_linear.out_features

        if d_in % self.g != 0:
            raise ValueError(f"d_in % g must be 0, got d_in={d_in}, g={self.g} (r={r}, m={m})")
        self.d_in_block = d_in // self.g

        if self.grouping_mode == "random":
            if perm_seed is None:
                raise ValueError("perm_seed must be provided when grouping_mode='random'")
            if not perm_group_id:
                raise ValueError("perm_group_id must be provided when grouping_mode='random'")
            perm_in = _make_deterministic_permutation(d_in, seed=int(perm_seed), salt=f"in:{perm_group_id}")
        else:
            perm_in = None

        self.register_buffer("perm_in", perm_in, persistent=False)

        self.lora_A_grouped = nn.Parameter(
            torch.empty((self.g, self.m, self.d_in_block), dtype=base_linear.weight.dtype)
        )
        self.lora_B = nn.Parameter(torch.zeros((d_out, self.r), dtype=base_linear.weight.dtype))

        nn.init.kaiming_uniform_(self.lora_A_grouped, a=5**0.5)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def adapter_param_counts(self) -> AdapterParamCounts:
        a = self.lora_A_grouped.numel()
        b = self.lora_B.numel()
        return AdapterParamCounts(a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        x_d = self.dropout(x)
        if self.perm_in is not None:
            x_d = x_d.index_select(-1, self.perm_in)
        x_blocks = x_d.reshape(*x_d.shape[:-1], self.g, self.d_in_block)

        z_blocks = torch.einsum("...gd,gmd->...gm", x_blocks, self.lora_A_grouped)
        z = z_blocks.reshape(*z_blocks.shape[:-2], self.r)
        delta = torch.matmul(z, self.lora_B.t())
        return base_out + (self.scaling * delta)


def iter_adapter_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, GroupLocalLoRALinear, InputLocalLoRALinear)):
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
