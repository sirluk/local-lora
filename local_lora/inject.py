from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch
from torch import nn

from .adapters import (
    GroupLocalLoRALinear,
    GroupingMode,
    InputLocalLoRALinear,
    LoRALinear,
    ScalingMode,
    adapter_trainable_param_count,
    count_params,
)

AdapterType = Literal["vanilla_lora", "group_local", "input_local", "bd_lora"]
BDLoRARowFactor = Literal["block_a", "block_b", "dense"]


@dataclass
class InjectedLayerInfo:
    name: str
    adapter_kind: str
    d_in: int
    d_out: int
    adapter_params: int


@dataclass
class InjectionReport:
    adapter_type: str
    r: int
    m: Optional[int]
    n: Optional[int]
    alpha: float
    dropout: float
    scaling_mode: str
    grouping_mode: str
    perm_seed: Optional[int]
    bd_row_factor: Optional[str]
    target_suffixes: Tuple[str, ...]
    injected: List[InjectedLayerInfo]

    @property
    def adapter_params_total(self) -> int:
        return sum(int(x.adapter_params) for x in self.injected)


def _iter_named_linears(model: nn.Module) -> Iterable[Tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def collect_target_linears(model: nn.Module, target_suffixes: Sequence[str]) -> List[Tuple[str, nn.Linear]]:
    target_suffixes = tuple(target_suffixes)
    out: List[Tuple[str, nn.Linear]] = []
    for name, module in _iter_named_linears(model):
        if any(name.endswith(suf) for suf in target_suffixes):
            out.append((name, module))
    return out


def inject_adapters(
    model: nn.Module,
    adapter_type: AdapterType,
    r: int,
    m: Optional[int] = None,
    alpha: float = 1.0,
    dropout: float = 0.0,
    *,
    scaling_mode: ScalingMode = "standard",
    grouping_mode: GroupingMode = "contiguous",
    perm_seed: Optional[int] = None,
    perm_group_by: Literal["suffix", "layer"] = "suffix",
    head_dim: Optional[int] = None,
    target_suffixes: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> InjectionReport:
    injected: List[InjectedLayerInfo] = []
    target_suffixes_t = tuple(target_suffixes)

    for name, linear in collect_target_linears(model, target_suffixes_t):
        matched_suffix = next((suf for suf in target_suffixes_t if name.endswith(suf)), "")
        perm_group_id = matched_suffix if perm_group_by == "suffix" else name

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model

        attention_suffixes = {"q_proj", "k_proj", "v_proj", "o_proj"}

        if adapter_type == "vanilla_lora":
            wrapped = LoRALinear(linear, r=r, alpha=alpha, dropout=dropout, scaling_mode=scaling_mode)
            adapter_params = wrapped.adapter_param_counts().total
            adapter_kind = "vanilla_lora"
        elif adapter_type == "group_local":
            if m is None:
                raise ValueError("m must be provided for group_local adapters")
            if grouping_mode == "head_aligned" and head_dim is not None and matched_suffix in attention_suffixes:
                g = int(r) // int(m)
                d_out = int(linear.out_features)
                if d_out % g != 0:
                    raise ValueError(f"Head-aligned grouping requires d_out % g == 0, got d_out={d_out}, g={g}")
                if (d_out // g) % int(head_dim) != 0:
                    raise ValueError(
                        f"Head-aligned grouping requires output block size divisible by head_dim, got "
                        f"d_out={d_out}, g={g}, head_dim={int(head_dim)}"
                    )
            wrapped = GroupLocalLoRALinear(
                linear,
                r=r,
                m=m,
                alpha=alpha,
                dropout=dropout,
                scaling_mode=scaling_mode,
                grouping_mode=grouping_mode,
                perm_seed=perm_seed,
                perm_group_id=perm_group_id,
            )
            adapter_params = wrapped.adapter_param_counts().total
            adapter_kind = "group_local"
        elif adapter_type == "input_local":
            if m is None:
                raise ValueError("m must be provided for input_local adapters")
            if grouping_mode == "head_aligned" and head_dim is not None and matched_suffix == "o_proj":
                g = int(r) // int(m)
                d_in = int(linear.in_features)
                if d_in % g != 0:
                    raise ValueError(f"Head-aligned grouping requires d_in % g == 0, got d_in={d_in}, g={g}")
                if (d_in // g) % int(head_dim) != 0:
                    raise ValueError(
                        f"Head-aligned grouping requires input block size divisible by head_dim, got "
                        f"d_in={d_in}, g={g}, head_dim={int(head_dim)}"
                    )
            wrapped = InputLocalLoRALinear(
                linear,
                r=r,
                m=m,
                alpha=alpha,
                dropout=dropout,
                scaling_mode=scaling_mode,
                grouping_mode=grouping_mode,
                perm_seed=perm_seed,
                perm_group_id=perm_group_id,
            )
            adapter_params = wrapped.adapter_param_counts().total
            adapter_kind = "input_local"
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

        setattr(parent, child_name, wrapped)
        injected.append(
            InjectedLayerInfo(
                name=name,
                adapter_kind=adapter_kind,
                d_in=linear.in_features,
                d_out=linear.out_features,
                adapter_params=int(adapter_params),
            )
        )

    return InjectionReport(
        adapter_type=adapter_type,
        r=int(r),
        m=int(m) if m is not None else None,
        n=None,
        alpha=float(alpha),
        dropout=float(dropout),
        scaling_mode=str(scaling_mode),
        grouping_mode=str(grouping_mode),
        perm_seed=int(perm_seed) if perm_seed is not None else None,
        bd_row_factor=None,
        target_suffixes=target_suffixes_t,
        injected=injected,
    )


def inject_bd_lora(
    model: nn.Module,
    *,
    r: int,
    n: int,
    alpha: float,
    dropout: float = 0.0,
    scaling_mode: ScalingMode = "standard",
    grouping_mode: GroupingMode = "contiguous",
    perm_seed: Optional[int] = None,
    bd_row_factor: BDLoRARowFactor = "block_a",
    target_suffixes: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    perm_group_by: Literal["suffix", "layer"] = "suffix",
    head_dim: Optional[int] = None,
) -> InjectionReport:
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if r <= 0:
        raise ValueError(f"r must be > 0, got {r}")
    if r % int(n) != 0:
        raise ValueError(f"r % n must be 0 for BD-LoRA, got r={r}, n={n}")

    injected: List[InjectedLayerInfo] = []
    target_suffixes_t = tuple(target_suffixes)
    m = int(r) // int(n)

    row_suffixes = {"o_proj", "down_proj"}
    attention_suffixes = {"q_proj", "k_proj", "v_proj", "o_proj"}

    for name, linear in collect_target_linears(model, target_suffixes_t):
        matched_suffix = next((suf for suf in target_suffixes_t if name.endswith(suf)), "")
        perm_group_id = matched_suffix if perm_group_by == "suffix" else name

        is_row = matched_suffix in row_suffixes

        if is_row:
            if bd_row_factor == "block_a":
                if grouping_mode == "head_aligned" and head_dim is not None and matched_suffix == "o_proj":
                    d_in = int(linear.in_features)
                    if d_in % int(n) != 0:
                        raise ValueError(f"Head-aligned BD-LoRA requires d_in % n == 0, got d_in={d_in}, n={int(n)}")
                    if (d_in // int(n)) % int(head_dim) != 0:
                        raise ValueError(
                            f"Head-aligned BD-LoRA requires input block size divisible by head_dim, got "
                            f"d_in={d_in}, n={int(n)}, head_dim={int(head_dim)}"
                        )
                wrapped = InputLocalLoRALinear(
                    linear,
                    r=r,
                    m=m,
                    alpha=alpha,
                    dropout=dropout,
                    scaling_mode=scaling_mode,
                    grouping_mode=grouping_mode,
                    perm_seed=perm_seed,
                    perm_group_id=perm_group_id,
                )
                adapter_kind = "bd_row_block_a"
            elif bd_row_factor == "block_b":
                if grouping_mode == "head_aligned" and head_dim is not None and matched_suffix == "o_proj":
                    d_out = int(linear.out_features)
                    if d_out % int(n) != 0:
                        raise ValueError(
                            f"Head-aligned BD-LoRA requires d_out % n == 0, got d_out={d_out}, n={int(n)}"
                        )
                    if (d_out // int(n)) % int(head_dim) != 0:
                        raise ValueError(
                            f"Head-aligned BD-LoRA requires output block size divisible by head_dim, got "
                            f"d_out={d_out}, n={int(n)}, head_dim={int(head_dim)}"
                        )
                wrapped = GroupLocalLoRALinear(
                    linear,
                    r=r,
                    m=m,
                    alpha=alpha,
                    dropout=dropout,
                    scaling_mode=scaling_mode,
                    grouping_mode=grouping_mode,
                    perm_seed=perm_seed,
                    perm_group_id=perm_group_id,
                )
                adapter_kind = "bd_row_block_b"
            elif bd_row_factor == "dense":
                wrapped = LoRALinear(linear, r=r, alpha=alpha, dropout=dropout, scaling_mode=scaling_mode)
                adapter_kind = "bd_row_dense"
            else:
                raise ValueError(f"Unknown bd_row_factor: {bd_row_factor}")
        else:
            # Column-parallel-like (q/k/v + MLP up/gate): block-diagonal B.
            if grouping_mode == "head_aligned" and head_dim is not None and matched_suffix in attention_suffixes:
                d_out = int(linear.out_features)
                if d_out % int(n) != 0:
                    raise ValueError(f"Head-aligned BD-LoRA requires d_out % n == 0, got d_out={d_out}, n={int(n)}")
                if (d_out // int(n)) % int(head_dim) != 0:
                    raise ValueError(
                        f"Head-aligned BD-LoRA requires output block size divisible by head_dim, got "
                        f"d_out={d_out}, n={int(n)}, head_dim={int(head_dim)}"
                    )
            wrapped = GroupLocalLoRALinear(
                linear,
                r=r,
                m=m,
                alpha=alpha,
                dropout=dropout,
                scaling_mode=scaling_mode,
                grouping_mode=grouping_mode,
                perm_seed=perm_seed,
                perm_group_id=perm_group_id,
            )
            adapter_kind = "bd_col_block_b"

        adapter_params = wrapped.adapter_param_counts().total

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, wrapped)
        injected.append(
            InjectedLayerInfo(
                name=name,
                adapter_kind=adapter_kind,
                d_in=linear.in_features,
                d_out=linear.out_features,
                adapter_params=int(adapter_params),
            )
        )

    return InjectionReport(
        adapter_type="bd_lora",
        r=int(r),
        m=int(m),
        n=int(n),
        alpha=float(alpha),
        dropout=float(dropout),
        scaling_mode=str(scaling_mode),
        grouping_mode=str(grouping_mode),
        perm_seed=int(perm_seed) if perm_seed is not None else None,
        bd_row_factor=str(bd_row_factor),
        target_suffixes=target_suffixes_t,
        injected=injected,
    )


def freeze_model_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_adapter_params(model: nn.Module) -> None:
    for _, module in model.named_modules():
        if isinstance(module, (LoRALinear, GroupLocalLoRALinear, InputLocalLoRALinear)):
            for p in module.parameters():
                p.requires_grad = True


def unfreeze_sequence_classification_head(model: nn.Module) -> None:
    # Common heads: LlamaForSequenceClassification.score, or generic "classifier".
    for attr in ("score", "classifier", "classification_head"):
        head = getattr(model, attr, None)
        if isinstance(head, nn.Module):
            for p in head.parameters():
                p.requires_grad = True


def trainable_param_summary(model: nn.Module) -> Dict[str, int]:
    return {
        "trainable_params": count_params(p for p in model.parameters() if p.requires_grad),
        "adapter_trainable_params": adapter_trainable_param_count(model),
    }
