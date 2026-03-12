from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch
from torch import nn

from .adapters import GroupLocalLoRALinear, LoRALinear, adapter_trainable_param_count, count_params

AdapterType = Literal["vanilla_lora", "group_local"]


@dataclass
class InjectedLayerInfo:
    name: str
    d_in: int
    d_out: int
    adapter_params: int


@dataclass
class InjectionReport:
    adapter_type: str
    r: int
    m: Optional[int]
    alpha: float
    dropout: float
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
    target_suffixes: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> InjectionReport:
    injected: List[InjectedLayerInfo] = []
    target_suffixes_t = tuple(target_suffixes)

    for name, linear in collect_target_linears(model, target_suffixes_t):
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model

        if adapter_type == "vanilla_lora":
            wrapped = LoRALinear(linear, r=r, alpha=alpha, dropout=dropout)
            adapter_params = wrapped.adapter_param_counts().total
        elif adapter_type == "group_local":
            if m is None:
                raise ValueError("m must be provided for group_local adapters")
            wrapped = GroupLocalLoRALinear(linear, r=r, m=m, alpha=alpha, dropout=dropout)
            adapter_params = wrapped.adapter_param_counts().total
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

        setattr(parent, child_name, wrapped)
        injected.append(
            InjectedLayerInfo(
                name=name,
                d_in=linear.in_features,
                d_out=linear.out_features,
                adapter_params=int(adapter_params),
            )
        )

    return InjectionReport(
        adapter_type=adapter_type,
        r=int(r),
        m=int(m) if m is not None else None,
        alpha=float(alpha),
        dropout=float(dropout),
        target_suffixes=target_suffixes_t,
        injected=injected,
    )


def freeze_model_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_adapter_params(model: nn.Module) -> None:
    for _, module in model.named_modules():
        if isinstance(module, (LoRALinear, GroupLocalLoRALinear)):
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

