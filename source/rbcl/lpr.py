"""Layerwise Proximal Replay plugin from the public LPR implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from avalanche.core import SupervisedPlugin


@dataclass
class _Preconditioner:
    matrix: torch.Tensor
    parallel: bool = False


class LPRPlugin(SupervisedPlugin):
    """Precondition ER-ACE gradients with replay-buffer activation statistics."""

    def __init__(self, *, storage_policy=None, omega_0=1.0, beta=1.0,
                 every_iter=None, n_data=None, batch_size=100,
                 conv_beta=None, bn_beta=None):
        super().__init__()
        if omega_0 <= 0 or beta < 0 or batch_size <= 0:
            raise ValueError("invalid LPR hyperparameters")
        if every_iter is not None and every_iter <= 0:
            raise ValueError("every_iter must be positive")
        self.omega_0 = float(omega_0)
        self.beta = float(beta)
        self.every_iter = every_iter
        self.n_data = n_data
        self.batch_size = int(batch_size)
        self.conv_beta = conv_beta
        self.bn_beta = bn_beta
        self.storage_policy = storage_policy
        self._preconditioners: Dict[str, _Preconditioner] = {}
        self.update_count = 0
        self.preconditioned_update_count = 0
        self.last_buffer_size = 0

    @torch.no_grad()
    def before_backward(self, strategy, **kwargs):
        storage = self.storage_policy or getattr(strategy, "storage_policy", None)
        if storage is None or len(storage.buffer) == 0:
            return
        every_iter = self.every_iter or max(1, int(strategy.train_epochs))
        if int(strategy.clock.train_iterations) % every_iter == 0:
            self._set_preconditioners(strategy)

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        if not self._preconditioners:
            return
        for name, module in strategy.model.named_modules():
            info = self._preconditioners.get(name)
            if info is None or module.weight.grad is None:
                continue
            _precondition_layer_gradient(info.matrix, module.weight, module.bias,
                                         parallel=info.parallel)
        self.preconditioned_update_count += 1

    def summary(self):
        return {
            "method": "Layerwise Proximal Replay",
            "omega_0": self.omega_0,
            "beta": self.beta,
            "every_iter": self.every_iter,
            "n_data": self.n_data,
            "batch_size": self.batch_size,
            "preconditioner_updates": self.update_count,
            "preconditioned_updates": self.preconditioned_update_count,
            "last_buffer_size": self.last_buffer_size,
            "extra_memory_scans": self.update_count,
        }

    @torch.no_grad()
    def _set_preconditioners(self, strategy):
        storage = self.storage_policy or getattr(strategy, "storage_policy", None)
        if storage is None:
            raise RuntimeError("LPR requires a replay storage policy")
        buffer = storage.buffer
        n_data = len(buffer) if self.n_data is None else min(len(buffer), self.n_data)
        loader = torch.utils.data.DataLoader(
            buffer, shuffle=True, batch_size=min(self.batch_size, n_data)
        )
        stats = {}
        was_training = strategy.model.training
        strategy.model.eval()
        seen = 0
        for batch in loader:
            if seen >= n_data:
                break
            x = batch[0].to(strategy.device)
            activations = {}
            hooks = []

            def capture(name):
                def hook(module, inputs, output):
                    activations[name] = (module, inputs[0].detach())
                return hook

            for name, module in strategy.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                    hooks.append(module.register_forward_hook(capture(name)))
            strategy.model(x)
            for hook in hooks:
                hook.remove()
            seen += len(x)

            for name, (module, acts) in activations.items():
                if isinstance(module, nn.Conv2d):
                    ucov, n_per_data = _conv_ucov(acts, module)
                    exponent = self.conv_beta if self.conv_beta is not None else self.beta
                elif isinstance(module, nn.Linear):
                    ucov, n_per_data = _linear_ucov(acts, module)
                    exponent = 0.0
                else:
                    ucov, n_per_data = _bn_ucov(acts, module)
                    exponent = self.bn_beta if self.bn_beta is not None else self.beta
                scale = self.omega_0 / (n_per_data ** exponent)
                scaled = [scale * value for value in ucov] if isinstance(ucov, list) else scale * ucov
                if name not in stats:
                    stats[name] = scaled
                elif isinstance(scaled, list):
                    stats[name] = [a + b for a, b in zip(stats[name], scaled)]
                else:
                    stats[name] = stats[name] + scaled
        if was_training:
            strategy.model.train()

        result = {}
        for name, ucov in stats.items():
            if isinstance(ucov, list):
                matrices = [torch.linalg.inv(v / max(1, seen) + torch.eye(v.size(0), device=v.device)) for v in ucov]
                result[name] = _Preconditioner(torch.stack(matrices), parallel=True)
            else:
                matrix = torch.linalg.inv(ucov / max(1, seen) + torch.eye(ucov.size(0), device=ucov.device))
                result[name] = _Preconditioner(matrix)
        self._preconditioners = result
        self.update_count += 1
        self.last_buffer_size = len(buffer)


def _with_bias(acts, has_bias):
    if not has_bias:
        return acts
    ones = torch.ones(acts.size(0), 1, device=acts.device, dtype=acts.dtype)
    return torch.cat((acts, ones), dim=1)


def _linear_ucov(acts, module):
    acts = _with_bias(acts, module.bias is not None)
    return acts.T @ acts, 1


def _conv_ucov(acts, module):
    unfolded = nn.Unfold(module.kernel_size, dilation=module.dilation,
                         padding=module.padding, stride=module.stride)(acts)
    unfolded = unfolded.transpose(-1, -2).reshape(-1, unfolded.size(1))
    unfolded = _with_bias(unfolded, module.bias is not None)
    return unfolded.T @ unfolded, unfolded.size(0) // acts.size(0)


def _bn_ucov(acts, module):
    flat = acts.transpose(0, 1).flatten(1)
    mean = module.running_mean if module.track_running_stats else flat.mean(dim=1)
    var = module.running_var if module.track_running_stats else flat.var(dim=1, unbiased=False)
    flat = (flat - mean.view(-1, 1)) / torch.sqrt(var.view(-1, 1) + module.eps)
    covs = []
    for channel in flat:
        channel = _with_bias(channel[:, None], module.bias is not None).T
        covs.append(channel @ channel.T)
    return covs, acts.size(-2) * acts.size(-1)


def _precondition_layer_gradient(matrix, weight, bias, *, parallel):
    if bias is not None:
        grad = torch.cat((weight.grad.data.view(weight.size(0), -1),
                          bias.grad.data.view(bias.size(0), -1)), dim=1)
    else:
        grad = weight.grad.data.view(weight.size(0), -1)
    projected = torch.einsum("cd,cde->ce" if parallel else "cd,de->ce", grad, matrix)
    if bias is not None:
        weight.grad.data = projected[..., :-1].view_as(weight)
        bias.grad.data = projected[..., -1:].view_as(bias)
    else:
        weight.grad.data = projected.view_as(weight)


__all__ = ["LPRPlugin"]
