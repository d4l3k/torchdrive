from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Set, Union

import torch
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from torchdrive.autograd import log_grad_norm

from torchdrive.losses import losses_backward


def _cpu_float(
    v: Union[torch.Tensor, int, float, bool]
) -> Union[torch.Tensor, int, float, bool]:
    if isinstance(v, torch.Tensor):
        return v.detach().float().cpu()
    return v


@dataclass
class Context:
    log_img: bool
    log_text: bool
    global_step: int
    scaler: Optional[amp.GradScaler]
    writer: Optional[SummaryWriter]
    start_frame: int
    output: str
    weights: torch.Tensor
    cam_feats: Mapping[str, torch.Tensor] = field(default_factory=dict)

    name: str = "<unknown>"

    _logged: Set[str] = field(default_factory=set)

    def backward(self, losses: Dict[str, torch.Tensor]) -> None:
        losses_backward(losses, scaler=self.scaler, weights=self.weights)

    def add_scalars(self, name: str, scalars: Dict[str, torch.Tensor]) -> None:
        if self.writer:
            assert self.log_text
            key = f"{self.name}-{name}"
            self._check_key(key)
            # pyre-fixme[16]: `Optional` has no attribute `add_scalars`.
            self.writer.add_scalars(
                key,
                {k: _cpu_float(v) for k, v in scalars.items()},
                global_step=self.global_step,
            )

    def add_scalar(
        self, name: str, scalar: Union[int, float, bool, torch.Tensor]
    ) -> None:
        if self.writer:
            assert self.log_text
            key = f"{self.name}-{name}"
            self._check_key(key)
            # pyre-fixme[16]: `Optional` has no attribute `add_scalar`.
            self.writer.add_scalar(
                key, _cpu_float(scalar), global_step=self.global_step
            )

    def add_image(self, name: str, img: torch.Tensor) -> None:
        if self.writer:
            assert self.log_img
            key = f"{self.name}-{name}"
            self._check_key(key)
            # pyre-fixme[16]: `Optional` has no attribute `add_image`.
            self.writer.add_image(key, img, global_step=self.global_step)

    def add_figure(self, name: str, figure: object) -> None:
        if self.writer:
            assert self.log_img
            key = f"{self.name}-{name}"
            self._check_key(key)
            # pyre-fixme[16]: `Optional` has no attribute `add_figure`.
            self.writer.add_figure(key, figure, global_step=self.global_step)

    def log_grad_norm(self, tensor: torch.Tensor, key: str, tag: str) -> torch.Tensor:
        if not self.log_text or not self.writer:
            return tensor
        return log_grad_norm(
            tensor, self.writer, f"{self.name}-{key}", tag, self.global_step
        )

    def _check_key(self, key: str) -> None:
        if key in self._logged:
            raise RuntimeError(f"already logged {key}")
        self._logged.add(key)
