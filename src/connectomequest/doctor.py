"""Hardware and dependency diagnostics used in experiment manifests."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass

import pyarrow
import torch


@dataclass(frozen=True, slots=True)
class HardwareReport:
    python: str
    platform: str
    cpu_count: int
    torch: str
    pyarrow: str
    cuda_available: bool
    cuda_device_count: int
    cuda_devices: tuple[str, ...]
    bf16_supported: bool


def inspect_hardware() -> HardwareReport:
    cuda = torch.cuda.is_available()
    devices = (
        tuple(torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count()))
        if cuda
        else ()
    )
    bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    return HardwareReport(
        python=platform.python_version(),
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 1,
        torch=torch.__version__,
        pyarrow=pyarrow.__version__,
        cuda_available=cuda,
        cuda_device_count=len(devices),
        cuda_devices=devices,
        bf16_supported=bf16,
    )


def report_dict() -> dict:
    return asdict(inspect_hardware())
