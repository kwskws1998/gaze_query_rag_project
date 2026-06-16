from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TorchDeviceStatus:
    cuda_available: bool
    mps_built: bool
    mps_available: bool
    mps_usable: bool
    mps_error: str | None = None


def get_torch_device_status() -> TorchDeviceStatus:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend is not None and mps_backend.is_built())
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    mps_usable = False
    mps_error: str | None = None
    if mps_available:
        try:
            torch.empty(1, device="mps")
            mps_usable = True
        except Exception as exc:
            mps_error = str(exc)
    elif mps_built:
        try:
            torch.empty(1, device="mps")
        except Exception as exc:
            mps_error = str(exc)
    return TorchDeviceStatus(
        cuda_available=cuda_available,
        mps_built=mps_built,
        mps_available=mps_available,
        mps_usable=mps_usable,
        mps_error=mps_error,
    )


def resolve_torch_device(device: str):
    import torch

    requested = device.lower()
    status = get_torch_device_status()
    if requested == "auto":
        if status.cuda_available:
            return torch.device("cuda")
        if status.mps_usable:
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not status.cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if requested == "mps" and not status.mps_usable:
        detail = f" Detail: {status.mps_error}" if status.mps_error else ""
        raise RuntimeError(
            "MPS was requested but the current PyTorch runtime cannot allocate an MPS tensor."
            f" mps_built={status.mps_built}, mps_available={status.mps_available}.{detail}"
        )
    return torch.device(device)
