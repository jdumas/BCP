"""Select the tensor device; an explicit CPU request never probes CUDA."""
import os
import torch

requested = os.environ.get("BCP_DEVICE", "auto")
if requested == "auto":
    requested = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(requested)
if DEVICE.type not in ("cpu", "cuda"):
    raise ValueError(f"Unsupported BCP_DEVICE: {requested}")
if DEVICE.type == "cuda":
    try:
        torch.cuda.init()
        torch.cuda.get_device_properties(DEVICE)
    except RuntimeError as error:
        raise RuntimeError(f"BCP_DEVICE={requested} requires working CUDA: {error}") from error
