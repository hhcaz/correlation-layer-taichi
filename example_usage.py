import torch
import taichi as ti
from corr_taichi import CorrTaichi


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    ti.init(arch=ti.cuda, device_memory_GB=0.5)
else:
    device = torch.device("cpu")
    ti.init(arch=ti.cpu)

# input tensor shape
B, C, H, W = (4, 256, 48, 64)

# correlation config (same as FlowNetC)
max_displacement = 20
stride2 = 2

kernel = CorrTaichi(
    max_disp=max_displacement,
    dila_patch=stride2
)

x0 = torch.randn((B, C, H, W), device=device, requires_grad=True)
x1 = torch.randn_like(x0, requires_grad=True)
corr: torch.Tensor = kernel(x0, x1)

print("[INFO] Input tensor size: {}".format(x0.size()))  # (4, 256, 48, 64)
print("[INFO] Output tensor size: {}".format(corr.size()))  # (4, 441, 48, 64)

