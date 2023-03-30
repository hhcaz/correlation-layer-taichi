import torch
import argparse
import taichi as ti
from corr_taichi import CorrTaichi
from torch.autograd import gradcheck


parser = argparse.ArgumentParser()
parser.add_argument("backend", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("-b", "--batch-size", type=int, default=2)
parser.add_argument("-c", "--channel", type=int, default=2)
parser.add_argument("-w", "--width", type=int, default=10)
parser.add_argument("--max_disp", type=int, default=9)
parser.add_argument("--dila_patch", type=int, default=2)

args = parser.parse_args()


if args.backend == "cuda":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ti.init(arch=ti.cuda, device_memory_GB=0.5)
else:
    ti.init(arch=ti.cpu)
device = torch.device(args.backend)

B = args.batch_size
C = args.channel
H = W = args.width
max_disp = args.max_disp
dila_patch = args.dila_patch

ti_kernel = CorrTaichi(max_disp, dila_patch)

x0 = torch.randn(
    size=(B, C, H, W),
    dtype=torch.float64,
    device=device,
    requires_grad=True
)
x1 = torch.randn_like(x0, requires_grad=True)

print("[INFO] Start checking grad...")
if gradcheck(ti_kernel, [x0, x1]):
    print("[INFO] OK!")

