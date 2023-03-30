import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import time
import torch
import argparse
from tqdm import tqdm
from corr_pure_torch import CorrTorch


parser = argparse.ArgumentParser()
parser.add_argument("backend", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("-b", "--batch-size", type=int, default=4)
parser.add_argument("-c", "--channel", type=int, default=256)
parser.add_argument("--height", type=int, default=48)
parser.add_argument("-w", "--width", type=int, default=64)
parser.add_argument("--max_disp", type=int, default=20)
parser.add_argument("--dila_patch", type=int, default=2)
parser.add_argument("-r", "--runs", type=int, default=100)
parser.add_argument("-d", "--dtype", choices=["f16", "f32", "f64"], default="f32")
args = parser.parse_args()


device = torch.device(args.backend)

if args.dtype == "f16":
    dtype = torch.float16
elif args.dtype == "f32":
    dtype = torch.float32
elif args.dtype == "f64":
    dype = torch.float64


B = args.batch_size
C = args.channel
H = args.height
W = args.width
max_disp = args.max_disp
dila_patch = args.dila_patch
runs = args.runs


kernel = CorrTorch(max_disp, dila_patch)
kernel = torch.jit.script(kernel)

x0 = torch.randn((B, C, H, W), dtype=dtype, device=device, requires_grad=True)
x1 = torch.randn_like(x0).requires_grad_(True)

# warm up
for _ in range(2):
    corr: torch.Tensor = kernel(x0, x1)
corr.mean().backward()

print("[INFO] Input tensor size = {}".format(x0.size()))
print("[INFO] Output tensor size = {}".format(corr.size()))


fw_time = 0
bw_time = 0
for _ in tqdm(range(runs)):
    if x0.grad is not None:
        x0.grad.data.zero_()
    if x1.grad is not None:
        x1.grad.data.zero_()

    t0 = time.time()
    corr: torch.Tensor = kernel(x0, x1)
    fw_time += time.time() - t0

    m = corr.mean()

    t0 = time.time()
    m.backward()
    bw_time += time.time() - t0


fw_time /= runs
bw_time /= runs


print("[INFO] Avg forward time: {:.3f}ms, avg backward time: {:.3f}ms"
      .format(fw_time * 1e3, bw_time * 1e3))

