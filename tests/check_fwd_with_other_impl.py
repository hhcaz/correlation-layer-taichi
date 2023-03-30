import torch
import argparse
import taichi as ti
from corr_taichi import CorrTaichi

try:
    assert torch.cuda.is_available()
except Exception as e:
    print("[INFO] Will evaluate on cuda which is not availabel")
    quit()

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except Exception as e:
    print("[INFO] Please install `spatial-correlation-sampler` first!")
    print("[INFO] See: https://github.com/ClementPinard/Pytorch-Correlation-extension")
    quit()


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, default=4)
parser.add_argument("-c", "--channel", type=int, default=256)
parser.add_argument("--height", type=int, default=48)
parser.add_argument("-w", "--width", type=int, default=64)
parser.add_argument("--max_disp", type=int, default=20)
parser.add_argument("--dila_patch", type=int, default=2)
parser.add_argument("-d", "--dtype", choices=["f16", "f32", "f64"], default="f32")
args = parser.parse_args()


ti.init(arch=ti.cuda)
device = torch.device("cuda:0")


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


ti_kernel = CorrTaichi(max_disp, dila_patch)

patch_size = max_disp * 2 // dila_patch + 1
sp_kernel = SpatialCorrelationSampler(
    kernel_size=1,
    patch_size=patch_size,
    stride=1,
    padding=0,
    dilation=1,
    dilation_patch=dila_patch
)

x0 = torch.randn(B, C, H, W).to(device)
x1 = torch.randn_like(x0)

corr_ti: torch.Tensor = ti_kernel(x0, x1)
corr_sp: torch.Tensor = sp_kernel(x0, x1)
corr_sp = corr_sp.view_as(corr_ti)

print("[INFO] Max abs error with spatial-correlation-sampler = {}"
      .format((corr_sp - corr_ti).abs().max()))

assert torch.allclose(corr_sp, corr_ti, atol=1e-4)


