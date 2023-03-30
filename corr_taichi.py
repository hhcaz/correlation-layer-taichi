import os
import torch
import inspect
import taichi as ti
import torch.nn.functional as F


@ti.data_oriented
class CorrKernel(object):

    PARALLELIZE = os.cpu_count()
    FWD_BLOCK_DIM = 64
    BWD_BLOCK_DIM = 32
    USE_SHARED_ARRAY = True

    def __init__(self):
        self.dtype = ti.float32

    def config_shared_array_dtype(self, dtype):
        self.dtype = dtype

    @ti.kernel
    def forward(
        self,
        ox: ti.types.ndarray(dtype=ti.i32, ndim=1),
        oy: ti.types.ndarray(dtype=ti.i32, ndim=1),
        fmap0: ti.types.ndarray(ndim=4),
        fmap1_pad: ti.types.ndarray(ndim=4),
        corr: ti.types.ndarray(ndim=4),
    ):
        """
        Arguments:
        - ox: (L,), offset x
        - oy: (L,), offset y
        - fmap0: (B, H, W, C)
        - fmap1_pad: (B, Hp, Wp, C)
        - corr: (B, L, H, W)
        """
        L = ox.shape[0]
        B, H, W, C = fmap0.shape

        ti.loop_config(
            block_dim=self.FWD_BLOCK_DIM,
            parallelize=self.PARALLELIZE
        )

        for b, i0, j0, l in ti.ndrange(B, H, W, L):
            i1 = i0 + oy[l]
            j1 = j0 + ox[l]
            dot_sum = fmap0[b, i0, j0, 0] * 0
            for c in range(C):
                dot_sum += fmap0[b, i0, j0, c] * fmap1_pad[b, i1, j1, c]
            corr[b, l, i0, j0] = dot_sum

    @ti.kernel
    def forward_shared(
        self,
        ox: ti.types.ndarray(dtype=ti.i32, ndim=1),
        oy: ti.types.ndarray(dtype=ti.i32, ndim=1),
        fmap0: ti.types.ndarray(ndim=4),
        fmap1_pad: ti.types.ndarray(ndim=4),
        corr: ti.types.ndarray(ndim=4), 
    ):
        """
        Arguments:
        - ox: (L,), offset x
        - oy: (L,), offset y
        - fmap0: (B, H, W, C)
        - fmap1_pad: (B, Hp, Wp, C)
        - corr: (B, L, H, W)
        """
        L = ox.shape[0]
        B, H, W, C = fmap0.shape

        ti.loop_config(block_dim=self.FWD_BLOCK_DIM)
        num_c_partitions = C//self.FWD_BLOCK_DIM + (C%self.FWD_BLOCK_DIM > 0)

        for b, i0, j0, thread_idx in ti.ndrange(B, H, W, self.FWD_BLOCK_DIM):
            prod_sum = ti.simt.block.SharedArray((self.FWD_BLOCK_DIM,), self.dtype)

            for l in range(L):
                oyl = oy[l]; oxl = ox[l]
                prod_sum[thread_idx] = 0

                for c_part_id in range(num_c_partitions):
                    c = c_part_id * self.FWD_BLOCK_DIM + thread_idx
                    if c < C:
                        prod_sum[thread_idx] += fmap0[b, i0, j0, c] * fmap1_pad[b, i0+oyl, j0+oxl, c]
                
                ti.simt.block.sync()
                if thread_idx == 0:
                    reduce_sum = ti.cast(0, self.dtype)
                    for tids in range(self.FWD_BLOCK_DIM):
                        reduce_sum += prod_sum[tids]
                    corr[b, l, i0, j0] = reduce_sum
                ti.simt.block.sync()

    @ti.kernel
    def backward_grad1(
        self,
        ox: ti.types.ndarray(dtype=ti.i32, ndim=1),
        oy: ti.types.ndarray(dtype=ti.i32, ndim=1),
        fmap1_pad: ti.types.ndarray(ndim=4),
        grad_corr: ti.types.ndarray(ndim=4), 
        grad_fmap0: ti.types.ndarray(ndim=4), 
    ):
        """
        Arguments:
        - ox: (L,)
        - oy: (L,)
        - fmap1_pad: (B, C, Hp, Wp)
        - grad_corr: (B, L, H, W)
        - grad_fmap0: (B, C, H, W)
        """
        L = ox.shape[0]
        B, C, H, W = grad_fmap0.shape

        ti.loop_config(
            block_dim=self.BWD_BLOCK_DIM,
            parallelize=self.PARALLELIZE
        )

        for b, c, l in ti.ndrange(B, C, L):
            oyl = oy[l]; oxl = ox[l]
            for i0, j0 in ti.ndrange(H, W):
                ti.atomic_add(
                    grad_fmap0[b, c, i0, j0],
                    grad_corr[b, l, i0, j0] * fmap1_pad[b, c, i0+oyl, j0+oxl]
                )

    @ti.kernel
    def backward_grad2(
        self,
        ox: ti.types.ndarray(dtype=ti.i32, ndim=1),
        oy: ti.types.ndarray(dtype=ti.i32, ndim=1),
        fmap0: ti.types.ndarray(ndim=4),
        grad_corr: ti.types.ndarray(ndim=4), 
        grad_fmap1_pad: ti.types.ndarray(ndim=4), 
    ):
        """
        Arguments:
        - ox: (L,)
        - oy: (L,)
        - fmap0: (B, C, H, W)
        - grad_corr: (B, L, H, W)
        - grad_fmap1_pad: (B, C, Hp, Wp)
        """
        L = ox.shape[0]
        B, C, H, W = fmap0.shape

        ti.loop_config(
            block_dim=self.BWD_BLOCK_DIM,
            parallelize=self.PARALLELIZE
        )

        for b, c, l in ti.ndrange(B, C, L):
            oyl = oy[l]; oxl = ox[l]
            for i0, j0 in ti.ndrange(H, W):
                ti.atomic_add(
                    grad_fmap1_pad[b, c, i0+oyl, j0+oxl],
                    grad_corr[b, l, i0, j0] * fmap0[b, c, i0, j0]
                )


dtype_mapping = {
    torch.float16: ti.float16,
    torch.float32: ti.float32,
    torch.float64: ti.float64,
    torch.uint8: ti.uint8,
    torch.int8: ti.int8,
    torch.int16: ti.int16,
    torch.int32: ti.int32,
    torch.int64: ti.int64
}

class _CorrFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        fmap0: torch.Tensor, 
        fmap1_pad: torch.Tensor, 
        ox: torch.Tensor, 
        oy: torch.Tensor, 
        kernel: CorrKernel
    ):
        """
        Arguments:
        - fmap0: (B, C, H, W)
        - fmap1_pad: (B, C, Hp, Wp)
        - ox: (L,), offset x
        - oy: (L,), offset y
        """
        ctx.save_for_backward(fmap0, fmap1_pad, ox, oy)
        ctx.kernel = kernel

        L = ox.size(0)
        B, _, H, W = fmap0.size()

        fmap0_rearrange = fmap0.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        fmap1_pad_rearange = fmap1_pad.permute(0, 2, 3, 1).contiguous()  # (B, Hp, Wp, C)

        corr = fmap0.new_zeros((B, L, H, W), requires_grad=False)  # (B, L, H, W)
        if fmap0.device != torch.device("cpu") and kernel.USE_SHARED_ARRAY:
            # enable simd shared array if specified and runs on cuda
            # config is effective only at the first run
            kernel.config_shared_array_dtype(dtype=dtype_mapping[fmap0.dtype])
            kernel.forward_shared(ox, oy, fmap0_rearrange, fmap1_pad_rearange, corr)
        else:
            kernel.forward(ox, oy, fmap0_rearrange, fmap1_pad_rearange, corr)

        return corr

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_out = grad_out.contiguous()  # (B, L, H, W)
        fmap0, fmap1_pad, ox, oy = ctx.saved_tensors
        kernel: CorrKernel = ctx.kernel

        if ctx.needs_input_grad[0]:
            grad_fmap0 = torch.zeros_like(fmap0, requires_grad=False)
            kernel.backward_grad1(ox, oy, fmap1_pad, grad_out, grad_fmap0)
        else:
            grad_fmap0 = None
        
        if ctx.needs_input_grad[1]:
            grad_fmap1_pad = torch.zeros_like(fmap1_pad, requires_grad=False)
            kernel.backward_grad2(ox, oy, fmap0, grad_out, grad_fmap1_pad)
        else:
            grad_fmap1_pad = None
        
        return grad_fmap0, grad_fmap1_pad, None, None, None


class CorrTaichi(object):
    def __init__(self, max_disp, dila_patch=1):
        """
        Arguments:
        - max_disp: maximum displacement
        - dila_patch: dilation on patch
        """
        super(CorrTaichi, self).__init__()

        patch_size = max_disp * 2 // dila_patch + 1
        pad_l = pad_t = pad_r = pad_b = max_disp

        self.patch_size = patch_size
        self.pad_size = (pad_l, pad_r, pad_t, pad_b)

        meshgrid_need_index = "indexing" in inspect.getfullargspec(torch.meshgrid).kwonlyargs
        meshgrid_kwargs = {"indexing": "ij"} if meshgrid_need_index else {}
        oy, ox = torch.meshgrid(
            torch.arange(0, patch_size) * dila_patch, 
            torch.arange(0, patch_size) * dila_patch, 
            **meshgrid_kwargs
        )
        self.oy = oy.flatten().int()  # int32
        self.ox = ox.flatten().int()  # int32
        self.kernel = CorrKernel()

    @property
    def out_channels(self):
        return self.patch_size ** 2

    def __call__(self, fmap0: torch.Tensor, fmap1: torch.Tensor):
        fmap1_pad = F.pad(fmap1, self.pad_size, "constant", 0)
        corr = _CorrFunction.apply(fmap0, fmap1_pad, self.ox, self.oy, self.kernel)
        return corr
