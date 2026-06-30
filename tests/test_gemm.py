import pytest
import torch

import cuda_gemm  # noqa: F401  (registers torch.ops.cuda_gemm.*)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")

OPS = [
    pytest.param(torch.ops.cuda_gemm.gemm_naive, id="naive"),
    pytest.param(torch.ops.cuda_gemm.gemm_gmem_coalesce, id="gmem_coalesce"),
    pytest.param(torch.ops.cuda_gemm.gemm_smem, id="gmem_smem"),
    pytest.param(torch.ops.cuda_gemm.gemm_tiled, id="tiled"),
]

# (M, K, N) — includes non-power-of-two and non-tile-aligned dims.
SHAPES = [
    (8, 8, 8),
    (33, 17, 5),
    (128, 256, 64),
    (1024, 1024, 1024),
    (1, 100, 1),
]


@cuda_only
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("shapes", SHAPES, ids=[f"{m}x{k}x{n}" for m, k, n in SHAPES])
def test_gemm_correctness(op, shapes):
    M, K, N = shapes
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    ref = torch.matmul(a, b)
    out = op(a, b)

    assert out.shape == (M, N)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@cuda_only
def test_shape_mismatch_raises():
    a = torch.randn(4, 5, device="cuda")
    b = torch.randn(6, 7, device="cuda")
    with pytest.raises(RuntimeError):
        torch.ops.cuda_gemm.gemm_naive(a, b)


@cuda_only
def test_non_contiguous_input():
    a = torch.randn(64, 64, device="cuda", dtype=torch.float32).t().t()  # still contiguous
    a = torch.randn(128, 128, device="cuda", dtype=torch.float32)[:, ::2]  # non-contiguous
    b = torch.randn(64, 16, device="cuda", dtype=torch.float32)
    ref = torch.matmul(a, b)
    out = torch.ops.cuda_gemm.gemm_tiled(a, b)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
