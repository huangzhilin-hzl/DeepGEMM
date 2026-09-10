import torch
import random

import deep_gemm
from deep_gemm.testing import (
    test_filter,
    bench_kineto,
    calc_diff, count_bytes
)
from deep_gemm.utils import align
from generators import get_arch_major


@test_filter(lambda: get_arch_major() >= 9)
def test_hc_prenorm_gemm() -> None:
    # Needs TF32 precision for PyTorch GEMMs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print('Testing hyperconnection prenorm GEMM:')
    for m in (13, 137, 4096, 8192):
        for n, k in [(24, 28672), (24, 7680), (24, 7168)]:
            for num_splits in [None, 16]:
                a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
                b = torch.randn((n, k), dtype=torch.float, device='cuda')
                d = torch.empty((m, n), dtype=torch.float, device='cuda') if num_splits is None else \
                        torch.empty((num_splits, m, n), dtype=torch.float, device='cuda')
                s = torch.empty((m, ), dtype=torch.float, device='cuda') if num_splits is None else \
                        torch.empty((num_splits, m), dtype=torch.float, device='cuda')
                deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)
                final_d = d if num_splits is None else d.sum(0)
                final_s = s if num_splits is None else s.sum(0)

                ref_d = a.float() @ b.T
                ref_s = a.float().square().sum(-1)

                diff = max(calc_diff(final_d, ref_d), calc_diff(final_s, ref_s))
                assert diff < 1e-8, f'{m=}, {n=}, {k=}, {diff:.10f}'

                t = bench_kineto(lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits), 'tf32_hc_prenorm_gemm', suppress_kineto_output=True)
                print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, num_splits={(num_splits or 0):2}): '
                      f'{t * 1e6:4.0f} us | '
                      f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                      f'{count_bytes(a, b, d, s) / 1e9 / t:4.0f} GB/s')
    print()




@test_filter(lambda: get_arch_major() == 9)
def test_hc_prenorm_register_lifetime() -> None:
    """Exercise the RS WGMMA loop tail and output reuse under CUDA Graphs."""
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for m in (192, 240, 256):
            for num_splits in (16, 19, 20):
                # K has 256 blocks: split19 executes 14 steps in its first
                # nine CTAs and 13 in the rest, crossing the unrolled tail.
                generator = torch.Generator(device='cuda').manual_seed(286)
                sources = [torch.randn((m, 16384), dtype=torch.bfloat16,
                                       device='cuda', generator=generator)
                           for _ in range(2)]
                a = sources[0].clone()
                # BF16-valued FP32 weights are exact in TF32, so the
                # reference checks register lifetime without input rounding.
                b = torch.randn((24, 16384), dtype=torch.bfloat16,
                                device='cuda', generator=generator).float()
                d = torch.empty((num_splits, m, 24), device='cuda')
                s = torch.empty((num_splits, m), device='cuda')

                def run():
                    deep_gemm.tf32_hc_prenorm_gemm(
                        a, b, d, s, num_splits=num_splits)

                run()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    run()

                first_state = None
                for state in (0, 1, 0):
                    a.copy_(sources[state])
                    run()
                    first = (d.clone(), s.clone())
                    reference_d = a.float() @ b.T
                    reference_s = a.float().square().sum(-1)
                    # Use the existing HC test's numerical reference bound.
                    diff = max(calc_diff(d.sum(0), reference_d),
                               calc_diff(s.sum(0), reference_s))
                    assert diff < 1e-8, (m, num_splits, state, diff)
                    if state == 0:
                        if first_state is None:
                            first_state = first
                        else:
                            assert all(torch.equal(x.view(torch.uint8), y.view(torch.uint8))
                                       for x, y in zip(first, first_state)), (m, num_splits, 'revisit')
                    for kind in ('ordinary', 'graph'):
                        for repeat in range(32):
                            d.fill_(float('nan'))
                            s.fill_(float('nan'))
                            run() if kind == 'ordinary' else graph.replay()
                            assert all(torch.isfinite(x).all().item() for x in (d, s)), (
                                m, num_splits, state, kind, repeat, 'nonfinite')
                            assert all(torch.equal(x.view(torch.uint8), y.view(torch.uint8))
                                       for x, y in zip((d, s), first)), (
                                           m, num_splits, state, kind, repeat, 'raw repeat')
                del graph
                print(f' > HC register lifetime: {m=}, {num_splits=}, '
                      'three input states, 32 ordinary and 32 Graph replays passed')
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_hc_prenorm_register_lifetime()
    test_hc_prenorm_gemm()
