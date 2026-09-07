"""Distributed regression for replicated-input MXFP4 MegaMoE.

Run with: python tests/test_sm90_mega_moe_replicated.py --num-processes 8
Uses different expert shards, identical input buffers, receive-count checks,
empty/masked/imbalanced routes, and repeated CUDA Graph replay.
"""

import argparse
import json

import torch
import torch.distributed as dist

def _quantize_grouped_mxfp4(weight):
    from deep_gemm.utils import per_token_cast_to_fp4

    experts, n, k = weight.shape
    packed = torch.empty(experts, n, k // 2, device=weight.device, dtype=torch.int8)
    scales = torch.empty(experts, n, k // 32, device=weight.device, dtype=torch.float32)
    for expert in range(experts):
        packed[expert], scales[expert] = per_token_cast_to_fp4(
            weight[expert], use_ue8m0=True, gran_k=32
        )
    return packed, scales


def worker(local_rank, args):
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8
    from deep_gemm.utils.dist import init_dist

    rank, world, group = init_dist(local_rank, args.num_processes)
    hidden, intermediate, experts, topk = 4096, 2048, 256, 6
    local_experts = experts // world
    torch.manual_seed(1709 + rank)
    l1, l2 = deep_gemm.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
        _quantize_grouped_mxfp4(
            torch.randn(
                local_experts,
                2 * intermediate,
                hidden,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.05
        ),
        _quantize_grouped_mxfp4(
            torch.randn(
                local_experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16
            )
            * 0.05
        ),
    )
    buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
        group, experts, 256, topk, hidden, intermediate
    )
    cases = [(m, "random") for m in (0, 1, 3, 8, 16, 64, 128, 256)]
    cases += [(16, "masked"), (128, "concentrated")]
    try:
        for fast_math in (True, False):
            for m, routing in cases:
                acts = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
                quant, scales = per_token_cast_to_fp8(
                    acts, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
                )
                scores = torch.randn(m, experts, device="cuda")
                values, ids = scores.topk(topk, dim=-1)
                weights = values.softmax(-1)
                if routing == "concentrated":
                    ids.copy_(torch.arange(topk, device="cuda").expand(m, topk))
                if routing == "masked":
                    ids.fill_(-1)
                    weights.zero_()
                else:
                    mask = torch.rand(m, topk, device="cuda") < 0.1
                    ids.masked_fill_(mask, -1)
                    weights.masked_fill_(mask, 0)
                if m:
                    for tensor in (quant, scales, ids, weights):
                        dist.broadcast(tensor.view(torch.uint8), src=0, group=group)
                buffer.x[:m].copy_(quant)
                buffer.x_sf[:m].copy_(scales)
                buffer.topk_idx[:m].copy_(ids)
                buffer.topk_weights[:m].copy_(weights)
                counts = torch.bincount(ids[ids >= 0], minlength=experts)
                expected = counts[rank * local_experts : (rank + 1) * local_experts]
                recv = torch.zeros(local_experts, device="cuda", dtype=torch.int32)
                reference = torch.empty(m, hidden, device="cuda", dtype=torch.bfloat16)
                output = torch.empty_like(reference)

                def run(y, replicated):
                    deep_gemm.fp8_mxfp4_mega_moe(
                        y,
                        l1,
                        l2,
                        buffer,
                        cumulative_local_expert_recv_stats=recv,
                        activation_clamp=10.0,
                        fast_math=fast_math,
                        replicated_input=replicated,
                    )

                run(reference, False)
                torch.cuda.synchronize()
                assert torch.equal(recv.long(), expected * world)
                recv.zero_()
                run(output, True)
                torch.cuda.synchronize()
                assert torch.equal(recv.long(), expected)
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                assert bool(torch.isfinite(output).all())

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(output, True)
                recv.zero_()
                for _ in range(4):
                    graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                assert torch.equal(recv.long(), expected * 4)
                del graph
                dist.barrier(group=group)
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "m": m,
                                "routing": routing,
                                "fast_math": fast_math,
                                "bitwise_equal": True,
                                "receive_counts": "passed",
                                "graph_replays": 4,
                            }
                        ),
                        flush=True,
                    )
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=8)
    args = parser.parse_args()
    assert 256 % args.num_processes == 0
    torch.multiprocessing.spawn(worker, args=(args,), nprocs=args.num_processes)
