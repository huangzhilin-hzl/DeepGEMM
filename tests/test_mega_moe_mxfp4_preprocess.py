"""CPU contract tests for SM90 Humming-compatible MXFP4 weight transforms."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


# Load the CUDA-independent transform module without importing deep_gemm._C.
_MODULE_PATH = Path(__file__).parents[1] / 'deep_gemm' / 'mega' / 'mxfp4.py'
_SPEC = importlib.util.spec_from_file_location('deep_gemm_mxfp4_contract', _MODULE_PATH)
mxfp4 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mxfp4)


def _valid_checkpoint_weights(
    num_experts: int = 1,
    hidden: int = 512,
    intermediate: int = 256,
):
    l1_w = torch.arange(
        num_experts * 2 * intermediate * (hidden // 2), dtype=torch.int32
    ).to(torch.uint8).reshape(num_experts, 2 * intermediate, hidden // 2)
    l1_sf = torch.full(
        (num_experts, 2 * intermediate, hidden // 32), 109, dtype=torch.uint8)
    l2_w = torch.arange(
        num_experts * hidden * (intermediate // 2), dtype=torch.int32
    ).to(torch.uint8).reshape(num_experts, hidden, intermediate // 2)
    l2_sf = torch.full(
        (num_experts, hidden, intermediate // 32), 109, dtype=torch.uint8)
    return (l1_w, l1_sf), (l2_w, l2_sf)


def _load_mega_api_for_cpu():
    package_name = '_deep_gemm_sm90_api_contract'
    package = types.ModuleType(package_name)
    package.__path__ = [str(_MODULE_PATH.parents[1])]
    package._C = types.SimpleNamespace(get_num_sms=lambda: 1)
    utils = types.ModuleType(f'{package_name}.utils')
    utils.__path__ = [str(_MODULE_PATH.parents[1] / 'utils')]
    math_module = types.ModuleType(f'{package_name}.utils.math')
    math_module.align = lambda value, alignment: (
        (value + alignment - 1) // alignment * alignment)
    module_name = f'{package_name}.mega'
    spec = importlib.util.spec_from_file_location(
        module_name,
        _MODULE_PATH.parent / '__init__.py',
        submodule_search_locations=[str(_MODULE_PATH.parent)],
    )
    mega = importlib.util.module_from_spec(spec)
    sys.modules.update({
        package_name: package,
        f'{package_name}.utils': utils,
        f'{package_name}.utils.math': math_module,
        module_name: mega,
    })
    spec.loader.exec_module(mega)
    return mega, package_name


def _unload_fake_package(package_name: str) -> None:
    for name in tuple(sys.modules):
        if name == package_name or name.startswith(f'{package_name}.'):
            sys.modules.pop(name, None)


def test_canonical_transform_returns_processed_triples_and_interleaves_l1():
    (l1_w, l1_sf), (l2_w, l2_sf) = _valid_checkpoint_weights()
    # Avoid negative-zero E2M1 nibbles, which preprocessing intentionally
    # canonicalizes to positive zero, while retaining varied sign bits.
    l1_w.bitwise_or_(0x11)
    l2_w.bitwise_or_(0x11)
    l1_sf.copy_((
        109 + torch.arange(l1_sf.size(1), dtype=torch.int32) % 4
    ).to(torch.uint8).reshape(1, -1, 1).expand_as(l1_sf))
    l2_sf.copy_((
        109 + torch.arange(l2_sf.size(1), dtype=torch.int32) % 4
    ).to(torch.uint8).reshape(1, -1, 1).expand_as(l2_sf))
    e8m0_dtype = getattr(torch, 'float8_e8m0fnu', None)
    l1_sf_input = l1_sf if e8m0_dtype is None else l1_sf.view(e8m0_dtype)
    l2_sf_input = l2_sf if e8m0_dtype is None else l2_sf.view(e8m0_dtype)

    transformed_l1, transformed_l2 = (
        mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            (l1_w, l1_sf_input), (l2_w, l2_sf_input)))
    half = l1_w.size(1) // 2
    row_order = torch.tensor([
        row
        for start in range(0, half, 8)
        for row in (
            *range(start, start + 8),
            *range(half + start, half + start + 8),
        )
    ])

    assert len(transformed_l1) == len(transformed_l2) == 3
    assert transformed_l1[0].dtype == torch.int8
    assert transformed_l1[1].dtype == torch.uint8
    assert torch.equal(
        mxfp4._restore_mxfp4_sign_bits_from_sm90(transformed_l1[0])
            .view(torch.uint8),
        l1_w[:, row_order],
    )
    assert torch.equal(
        mxfp4._restore_mxfp4_scales_from_sm90(transformed_l1[1]),
        l1_sf[:, row_order] - 108,
    )
    assert transformed_l2[0].dtype == torch.int8
    assert torch.equal(
        mxfp4._restore_mxfp4_sign_bits_from_sm90(transformed_l2[0])
            .view(torch.uint8),
        l2_w,
    )
    assert torch.equal(
        mxfp4._restore_mxfp4_scales_from_sm90(transformed_l2[1]),
        l2_sf - 108,
    )
    assert transformed_l1[2].item() == pytest.approx(2.0 ** -19)
    assert transformed_l2[2].item() == pytest.approx(2.0 ** -19)
    assert all(tensor.is_contiguous() for tensor in transformed_l1 + transformed_l2)


def test_large_hidden_preserves_natural_scale_layout():
    (l1_w, l1_sf), (l2_w, l2_sf) = _valid_checkpoint_weights(hidden=4608)
    l1_sf.copy_((
        109 + torch.arange(l1_sf.numel(), dtype=torch.int32).reshape(l1_sf.shape) % 4
    ).to(torch.uint8))
    l2_sf.copy_((
        109 + torch.arange(l2_sf.numel(), dtype=torch.int32).reshape(l2_sf.shape) % 4
    ).to(torch.uint8))

    transformed_l1, transformed_l2 = (
        mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            (l1_w, l1_sf), (l2_w, l2_sf)))

    assert torch.equal(
        transformed_l1[1], mxfp4._interleave_mxfp4_rows(l1_sf - 108))
    assert torch.equal(transformed_l2[1], l2_sf - 108)


def test_fp32_ue8m0_conversion_including_finite_endpoints():
    fp32_scales = torch.tensor(
        [2.0 ** -127, 2.0 ** -126, 1.0, 2.0 ** 127], dtype=torch.float32)
    assert torch.equal(
        mxfp4._normalize_mxfp4_ue8m0(fp32_scales),
        torch.tensor([0, 1, 127, 254], dtype=torch.uint8))

    for invalid in (0.0, -1.0, 1.5, float('inf')):
        with pytest.raises(ValueError, match='powers of two'):
            mxfp4._normalize_mxfp4_ue8m0(
                torch.tensor([invalid], dtype=torch.float32))


def test_hidden_size_contract_matches_vectorized_combine_chunks():
    assert mxfp4._is_valid_sm90_mxfp4_hidden_size(512)
    assert mxfp4._is_valid_sm90_mxfp4_hidden_size(8192)
    assert not mxfp4._is_valid_sm90_mxfp4_hidden_size(8704)
    assert mxfp4._is_valid_sm90_mxfp4_hidden_size(9216)
    assert mxfp4._uses_coalesced_mxfp4_scales_sm90(4096)
    assert not mxfp4._uses_coalesced_mxfp4_scales_sm90(4608)


def test_processed_sign_layout_is_invertible_and_has_golden_word():
    source = torch.tensor([0x91, 0xa2, 0x3b, 0x4c], dtype=torch.uint8)
    reordered = mxfp4._reorder_mxfp4_sign_bits_for_sm90(source)
    assert torch.equal(
        reordered,
        torch.tensor([0x91, 0x2a, 0xb3, 0x4c], dtype=torch.uint8))
    assert torch.equal(mxfp4._restore_mxfp4_sign_bits_from_sm90(reordered), source)

    exhaustive = torch.arange(256, dtype=torch.uint8).repeat(4).reshape(1, 1, -1)
    assert torch.equal(
        mxfp4._restore_mxfp4_sign_bits_from_sm90(
            mxfp4._reorder_mxfp4_sign_bits_for_sm90(exhaustive)),
        exhaustive)


def test_processed_e8m0_requantization_matches_golden_contract():
    packed_codes = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe] * 2,
        dtype=torch.uint8)
    weight = packed_codes.repeat(7, 1).reshape(1, 7, 16)
    sf = torch.tensor(
        [109, 108, 107, 106, 105, 104, 120], dtype=torch.uint8
    ).reshape(1, 7, 1)
    processed_w, relative_sf, weight_scale_2 = (
        mxfp4._process_mxfp4_e8m0(weight, sf))
    expected = torch.tensor([
        [0x10, 0x32, 0x54, 0x76, 0x90, 0xba, 0xdc, 0xfe],
        [0x10, 0x21, 0x32, 0x54, 0x90, 0xa9, 0xba, 0xdc],
        [0x00, 0x11, 0x21, 0x32, 0x80, 0x99, 0xa9, 0xba],
        [0x00, 0x00, 0x11, 0x21, 0x80, 0x88, 0x99, 0xa9],
        [0x00, 0x00, 0x00, 0x11, 0x80, 0x88, 0x88, 0x99],
        [0x00, 0x00, 0x00, 0x00, 0x80, 0x88, 0x88, 0x88],
        [0x10, 0x32, 0x54, 0x76, 0x90, 0xba, 0xdc, 0xfe],
    ], dtype=torch.uint8).repeat(1, 2)

    assert torch.equal(
        mxfp4._restore_mxfp4_sign_bits_from_sm90(processed_w).view(torch.uint8)[0],
        expected)
    assert torch.equal(
        relative_sf,
        torch.tensor([1, 1, 1, 1, 1, 1, 12], dtype=torch.uint8).reshape(1, 7, 1))
    assert torch.equal(weight_scale_2, torch.tensor([2.0 ** -19], dtype=torch.float32))


def test_canonical_transform_applies_checkpoint_weight_scale_2():
    (l1_w, l1_sf), (l2_w, l2_sf) = _valid_checkpoint_weights(num_experts=2)
    l1_sf[:, -1] = 120
    l2_sf[:, -1] = 120
    transformed_l1, transformed_l2 = (
        mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            (l1_w, l1_sf),
            (l2_w, l2_sf),
            l1_weight_scale_2=torch.tensor([2.0, 4.0], dtype=torch.float32),
            l2_weight_scale_2=torch.tensor([3.0, 5.0], dtype=torch.float32),
        ))

    assert len(transformed_l1) == len(transformed_l2) == 3
    assert transformed_l1[0].dtype == transformed_l2[0].dtype == torch.int8
    assert transformed_l1[1].dtype == transformed_l2[1].dtype == torch.uint8
    assert torch.equal(
        transformed_l1[2],
        torch.tensor([2.0, 4.0], dtype=torch.float32) * (2.0 ** -19))
    assert torch.equal(
        transformed_l2[2],
        torch.tensor([3.0, 5.0], dtype=torch.float32) * (2.0 ** -19))
    mxfp4._validate_processed_mxfp4_kernel_weights(transformed_l1, transformed_l2)


def test_invalid_mxfp4_contracts_fail_before_kernel_launch():
    l1, l2 = _valid_checkpoint_weights()

    with pytest.raises(TypeError, match='uint8 or int8'):
        mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            (l1[0].float(), l1[1]), l2)
    with pytest.raises(ValueError, match='packed K/2'):
        mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            (l1[0][..., :-1], l1[1]), l2)
    with pytest.raises(ValueError, match='incompatible L1/L2'):
        mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            l1, (l2[0][:, :-1], l2[1][:, :-1]))
    with pytest.raises(ValueError, match='hidden size must be divisible by 512'):
        mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            (l1[0][..., :128], l1[1][..., :8]),
            (l2[0][:, :256], l2[1][:, :256]),
        )
    with pytest.raises(ValueError, match='intermediate hidden size must be divisible by 256'):
        mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            (l1[0][:, :256], l1[1][:, :256]),
            (l2[0][..., :64], l2[1][..., :4]),
        )
    with pytest.raises(ValueError, match='NaN code 255'):
        bad_sf = torch.full_like(l1[1], 255)
        mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            (l1[0], bad_sf), l2)
    with pytest.raises(ValueError, match='both be provided'):
        mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            l1, l2, l1_weight_scale_2=torch.ones(1))


def test_kernel_payload_validation_accepts_only_processed_contiguous_triples():
    checkpoint_l1, checkpoint_l2 = _valid_checkpoint_weights(num_experts=2)
    with pytest.raises(ValueError, match='processed triples'):
        mxfp4._validate_processed_mxfp4_kernel_weights(checkpoint_l1, checkpoint_l2)

    processed_l1, processed_l2 = (
        mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            checkpoint_l1, checkpoint_l2))
    mxfp4._validate_processed_mxfp4_kernel_weights(processed_l1, processed_l2)
    noncontiguous_scale = torch.ones(4, dtype=torch.float32)[::2]
    assert not noncontiguous_scale.is_contiguous()
    with pytest.raises(ValueError, match='contiguous before launch'):
        mxfp4._validate_processed_mxfp4_kernel_weights(
            (processed_l1[0], processed_l1[1], noncontiguous_scale),
            processed_l2,
        )


def test_explicit_wrapper_rejects_the_sm100_buffer_abi_before_launch():
    mega, package_name = _load_mega_api_for_cpu()
    try:
        with pytest.raises(TypeError, match='requires an SM90SymmBuffer'):
            mega.fp8_mxfp4_mega_moe(None, (), (), object())
    finally:
        _unload_fake_package(package_name)


def test_in_kernel_profiler_decodes_ranges_and_reports_truncation(tmp_path):
    mega, package_name = _load_mega_api_for_cpu()
    try:
        profiler = mega.MegaMoeProfiler(capacity=2, rank=0, device='cpu')
        profiler.configure_workload(
            num_tokens=8192, num_ranks=8, num_experts=384, num_topk=6,
            hidden=7168, intermediate_hidden=3072)
        payload = 1 | (3 << 3) | (4 << 12) | (5 << 28) | (37 << 38)
        profiler.buffer[0, 4, 0, 0] = 3
        profiler.buffer[0, 4, 0, 1] = 7
        profiler.buffer[0, 4, 1, 0] = 1000
        profiler.buffer[0, 4, 1, 1] = 9 | (payload << 16)
        profiler.buffer[0, 4, 2, 0] = 5000
        profiler.buffer[0, 4, 2, 1] = 9 | (1 << 8) | (payload << 16)

        events = profiler.events()
        assert [event['phase'] for event in events] == ['B', 'E']
        assert events[0]['args']['phase'] == 'linear1'
        assert events[0]['args']['expert'] == 3
        assert events[0]['args']['m_block'] == 4
        assert events[0]['args']['n_block'] == 5
        assert events[0]['args']['valid_m'] == 37
        assert events[0]['args']['global_expert'] == 3
        assert events[0]['args']['m_start'] == 256
        assert events[0]['args']['m_end'] == 293
        assert events[0]['args']['n_start'] == 640
        assert events[0]['args']['n_end'] == 768
        assert events[0]['args']['shape_k'] == 7168
        assert profiler.summary()['task']['total_us'] == pytest.approx(4.0)
        assert profiler.truncation()['dropped'] == 1

        snapshot_calls = 0
        original_snapshot = profiler._snapshot

        def counted_snapshot():
            nonlocal snapshot_calls
            snapshot_calls += 1
            return original_snapshot()

        profiler._snapshot = counted_snapshot
        trace_path = profiler.export_chrome_trace(tmp_path / 'trace.json')
        trace = __import__('json').loads(trace_path.read_text())
        assert snapshot_calls == 1
        assert trace['deep_gemm']['format'] == 'mega_moe_in_kernel_v5'
        assert trace['deep_gemm']['workload']['hidden'] == 7168
        assert trace['deep_gemm']['truncation']['truncated_tracks'] == 1
        assert trace['deep_gemm']['truncation']['counts_are_lower_bounds'] is True
        task_slices = [
            event for event in trace['traceEvents']
            if event.get('cat') == 'task'
        ]
        assert task_slices[0]['name'].startswith('task L1 E3 M[256,293)')
        assert task_slices[0]['args']['event_type'] == 'task'
        assert profiler.last_export_metadata['truncation']['dropped'] == 1

        detail_profiler = mega.MegaMoeProfiler(capacity=5, rank=0, device='cpu')
        detail_profiler.configure_workload(
            num_tokens=8192, num_ranks=8, num_experts=384, num_topk=6,
            hidden=7168, intermediate_hidden=3072)
        combine_payload = (
            9 | (2 << 32) | (4 << 35) | (3 << 38) | (1 << 44))
        wgmma_payload = (
            1 | (2 << 3) | (7 << 5) | (2 << 17) | (2 << 20) |
            (1 << 23) | (1 << 25))
        pull_payload = 123 | (2 << 32) | (3 << 37)
        select_payload = 17 | (45 << 9)
        detail_profiler.buffer[0, 1, 0, 0] = 5
        detail_profiler.buffer[0, 1, 1, 0] = 100
        detail_profiler.buffer[0, 1, 1, 1] = (
            32 | (select_payload << 16))
        detail_profiler.buffer[0, 1, 2, 0] = 200
        detail_profiler.buffer[0, 1, 2, 1] = (
            32 | (1 << 8) | (select_payload << 16)
        )
        detail_profiler.buffer[0, 1, 3, 0] = 300
        detail_profiler.buffer[0, 1, 3, 1] = (
            5 | (2 << 8) | (pull_payload << 16))
        detail_profiler.buffer[0, 1, 4, 0] = 400
        detail_profiler.buffer[0, 1, 4, 1] = (
            26 | (2 << 8) | (combine_payload << 16)
        )
        detail_profiler.buffer[0, 1, 5, 0] = 500
        detail_profiler.buffer[0, 1, 5, 1] = (
            30 | (2 << 8) | (wgmma_payload << 16)
        )
        detail_events = detail_profiler.events()
        pull_args = detail_events[2]['args']
        assert pull_args['src_rank'] == 3
        assert pull_args['src_token'] == 123
        assert pull_args['src_topk'] == 2
        assert pull_args['src_route_index'] == 740
        assert pull_args['dst_rank'] == 0
        assert pull_args['dst_local_expert'] == 17
        assert pull_args['dst_global_expert'] == 17
        assert pull_args['dst_expert_token'] == 45
        assert pull_args['dst_m'] == 45
        assert pull_args['payload_overflow'] is False
        combine_args = detail_events[3]['args']
        assert combine_args['token'] == 9
        assert combine_args['chunk'] == 2
        assert combine_args['num_chunks'] == 4
        assert combine_args['topk_slot'] == 3
        assert combine_args['hidden_start'] == 3584
        assert combine_args['hidden_end'] == 5376
        assert detail_events[4]['name'] == 'wgmma.wait'
        assert detail_events[4]['args']['stage'] == 2
        assert detail_events[4]['args']['k_block'] == 7
        assert detail_events[4]['args']['k32_start'] == 2
        assert detail_events[4]['args']['k32_count'] == 2
        assert detail_events[4]['args']['expanded_slot'] == 1
        assert detail_events[4]['args']['accumulate'] is True
        assert detail_events[4]['args']['k_start'] == 960
        assert detail_events[4]['args']['k_end'] == 1024
        assert detail_events[0]['name'] == 'dispatch.select'
        select_args = detail_events[0]['args']
        assert select_args['dst_rank'] == 0
        assert select_args['dst_global_expert'] == 17
        assert select_args['dst_m'] == 45
    finally:
        _unload_fake_package(package_name)


def test_in_kernel_profiler_inherits_task_geometry_into_wgmma():
    mega, package_name = _load_mega_api_for_cpu()
    try:
        profiler = mega.MegaMoeProfiler(capacity=4, rank=3, device='cpu')
        profiler.configure_workload(
            num_tokens=8192, num_ranks=8, num_experts=384, num_topk=6,
            hidden=7168, intermediate_hidden=3072)
        task_payload = (
            2 | (17 << 3) | (2 << 12) | (4 << 28) | (37 << 38))
        wgmma_payload = (
            2 | (1 << 3) | (7 << 5) | (2 << 17) | (2 << 20))
        profiler.buffer[0, 4, 0, 0] = 4
        profiler.buffer[0, 4, 1, 0] = 100
        profiler.buffer[0, 4, 1, 1] = 9 | (task_payload << 16)
        profiler.buffer[0, 4, 2, 0] = 200
        profiler.buffer[0, 4, 2, 1] = 16 | (wgmma_payload << 16)
        profiler.buffer[0, 4, 3, 0] = 300
        profiler.buffer[0, 4, 3, 1] = (
            16 | (1 << 8) | (wgmma_payload << 16))
        profiler.buffer[0, 4, 4, 0] = 400
        profiler.buffer[0, 4, 4, 1] = (
            9 | (1 << 8) | (task_payload << 16))

        events = profiler.events()
        issue = events[1]
        assert issue['name'] == 'wgmma.issue'
        assert issue['args']['global_expert'] == 161
        assert issue['args']['m_space'] == 'expert_packed'
        assert issue['args']['m_start'] == 128
        assert issue['args']['m_end'] == 165
        assert issue['args']['n_start'] == 512
        assert issue['args']['n_end'] == 640
        assert issue['args']['k_start'] == 960
        assert issue['args']['k_end'] == 1024
        assert issue['args']['shape_k'] == 3072
        assert profiler.summary()['wgmma.issue']['total_us'] == pytest.approx(0.1)
    finally:
        _unload_fake_package(package_name)


def test_in_kernel_profiler_dispatch_keeps_ep8_hot_expert_m_exact():
    mega, package_name = _load_mega_api_for_cpu()
    try:
        profiler = mega.MegaMoeProfiler(capacity=4, rank=7, device='cpu')
        profiler.configure_workload(
            num_tokens=8192, num_ranks=8, num_experts=384, num_topk=6,
            hidden=7168, intermediate_hidden=3072)
        select_payload = 17 | (65535 << 9)
        pull_payload = 8191 | (5 << 32) | (7 << 37)
        profiler.buffer[0, 1, 0, 0] = 4
        for record, (timestamp, event, phase, payload) in enumerate((
            (100, 32, 0, select_payload),
            (200, 32, 1, select_payload),
            (300, 5, 0, pull_payload),
            (400, 5, 1, pull_payload),
        ), start=1):
            profiler.buffer[0, 1, record, 0] = timestamp
            profiler.buffer[0, 1, record, 1] = (
                event | (phase << 8) | (payload << 16))

        pull = profiler.events()[2]
        assert pull['args']['coordinates_valid'] is True
        assert pull['args']['dst_global_expert'] == 353
        assert pull['args']['dst_m'] == 65535
        assert pull['args']['src_rank'] == 7
        assert pull['args']['src_token'] == 8191
        assert pull['args']['src_topk'] == 5
        assert pull['display_name'] == (
            'dispatch.pull r7:t8191:k5 -> E353:M65535')
    finally:
        _unload_fake_package(package_name)


def test_in_kernel_profiler_dispatch_keeps_full_uint32_token_coordinates():
    mega, package_name = _load_mega_api_for_cpu()
    try:
        profiler = mega.MegaMoeProfiler(capacity=4, rank=7, device='cpu')
        profiler.configure_workload(
            num_tokens=100001, num_ranks=8, num_experts=384, num_topk=6,
            hidden=7168, intermediate_hidden=3072)
        select_payload = 17 | (1000000 << 9)
        pull_payload = 100000 | (5 << 32) | (7 << 37)
        profiler.buffer[0, 1, 0, 0] = 4
        event_index = mega.MEGA_MOE_EVENT_NAMES.index
        for record, (timestamp, event, phase, payload) in enumerate((
            (100, event_index('dispatch.select'), 0, select_payload),
            (200, event_index('dispatch.select'), 1, select_payload),
            (300, event_index('dispatch.pull'), 0, pull_payload),
            (400, event_index('dispatch.pull'), 1, pull_payload),
        ), start=1):
            profiler.buffer[0, 1, record, 0] = timestamp
            profiler.buffer[0, 1, record, 1] = (
                event | (phase << 8) | (payload << 16))

        pull = profiler.events()[2]
        assert pull['args']['coordinates_valid'] is True
        assert pull['args']['dst_m'] == 1000000
        assert pull['args']['src_token'] == 100000
        assert pull['display_name'] == (
            'dispatch.pull r7:t100000:k5 -> E353:M1000000')
    finally:
        _unload_fake_package(package_name)


def test_in_kernel_profiler_combine_keeps_full_uint32_token_coordinate():
    mega, package_name = _load_mega_api_for_cpu()
    try:
        profiler = mega.MegaMoeProfiler(capacity=1, device='cpu')
        profiler.configure_workload(
            num_tokens=100001, num_ranks=1, num_experts=8, num_topk=6,
            hidden=7168, intermediate_hidden=3072)
        payload = 100000 | (1 << 32) | (4 << 35) | (5 << 38) | (1 << 44)
        profiler.buffer[0, 4, 0, 0] = 1
        profiler.buffer[0, 4, 1, 0] = 100
        profiler.buffer[0, 4, 1, 1] = (
            mega.MEGA_MOE_EVENT_NAMES.index('combine.reduce') |
            (2 << 8) | (payload << 16))

        event = profiler.events()[0]
        assert event['args']['token'] == 100000
        assert event['args']['topk_slot'] == 5
        assert event['args']['hidden_start'] == 1792
        assert event['args']['hidden_end'] == 3584
        assert event['display_name'].startswith(
            'combine.reduce token100000 slot5 H[1792,3584)')
    finally:
        _unload_fake_package(package_name)


def test_in_kernel_profiler_overflow_never_derives_truncated_coordinates():
    mega, package_name = _load_mega_api_for_cpu()
    try:
        profiler = mega.MegaMoeProfiler(capacity=1, rank=0, device='cpu')
        profiler.configure_workload(
            num_tokens=8192, num_ranks=64, num_experts=512, num_topk=8,
            hidden=7168, intermediate_hidden=3072)
        overflow_payload = (1 << 47) | 17
        encoded = 32 | (overflow_payload << 16)
        if encoded >= 1 << 63:
            encoded -= 1 << 64
        profiler.buffer[0, 1, 0, 0] = 1
        profiler.buffer[0, 1, 1, 0] = 100
        profiler.buffer[0, 1, 1, 1] = encoded

        args = profiler.events()[0]['args']
        assert args['payload_overflow'] is True
        assert args['coordinates_valid'] is False
        assert args['coordinate_status'] == 'payload_overflow_or_missing_select'
        assert 'dst_m' not in args
        assert 'dst_global_expert' not in args
    finally:
        _unload_fake_package(package_name)


def test_in_kernel_profiler_correlates_scheduler_and_scatter_routes():
    mega, package_name = _load_mega_api_for_cpu()
    try:
        profiler = mega.MegaMoeProfiler(capacity=6, rank=0, device='cpu')
        profiler.configure_workload(
            num_tokens=8, num_ranks=8, num_experts=384, num_topk=6,
            hidden=7168, intermediate_hidden=3072)
        select_payload = 17 | (16 << 9)
        pull_payload = 123 | (2 << 32) | (3 << 37)
        profiler.buffer[0, 1, 0, 0] = 4
        for record, (event, phase, payload) in enumerate((
            (32, 0, select_payload), (32, 1, select_payload),
            (5, 0, pull_payload), (5, 1, pull_payload),
        ), start=1):
            profiler.buffer[0, 1, record, 0] = record * 100
            profiler.buffer[0, 1, record, 1] = (
                event | (phase << 8) | (payload << 16))

        task_payload = 2 | (17 << 3) | (17 << 38)
        profiler.buffer[0, 5, 0, 0] = 6
        for record, (event, phase, payload) in enumerate((
            (8, 0, 0), (8, 1, 0), (9, 0, task_payload),
            (22, 0, task_payload), (22, 1, task_payload),
            (9, 1, task_payload),
        ), start=1):
            profiler.buffer[0, 5, record, 0] = 1000 + record * 100
            profiler.buffer[0, 5, record, 1] = (
                event | (phase << 8) | (payload << 16))

        events = profiler.events()
        wait = next(
            event for event in events
            if event['name'] == 'scheduler.wait' and event['phase'] == 'B')
        scatter = next(
            event for event in events
            if event['name'] == 'nvlink.scatter' and event['phase'] == 'B')
        assert wait['args']['next_task'].startswith('L2 E17 M[0,17)')
        assert wait['display_name'].startswith('scheduler.wait -> L2 E17')
        assert scatter['args']['scatter_routes_valid'] is True
        assert scatter['args']['task_m_start'] == 0
        assert scatter['args']['task_m_end'] == 17
        assert scatter['args']['scatter_m_start'] == 16
        assert scatter['args']['scatter_m_end'] == 17
        assert scatter['args']['scatter_routes'] == 'm16->r3:t123:k2'
        assert scatter['args']['output_ranks'] == '3'
        assert scatter['args']['output_tokens'] == '123'
        assert scatter['args']['output_topk_slots'] == '2'
        assert '-> ranks[3] routes=1' in scatter['display_name']
    finally:
        _unload_fake_package(package_name)


def test_in_kernel_profiler_filters_dependencies_and_names_every_event():
    mega, package_name = _load_mega_api_for_cpu()
    try:
        with pytest.raises(ValueError, match='unknown MegaMoE profiler events'):
            mega.MegaMoeProfiler(
                capacity=1, device='cpu', event_types=['not.an.event'])
        profiler = mega.MegaMoeProfiler(
            capacity=1, device='cpu', event_types=['wgmma.issue'],
            cta_indices=[0], warp_indices=[4])
        assert profiler.requested_event_types == ('wgmma.issue',)
        assert profiler.enabled_event_types == ('kernel', 'task', 'wgmma.issue')
        expected_mask = (
            (1 << mega.MEGA_MOE_EVENT_NAMES.index('kernel')) |
            (1 << mega.MEGA_MOE_EVENT_NAMES.index('task')) |
            (1 << mega.MEGA_MOE_EVENT_NAMES.index('wgmma.issue'))
        )
        assert profiler.event_mask == expected_mask
        configured_bit = 1 << 63
        assert int(profiler.buffer[0, 4, 0, 1]) & configured_bit
        assert int(profiler.buffer[0, 4, 0, 1]) & (configured_bit - 1) == expected_mask
        assert int(profiler.buffer[0, 0, 0, 1]) & configured_bit
        assert int(profiler.buffer[0, 0, 0, 1]) & (configured_bit - 1) == 0
        assert profiler.cta_indices == (0,)
        assert profiler.warp_indices == (4,)
        with pytest.raises(ValueError, match=r'cta_indices must be in'):
            mega.MegaMoeProfiler(
                capacity=1, device='cpu', cta_indices=[profiler.num_ctas])
        with pytest.raises(ValueError, match=r'warp_indices must contain'):
            mega.MegaMoeProfiler(
                capacity=1, device='cpu', warp_indices=[])

        scatter_profiler = mega.MegaMoeProfiler(
            capacity=1, device='cpu', event_types=['nvlink.scatter'],
            cta_indices=[0], warp_indices=[4])
        event_index = mega.MEGA_MOE_EVENT_NAMES.index
        route_mask = (
            (1 << event_index('dispatch.pull')) |
            (1 << event_index('dispatch.select'))
        )
        scatter_mask = (
            route_mask |
            (1 << event_index('kernel')) |
            (1 << event_index('task')) |
            (1 << event_index('nvlink.scatter'))
        )
        low_bits = configured_bit - 1
        assert int(scatter_profiler.buffer[0, 4, 0, 1]) & low_bits == scatter_mask
        assert int(scatter_profiler.buffer[-1, 0, 0, 1]) & low_bits == route_mask
        assert int(scatter_profiler.buffer[-1, 1, 0, 1]) & low_bits == route_mask
        assert int(scatter_profiler.buffer[-1, 4, 0, 1]) & low_bits == 0
        assert scatter_profiler.track_dependencies == ({
            'reason': 'nvlink.scatter_route_join',
            'event_types': ('dispatch.select', 'dispatch.pull'),
            'cta_indices': 'all',
            'warp_indices': (0, 1),
        },)

        args = {
            'coordinates_valid': True, 'phase': 'linear1',
            'expert_kind': 'routed', 'global_expert': 3, 'local_expert': 3,
            'm_start': 0, 'm_end': 8, 'n_start': 128, 'n_end': 256,
            'k_start': 0, 'k_end': 128, 'stage': 1, 'expanded_slot': 0,
            'activation_sf_group': 0, 'output_n_start': 64,
            'output_n_end': 128, 'num_tokens': 8, 'num_experts': 8,
            'hidden': 7168, 'token_count': 8, 'route_count': 48,
            'dst_rank_route_counts': 'R0:48',
            'workload_scope': 'publish_expert_counts',
            'dst_rank_expert_counts': 'R0:8', 'participants': 2,
            'participating_ranks': 8, 'next_task': 'L1 E3 M[0,8)',
            'src_rank': 0, 'src_token': 1, 'src_topk': 2,
            'dst_global_expert': 3, 'dst_m': 0, 'barrier': 'compute_stage_full',
            'output_ranks': '0', 'scatter_route_count': 8,
            'token_start': 0, 'token_stride': 8, 'token': 1,
            'topk_slot': 2, 'hidden_start': 0, 'hidden_end': 3584,
        }
        for name in mega.MEGA_MOE_EVENT_NAMES:
            event = {
                'name': name, 'role': 'math_epilogue.0',
                'cta': 0, 'warp': 4, 'args': dict(args),
            }
            assert profiler._display_name(event) != name
    finally:
        _unload_fake_package(package_name)


def test_in_kernel_profiler_does_not_call_truncated_wait_end_of_stream():
    mega, package_name = _load_mega_api_for_cpu()
    try:
        profiler = mega.MegaMoeProfiler(capacity=2, device='cpu')
        profiler.buffer[0, 4, 0, 0] = 3
        for record, phase in enumerate((0, 1), start=1):
            profiler.buffer[0, 4, record, 0] = record * 100
            profiler.buffer[0, 4, record, 1] = 8 | (phase << 8)

        waits = profiler.events()
        assert len(waits) == 2
        assert all(
            event['args']['next_task'] == 'unknown_profiler_truncated'
            for event in waits)
        assert all(
            event['args']['next_task_status'] == 'profiler_truncated'
            for event in waits)
    finally:
        _unload_fake_package(package_name)


def test_in_kernel_profiler_paths_alignment_and_merge(tmp_path):
    mega, package_name = _load_mega_api_for_cpu()
    try:
        with pytest.raises(ValueError, match='capacity must be positive'):
            mega.MegaMoeProfiler(capacity=0, device='cpu')

        trace_template = tmp_path / 'trace.{rank}.json'
        paths = []
        for rank, (timestamp, host_range) in enumerate((
                (1000, (10000, 14000)),
                (2000, (11000, 15000)))):
            profiler = mega.MegaMoeProfiler(capacity=2, rank=rank, device='cpu')
            profiler.buffer[0, 0, 0, 0] = 3 if rank == 1 else 2
            profiler.buffer[0, 0, 1, 0] = timestamp
            profiler.buffer[0, 0, 1, 1] = 0
            profiler.buffer[0, 0, 2, 0] = timestamp + 1000
            profiler.buffer[0, 0, 2, 1] = 1 << 8
            path = mega.mega_moe_rank_trace_path(trace_template, rank, 2)
            profiler.export_chrome_trace(
                path, host_time_range_ns=host_range)
            paths.append(path)

        assert paths == [
            tmp_path / 'trace.0.json', tmp_path / 'trace.1.json']
        assert mega.mega_moe_rank_trace_path(
            tmp_path / 'plain', 1, 2) == tmp_path / 'plain.rank1.json'
        assert mega.mega_moe_merged_trace_path(
            tmp_path / 'plain', 2) == tmp_path / 'plain.json'
        merged_path = mega.mega_moe_merged_trace_path(trace_template, 2)
        mega.merge_mega_moe_chrome_traces(paths, merged_path)
        merged = __import__('json').loads(merged_path.read_text())
        timed_events = [
            event for event in merged['traceEvents'] if 'ts' in event]
        assert [event['pid'] for event in timed_events] == [0, 0, 1, 1]
        assert [event['ts'] for event in timed_events] == [0.0, 1.0, 1.0, 2.0]
        assert merged['deep_gemm']['format'] == 'mega_moe_in_kernel_merged_v5'
        assert merged['deep_gemm']['num_ranks'] == 2
        assert merged['deep_gemm']['max_uncertainty_ns'] == 1500
        assert merged['deep_gemm']['truncation'] == {
            'attempted': 5,
            'retained': 4,
            'dropped': 1,
            'truncated_tracks': 1,
            'counts_are_lower_bounds': True,
        }
    finally:
        _unload_fake_package(package_name)


def test_in_kernel_profiler_marks_incomplete_kernel_alignment_unreliable(tmp_path):
    mega, package_name = _load_mega_api_for_cpu()
    try:
        profiler = mega.MegaMoeProfiler(capacity=1, device='cpu')
        profiler.buffer[0, 4, 0, 0] = 2
        profiler.buffer[0, 4, 1, 0] = 1000
        profiler.buffer[0, 4, 1, 1] = 0
        path = profiler.export_chrome_trace(
            tmp_path / 'incomplete.json', host_time_range_ns=(10000, 20000))
        trace = __import__('json').loads(path.read_text())
        alignment = trace['deep_gemm']['clock_alignment']
        assert alignment['mode'] == (
            'host_monotonic_midpoint_incomplete_kernel')
        assert alignment['kernel_interval_complete'] is False
        assert alignment['uncertainty_ns'] == 5000
        with pytest.raises(ValueError, match='complete kernel intervals'):
            mega.merge_mega_moe_chrome_traces([path, path], tmp_path / 'bad.json')
    finally:
        _unload_fake_package(package_name)


def test_in_kernel_profiler_validates_rank_device_and_sm_configuration():
    mega, package_name = _load_mega_api_for_cpu()

    class FakeGroup:
        @staticmethod
        def size():
            return 1

        @staticmethod
        def rank():
            return 0

    buffer = mega.SM90SymmBuffer.__new__(mega.SM90SymmBuffer)
    buffer.group = FakeGroup()
    buffer.num_shared_experts = 0
    y = torch.empty(1)
    try:
        wrong_rank = mega.MegaMoeProfiler(capacity=1, rank=1, device='cpu')
        with pytest.raises(ValueError, match='does not match process-group rank'):
            mega.fp8_mxfp4_mega_moe(y, (), (), buffer, profiler=wrong_rank)

        wrong_device = mega.MegaMoeProfiler(capacity=1, rank=0, device='cpu')
        wrong_device.buffer = torch.empty(
            wrong_device.buffer.shape, dtype=torch.int64, device='meta')
        with pytest.raises(ValueError, match='does not match output device'):
            mega.fp8_mxfp4_mega_moe(y, (), (), buffer, profiler=wrong_device)

        wrong_num_ctas = mega.MegaMoeProfiler(
            capacity=1, rank=0, device='cpu')
        wrong_num_ctas.num_ctas += 1
        with pytest.raises(ValueError, match='configured SM count'):
            mega.fp8_mxfp4_mega_moe(
                y, (), (), buffer, profiler=wrong_num_ctas)
    finally:
        _unload_fake_package(package_name)


def test_explicit_wrapper_rejects_wrong_local_expert_shard_before_launch():
    mega, package_name = _load_mega_api_for_cpu()

    class FakeGroup:
        @staticmethod
        def size():
            return 2

    buffer = mega.SM90SymmBuffer.__new__(mega.SM90SymmBuffer)
    buffer.group = FakeGroup()
    buffer.num_experts = 4
    checkpoint_l1, checkpoint_l2 = _valid_checkpoint_weights(num_experts=1)
    transformed_l1, transformed_l2 = (
        mega.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            checkpoint_l1, checkpoint_l2))
    try:
        with pytest.raises(ValueError, match='expected E_local=2, got 1'):
            mega.fp8_mxfp4_mega_moe(
                None, transformed_l1, transformed_l2, buffer)
    finally:
        _unload_fake_package(package_name)


def test_explicit_wrapper_forwards_only_processed_triples():
    mega, package_name = _load_mega_api_for_cpu()
    calls = []

    class FakeGroup:
        @staticmethod
        def size():
            return 1

        @staticmethod
        def rank():
            return 0

    buffer = mega.SM90SymmBuffer.__new__(mega.SM90SymmBuffer)
    buffer.group = FakeGroup()
    buffer.num_experts = 1
    buffer.num_max_tokens_per_rank = 128
    buffer.num_topk = 1
    buffer.hidden = 512
    buffer.intermediate_hidden = 256
    buffer.num_shared_experts = 0
    buffer.buffer = object()
    buffer.handle = types.SimpleNamespace(buffer_ptrs=[0x1234])
    checkpoint_l1, checkpoint_l2 = _valid_checkpoint_weights()
    processed_l1, processed_l2 = (
        mega.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            checkpoint_l1, checkpoint_l2))
    mega._C = types.SimpleNamespace(
        get_num_sms=lambda: 1,
        fp8_mxfp4_mega_moe=lambda *args: calls.append(args))
    y = torch.empty(1)
    profiler = mega.MegaMoeProfiler(capacity=1, rank=0, device='cpu')
    try:
        mega.fp8_mxfp4_mega_moe(
            y, processed_l1, processed_l2, buffer,
            recipe=(1, 1, 32), activation='swiglu',
            activation_clamp=10.0, fast_math=False,
            fp8_scale_mode='per_tensor',
            activation_dequant_scales=(0.5, 0.25),
            profiler=profiler)
    finally:
        _unload_fake_package(package_name)

    assert len(calls) == 1
    args = calls[0]
    assert len(args) == 19
    assert args[0] is y
    assert args[1] is processed_l1 and len(args[1]) == 3
    assert args[2] is processed_l2 and len(args[2]) == 3
    assert args[3] is None and args[4] is None
    assert args[7] == [0x1234]
    assert args[12] == (1, 1, 32)
    assert args[13:-1] == (
        'swiglu', 10.0, False, 'per_tensor', (0.5, 0.25))
    assert args[-1] is profiler.buffer


def test_sm90_buffer_uses_dedicated_alignment_and_twelve_view_abi():
    mega, package_name = _load_mega_api_for_cpu()
    sizing_calls = []

    class FakeBuffer:
        def __init__(self):
            self.zeroed = False

        def data_ptr(self):
            return 0x1234

        def zero_(self):
            self.zeroed = True
            return self

    class FakeGroup:
        def __init__(self):
            self.barriers = 0

        def size(self):
            return 1

        def rank(self):
            return 0

        def barrier(self):
            self.barriers += 1

    views = tuple(object() for _ in range(12))

    def get_size(*args):
        sizing_calls.append(args)
        return 4096, lambda _: views

    fake_torch = types.SimpleNamespace(
        int8=torch.int8,
        empty=lambda *args, **kwargs: FakeBuffer(),
        cuda=types.SimpleNamespace(synchronize=lambda: None),
    )
    mega.torch = fake_torch
    mega._C = types.SimpleNamespace(
        get_token_alignment_for_sm90_mega_moe=lambda: 128,
        get_symm_buffer_size_for_sm90_mega_moe=get_size,
    )
    group = FakeGroup()
    try:
        buffer = mega.get_symm_buffer_for_sm90_mega_moe(
            group,
            num_experts=16,
            num_max_tokens_per_rank=17,
            num_topk=2,
            hidden=512,
            intermediate_hidden=256,
        )
        assert sizing_calls == [(
            1, 16, 128, 2, 512, 256, True, 'swiglu', 0)]
        assert buffer.num_max_tokens_per_rank == 128
        assert group.barriers == 1
        assert (
            buffer.x, buffer.x_sf,
            buffer.topk_idx, buffer.topk_weights,
            buffer.shared_l1_acts, buffer.shared_l1_acts_sf,
            buffer.shared_l2_acts, buffer.shared_l2_acts_sf,
            buffer.l1_acts, buffer.l1_acts_sf,
            buffer.l2_acts, buffer.l2_acts_sf,
        ) == views
        buffer.destroy()
        assert all(getattr(buffer, name) is None for name in (
            'x', 'x_sf', 'topk_idx', 'topk_weights',
            'shared_l1_acts', 'shared_l1_acts_sf',
            'shared_l2_acts', 'shared_l2_acts_sf',
            'l1_acts', 'l1_acts_sf', 'l2_acts', 'l2_acts_sf'))
        shared_buffer = mega.get_symm_buffer_for_sm90_mega_moe(
            group, 16, 128, 2, 512, 256, num_shared_experts=1)
        assert shared_buffer.num_shared_experts == 1
        assert sizing_calls[-1] == (
            1, 16, 128, 2, 512, 256, True, 'swiglu', 1)
        shared_buffer.destroy()
    finally:
        _unload_fake_package(package_name)


def test_sm90_shared_weight_transform_interleaves_only_l1_fp8_rows():
    mega, package_name = _load_mega_api_for_cpu()
    try:
        row_values = (
            torch.arange(256, dtype=torch.float32) % 16
        ).to(torch.float8_e4m3fn)
        l1_weight = row_values[:, None].expand(256, 128).contiguous()
        l2_weight = torch.ones(
            (128, 128), dtype=torch.float8_e4m3fn)
        l1_scale = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        l2_scale = torch.tensor([[3.0]], dtype=torch.float32)

        transformed_l1, transformed_l2 = (
            mega.transform_shared_weights_for_fp8_mxfp4_mega_moe_sm90(
                (l1_weight, l1_scale), (l2_weight, l2_scale)))

        expected_rows = torch.cat((row_values[:8], row_values[128:136]))
        assert torch.equal(transformed_l1[0][:16, 0], expected_rows)
        assert transformed_l1[1].data_ptr() == l1_scale.data_ptr()
        assert transformed_l2[0].data_ptr() == l2_weight.data_ptr()
        assert transformed_l2[1].data_ptr() == l2_scale.data_ptr()
    finally:
        _unload_fake_package(package_name)
