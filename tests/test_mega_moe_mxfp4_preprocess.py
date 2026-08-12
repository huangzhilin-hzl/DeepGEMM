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
        payload = 1 | (3 << 3) | (4 << 12) | (5 << 22)
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
        assert trace['deep_gemm']['format'] == 'mega_moe_in_kernel_v2'
        assert trace['deep_gemm']['truncation']['truncated_tracks'] == 1
        assert profiler.last_export_metadata['truncation']['dropped'] == 1

        detail_profiler = mega.MegaMoeProfiler(capacity=4, rank=0, device='cpu')
        combine_payload = 9 | (2 << 16) | (3 << 24)
        pipeline_payload = 1 | (2 << 3) | (7 << 5) | (9 << 17)
        detail_profiler.buffer[0, 1, 0, 0] = 4
        detail_profiler.buffer[0, 1, 1, 0] = 100
        detail_profiler.buffer[0, 1, 1, 1] = 5 | (2 << 8) | (123 << 16)
        detail_profiler.buffer[0, 1, 2, 0] = 200
        detail_profiler.buffer[0, 1, 2, 1] = (
            26 | (2 << 8) | (combine_payload << 16)
        )
        detail_profiler.buffer[0, 1, 3, 0] = 300
        detail_profiler.buffer[0, 1, 3, 1] = 18 | (2 << 8) | (payload << 16)
        detail_profiler.buffer[0, 1, 4, 0] = 400
        detail_profiler.buffer[0, 1, 4, 1] = (
            30 | (2 << 8) | (pipeline_payload << 16)
        )
        detail_events = detail_profiler.events()
        assert detail_events[0]['args']['token'] == 123
        assert detail_events[1]['args'] == {
            'payload': combine_payload,
            'token': 9,
            'chunk': 2,
            'slot': 3,
        }
        assert detail_events[2]['name'] == 'epilogue.l1'
        assert detail_events[2]['args']['phase'] == 'linear1'
        assert detail_events[2]['args']['expert'] == 3
        assert detail_events[3]['name'] == 'wgmma.wait'
        assert detail_events[3]['args']['stage'] == 2
        assert detail_events[3]['args']['k_block'] == 7
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
            profiler = mega.MegaMoeProfiler(capacity=1, rank=rank, device='cpu')
            profiler.buffer[0, 0, 0, 0] = 1
            profiler.buffer[0, 0, 1, 0] = timestamp
            profiler.buffer[0, 0, 1, 1] = 2 << 8
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
        assert [event['pid'] for event in timed_events] == [0, 1]
        assert [event['ts'] for event in timed_events] == [0.0, 1.0]
        assert merged['deep_gemm']['format'] == 'mega_moe_in_kernel_merged_v2'
        assert merged['deep_gemm']['num_ranks'] == 2
        assert merged['deep_gemm']['max_uncertainty_ns'] == 2000
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
