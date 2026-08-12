"""In-kernel timeline profiling for the SM90 persistent MegaMoE kernel."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .. import _C


MEGA_MOE_EVENT_NAMES = (
    "kernel",
    "setup",
    "dispatch.count",
    "dispatch.pack",
    "dispatch.put",
    "dispatch.pull",
    "dispatch.nvlink",
    "dispatch.cleanup",
    "scheduler.wait",
    "task",
    "tma.activation",
    "tma.activation_scale",
    "tma.weight",
    "weight_scale.load",
    "pipeline.wait",
    "mxfp4.decode",
    "wgmma.issue",
    "scale.promote",
    "epilogue.l1",
    "swiglu.quantize",
    "tma.store_l1",
    "epilogue.l2",
    "nvlink.scatter",
    "l1_dependency.wait",
    "combine.nvlink",
    "combine",
    "combine.tma_issue",
    "combine.reduce",
    "combine.tma_store",
    "grid_barrier",
    "wgmma.wait",
    "combine.tma_wait",
    "dispatch.select",
)

MEGA_MOE_WARP_ROLES = (
    "dispatch.route",
    "dispatch.transfer",
    "producer.activation",
    "producer.weight",
    "math_epilogue.0",
    "math_epilogue.1",
    "math_epilogue.2",
    "math_epilogue.3",
)

_EVENT_PHASE = {0: "B", 1: "E", 2: "i"}
_BLOCK_PHASE = {
    0: "none",
    1: "linear1",
    2: "linear2",
    3: "shared_linear1",
    4: "shared_linear2",
}
_TASK_PAYLOAD_EVENTS = {
    "task",
    "epilogue.l1",
    "swiglu.quantize",
    "tma.store_l1",
    "epilogue.l2",
    "nvlink.scatter",
    "l1_dependency.wait",
}
_PIPELINE_PAYLOAD_EVENTS = {
    "tma.activation",
    "tma.activation_scale",
    "tma.weight",
    "weight_scale.load",
    "pipeline.wait",
    "mxfp4.decode",
    "scale.promote",
}
_WGMMA_PAYLOAD_EVENTS = {"wgmma.issue", "wgmma.wait"}
_COMBINE_PAYLOAD_EVENTS = {
    "combine.tma_issue",
    "combine.tma_wait",
    "combine.reduce",
    "combine.tma_store",
}
_TASK_CONTEXT_EVENTS = _PIPELINE_PAYLOAD_EVENTS | _WGMMA_PAYLOAD_EVENTS
_EVENT_DEPENDENCIES = {
    **{name: {"task"} for name in _TASK_CONTEXT_EVENTS},
    "scheduler.wait": {"task"},
    "dispatch.pull": {"dispatch.select"},
    "nvlink.scatter": {"task", "dispatch.select", "dispatch.pull"},
}
for _event_name in MEGA_MOE_EVENT_NAMES:
    if _event_name != "kernel":
        _EVENT_DEPENDENCIES.setdefault(_event_name, set()).add("kernel")

_BLOCK_M = 64
_BLOCK_N = 128
_BLOCK_K = 128


def _decode_payload(event_name: str, payload: int) -> Dict[str, Any]:
    args: Dict[str, Any] = {"payload": payload}
    if event_name in _TASK_PAYLOAD_EVENTS:
        phase = payload & 0x7
        args.update(
            phase=_BLOCK_PHASE.get(phase, f"unknown_{phase}"),
            local_expert=(payload >> 3) & 0x1FF,
            m_block=(payload >> 12) & 0xFFFF,
            n_block=(payload >> 28) & 0x3FF,
            valid_m=(payload >> 38) & 0x7F,
            payload_overflow=bool((payload >> 47) & 0x1),
        )
        args["expert"] = args["local_expert"]
        args["coordinates_valid"] = not args["payload_overflow"]
    elif event_name in _WGMMA_PAYLOAD_EVENTS:
        phase = payload & 0x7
        args.update(
            phase=_BLOCK_PHASE.get(phase, f"unknown_{phase}"),
            stage=(payload >> 3) & 0x3,
            k_block=(payload >> 5) & 0xFFF,
            k32_start=(payload >> 17) & 0x7,
            k32_count=(payload >> 20) & 0x7,
            expanded_slot=(payload >> 23) & 0x3,
            accumulate=bool((payload >> 25) & 0x1),
        )
    elif event_name in _PIPELINE_PAYLOAD_EVENTS:
        phase = payload & 0x7
        args.update(
            phase=_BLOCK_PHASE.get(phase, f"unknown_{phase}"),
            stage=(payload >> 3) & 0x3,
            k_block=(payload >> 5) & 0xFFF,
            auxiliary=(payload >> 17) & 0x7FFFFFFF,
        )
        if event_name == "mxfp4.decode":
            args["expanded_slot"] = args["auxiliary"]
        elif event_name == "scale.promote":
            args["activation_sf_group"] = args["auxiliary"]
    elif event_name == "dispatch.select":
        args.update(
            dst_local_expert=payload & 0x1FF,
            dst_expert_token=(payload >> 9) & 0xFFFFFFFF,
            payload_overflow=bool((payload >> 47) & 0x1),
        )
        args["coordinates_valid"] = not args["payload_overflow"]
    elif event_name == "dispatch.pull":
        args.update(
            src_token=payload & 0xFFFFFFFF,
            src_topk=(payload >> 32) & 0x1F,
            src_rank=(payload >> 37) & 0x3F,
            payload_overflow=bool((payload >> 47) & 0x1),
        )
        args["coordinates_valid"] = not args["payload_overflow"]
    elif event_name in _COMBINE_PAYLOAD_EVENTS:
        args.update(
            token=payload & 0xFFFFFFFF,
            chunk=(payload >> 32) & 0x7,
            num_chunks=(payload >> 35) & 0x7,
            payload_overflow=bool((payload >> 47) & 0x1),
        )
        args["coordinates_valid"] = not args["payload_overflow"]
        if bool((payload >> 44) & 0x1):
            args["topk_slot"] = (payload >> 38) & 0x3F
    return args


class MegaMoeProfiler:
    """Collect and export warp-level SM90 MegaMoE in-kernel events.

    ``capacity`` is the maximum number of events retained per CTA/warp track.
    Once a track fills, the device records one truncation sentinel and stops
    reading the global timer for that track. Exported attempted/dropped counts
    are therefore lower bounds whenever ``truncated_tracks`` is nonzero.
    """

    num_warps = len(MEGA_MOE_WARP_ROLES)

    def __init__(
        self,
        capacity: int = 4096,
        *,
        rank: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        event_types: Optional[Sequence[str]] = None,
        cta_indices: Optional[Sequence[int]] = None,
        warp_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if rank < 0:
            raise ValueError("rank must be non-negative")
        self.capacity = int(capacity)
        self.rank = int(rank)
        requested = (
            set(MEGA_MOE_EVENT_NAMES) if event_types is None
            else set(event_types)
        )
        unknown = requested - set(MEGA_MOE_EVENT_NAMES)
        if unknown:
            raise ValueError(f"unknown MegaMoE profiler events: {sorted(unknown)}")
        if not requested:
            raise ValueError("event_types must contain at least one event")
        enabled = set(requested)
        pending = list(requested)
        while pending:
            name = pending.pop()
            for dependency in _EVENT_DEPENDENCIES.get(name, ()):
                if dependency not in enabled:
                    enabled.add(dependency)
                    pending.append(dependency)
        self.requested_event_types = tuple(
            name for name in MEGA_MOE_EVENT_NAMES if name in requested
        )
        self.enabled_event_types = tuple(
            name for name in MEGA_MOE_EVENT_NAMES if name in enabled
        )
        self.event_mask = sum(
            1 << MEGA_MOE_EVENT_NAMES.index(name)
            for name in self.enabled_event_types
        )
        self.num_ctas = 2 * int(_C.get_num_sms())
        self.cta_indices = self._normalize_track_indices(
            cta_indices, self.num_ctas, "cta_indices")
        self.warp_indices = self._normalize_track_indices(
            warp_indices, self.num_warps, "warp_indices")
        self._track_event_masks = torch.zeros(
            (self.num_ctas, self.num_warps), dtype=torch.int64)
        for cta_idx in self.cta_indices:
            for warp_idx in self.warp_indices:
                self._track_event_masks[cta_idx, warp_idx] = self.event_mask
        self.track_dependencies: Tuple[Dict[str, Any], ...] = ()

        # Scatter tiles run on math warps, while their destination mapping is
        # established by dispatch producer warps across the persistent grid.
        # Retain only those correlation records on dependency-only tracks.
        if "nvlink.scatter" in requested:
            route_mask = (
                1 << MEGA_MOE_EVENT_NAMES.index("dispatch.select") |
                1 << MEGA_MOE_EVENT_NAMES.index("dispatch.pull")
            )
            self._track_event_masks[:, :2] |= route_mask
            self.track_dependencies = ({
                "reason": "nvlink.scatter_route_join",
                "event_types": ("dispatch.select", "dispatch.pull"),
                "cta_indices": "all",
                "warp_indices": (0, 1),
            },)
        self.buffer = torch.zeros(
            (self.num_ctas, self.num_warps, self.capacity + 1, 2),
            dtype=torch.int64,
            device=device or "cuda",
        )
        self.workload: Optional[Dict[str, Any]] = None
        self._topk_routes: Optional[List[List[int]]] = None
        self._topk_routes_tensor: Optional[torch.Tensor] = None
        self._last_export_metadata: Optional[Dict[str, Any]] = None
        self._arm_buffer()

    def configure_workload(
        self,
        *,
        num_tokens: int,
        num_ranks: int,
        num_experts: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        num_shared_experts: int = 0,
        topk_idx: Optional[torch.Tensor] = None,
    ) -> None:
        """Attach launch dimensions used to expand compact device payloads."""
        if num_ranks <= 0 or num_experts <= 0 or num_experts % num_ranks:
            raise ValueError("invalid profiler rank/expert workload")
        self.workload = {
            "num_tokens": int(num_tokens),
            "num_ranks": int(num_ranks),
            "num_experts": int(num_experts),
            "num_local_experts": int(num_experts // num_ranks),
            "num_topk": int(num_topk),
            "hidden": int(hidden),
            "intermediate_hidden": int(intermediate_hidden),
            "num_shared_experts": int(num_shared_experts),
            "block_m": _BLOCK_M,
            "block_n": _BLOCK_N,
            "block_k": _BLOCK_K,
        }
        self._topk_routes = None
        self._topk_routes_tensor = None
        if topk_idx is not None:
            if topk_idx.ndim != 2 or tuple(topk_idx.shape) != (
                num_tokens, num_topk
            ):
                raise ValueError("topk_idx does not match profiler workload")
            # Keep an asynchronous device snapshot: the symmetric route buffer
            # may be reused before the caller exports this launch's trace.
            self._topk_routes_tensor = topk_idx.detach().clone()

    def _prepare_topk_routes(self) -> None:
        if self._topk_routes is None and self._topk_routes_tensor is not None:
            self._topk_routes = self._topk_routes_tensor.cpu().tolist()

    @staticmethod
    def _normalize_track_indices(
        indices: Optional[Sequence[int]], limit: int, name: str
    ) -> Tuple[int, ...]:
        if indices is None:
            return tuple(range(limit))
        result = tuple(sorted({int(index) for index in indices}))
        if not result:
            raise ValueError(f"{name} must contain at least one index")
        if result[0] < 0 or result[-1] >= limit:
            raise ValueError(f"{name} must be in [0, {limit})")
        return result

    def _arm_buffer(self) -> None:
        # Bit 63 distinguishes an intentionally empty per-track mask from a
        # zero-initialized legacy buffer. int64 stores that bit as the sign bit.
        configured_bit = -(1 << 63)
        headers = self.buffer[:, :, 0, 1]
        headers.copy_(
            self._track_event_masks.to(device=self.buffer.device) +
            configured_bit
        )

    def reset(self) -> None:
        self.buffer.zero_()
        self._arm_buffer()
        self._last_export_metadata = None

    def _snapshot(self) -> torch.Tensor:
        return self.buffer.detach().cpu()

    def _expand_task_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self.workload is None:
            return args
        result = dict(args)
        if not result.get("coordinates_valid", True):
            result["coordinate_status"] = "payload_overflow"
            return result
        phase = result["phase"]
        is_shared = phase.startswith("shared_")
        is_linear1 = phase.endswith("linear1")
        shared_multiplier = (
            self.workload["num_shared_experts"] if is_shared else 1
        )
        shape_n = (
            2 * self.workload["intermediate_hidden"] * shared_multiplier
            if is_linear1
            else self.workload["hidden"]
        )
        shape_k = (
            self.workload["hidden"]
            if is_linear1
            else self.workload["intermediate_hidden"] * shared_multiplier
        )
        m_start = result["m_block"] * _BLOCK_M
        n_start = result["n_block"] * _BLOCK_N
        result.update(
            expert_kind="shared" if is_shared else "routed",
            m_space="rank_token" if is_shared else "expert_packed",
            gemm_coordinate_space="task_m,n,k",
            m_start=m_start,
            m_end=m_start + result["valid_m"],
            m_tile_end=m_start + _BLOCK_M,
            n_start=n_start,
            n_end=min(n_start + _BLOCK_N, shape_n),
            shape_n=shape_n,
            shape_k=shape_k,
            num_k_blocks=(shape_k + _BLOCK_K - 1) // _BLOCK_K,
        )
        if is_linear1:
            result.update(
                n_space="gate_up_interleaved",
                output_n_start=result["n_block"] * (_BLOCK_N // 2),
                output_n_end=min(
                    (result["n_block"] + 1) * (_BLOCK_N // 2),
                    self.workload["intermediate_hidden"] * shared_multiplier,
                ),
            )
        else:
            result["n_space"] = "hidden"
        if not is_shared:
            result["global_expert"] = (
                self.rank * self.workload["num_local_experts"]
                + result["local_expert"]
            )
        return result

    def _dispatch_partition_args(
        self, cta_idx: int, warp_idx: int
    ) -> Dict[str, Any]:
        assert self.workload is not None
        tokens_per_warp = 32 // self.workload["num_topk"]
        token_start = (cta_idx * 2 + warp_idx) * tokens_per_warp
        token_stride = self.num_ctas * 2 * tokens_per_warp
        result: Dict[str, Any] = {
            "workload_scope": "local_token_topk_routes",
            "num_tokens": self.workload["num_tokens"],
            "num_topk": self.workload["num_topk"],
            "token_batch_start": token_start,
            "token_batch_width": tokens_per_warp,
            "token_stride": token_stride,
        }
        if self._topk_routes is None:
            result["route_detail_status"] = "topk_snapshot_unavailable"
            return result

        expert_counts: Counter[int] = Counter()
        rank_counts: Counter[int] = Counter()
        tokens = set()
        for batch_start in range(
            token_start, self.workload["num_tokens"], token_stride
        ):
            for token in range(
                batch_start,
                min(batch_start + tokens_per_warp, self.workload["num_tokens"]),
            ):
                for expert in self._topk_routes[token]:
                    if expert >= 0:
                        tokens.add(token)
                        expert_counts[expert] += 1
                        rank_counts[
                            expert // self.workload["num_local_experts"]
                        ] += 1
        result.update(
            route_count=sum(expert_counts.values()),
            token_count=len(tokens),
            token_first=min(tokens) if tokens else None,
            token_last=max(tokens) if tokens else None,
            global_expert_route_counts=",".join(
                f"E{expert}:{count}"
                for expert, count in sorted(expert_counts.items())
            ),
            dst_rank_route_counts=",".join(
                f"R{rank}:{count}" for rank, count in sorted(rank_counts.items())
            ),
        )
        return result

    @staticmethod
    def _inherit_task_args(
        args: Dict[str, Any], task_args: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if task_args is None:
            return args
        result = dict(args)
        for key, value in task_args.items():
            if key != "payload":
                result.setdefault(key, value)
        return result

    @staticmethod
    def _inherit_route_args(
        args: Dict[str, Any], route_args: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if route_args is None:
            result = dict(args)
            result["coordinates_valid"] = False
            result["coordinate_status"] = "missing_dispatch_select"
            return result
        result = dict(args)
        for key, value in route_args.items():
            if key != "payload":
                result.setdefault(key, value)
        result["coordinates_valid"] = (
            result.get("coordinates_valid", True)
            and route_args.get("coordinates_valid", True)
        )
        return result

    def _enrich_event_args(
        self,
        event_name: str,
        args: Dict[str, Any],
        task_args: Optional[Dict[str, Any]],
        cta_idx: int,
        warp_idx: int,
    ) -> Dict[str, Any]:
        if event_name in _TASK_PAYLOAD_EVENTS:
            args = self._expand_task_args(args)
        elif event_name in _TASK_CONTEXT_EVENTS:
            args = self._inherit_task_args(args, task_args)

        if event_name in _PIPELINE_PAYLOAD_EVENTS and "k_block" in args:
            k_start = args["k_block"] * _BLOCK_K
            k_width = _BLOCK_K
            if event_name == "scale.promote" and args.get("phase", "").endswith("linear2"):
                k_start += args["activation_sf_group"] * (_BLOCK_K // 2)
                k_width = _BLOCK_K // 2
            args["k_start"] = k_start
            args["k_end"] = min(k_start + k_width, args.get("shape_k", k_start + k_width))
        elif event_name in _WGMMA_PAYLOAD_EVENTS:
            k_start = (
                args["k_block"] * _BLOCK_K + args["k32_start"] * 32
            )
            args["k_start"] = k_start
            args["k_end"] = min(
                k_start + args["k32_count"] * 32,
                args.get("shape_k", k_start + args["k32_count"] * 32),
            )

        if event_name in {"epilogue.l1", "swiglu.quantize", "tma.store_l1"}:
            args["output_n_start"] = args.get("output_n_start", 0)
            args["output_n_end"] = args.get("output_n_end", 0)

        workload = self.workload
        if workload is None:
            return args
        if event_name in {"kernel", "setup"}:
            args.update(workload)
        elif event_name in {"dispatch.count", "dispatch.pack"}:
            args.update(self._dispatch_partition_args(cta_idx, warp_idx))
        elif event_name == "dispatch.put":
            active_worker = cta_idx == 0
            expert_indices = []
            if active_worker:
                for lane in range(32):
                    expert_indices.extend(
                        range(
                            warp_idx * 32 + lane,
                            workload["num_experts"],
                            256,
                        )
                    )
            rank_counts = Counter(
                expert // workload["num_local_experts"]
                for expert in expert_indices
            )
            args.update(
                workload_scope=(
                    "publish_expert_counts" if active_worker else "sync_only"
                ),
                num_global_experts=workload["num_experts"],
                active_worker=active_worker,
                global_experts=",".join(map(str, sorted(expert_indices))),
                dst_rank_expert_counts=",".join(
                    f"R{rank}:{count}"
                    for rank, count in sorted(rank_counts.items())
                ),
            )
        elif event_name == "grid_barrier":
            args.update(barrier="dispatch_grid_sync", participants=self.num_ctas)
        elif event_name == "dispatch.nvlink":
            args.update(
                barrier="before_dispatch_pull",
                participating_ranks=workload["num_ranks"],
            )
        elif event_name == "dispatch.cleanup":
            args.update(
                workload_scope="workspace_generation_cleanup",
                barrier="after_workspace_clean",
            )
        elif event_name == "scheduler.wait":
            args["workload_scope"] = "next_persistent_task"
        elif event_name in {"dispatch.select", "dispatch.pull"}:
            if args.get("coordinates_valid", True) and (
                "dst_local_expert" in args and "dst_expert_token" in args
            ):
                args.update(
                    dst_rank=self.rank,
                    dst_global_expert=(
                        self.rank * workload["num_local_experts"]
                        + args["dst_local_expert"]
                    ),
                    dst_m=args["dst_expert_token"],
                    dst_m_space="expert_packed",
                )
            else:
                args["coordinate_status"] = "payload_overflow_or_missing_select"
            if event_name == "dispatch.pull" and args.get(
                "coordinates_valid", True
            ):
                args["src_route_index"] = (
                    args["src_token"] * workload["num_topk"]
                    + args["src_topk"]
                )
        elif event_name == "combine.nvlink":
            args.update(
                barrier="before_combine_reduce",
                participating_ranks=workload["num_ranks"],
            )
        elif event_name == "combine":
            epilogue_warp = warp_idx - 4
            args.update(
                workload_scope="local_output_tokens",
                token_start=cta_idx * 4 + epilogue_warp,
                token_stride=self.num_ctas * 4,
                hidden=workload["hidden"],
            )
        elif event_name in _COMBINE_PAYLOAD_EVENTS:
            if not args.get("coordinates_valid", True):
                args["coordinate_status"] = "payload_overflow"
                return args
            num_chunks = args["num_chunks"]
            chunk_width = workload["hidden"] // num_chunks
            args.update(
                hidden_start=args["chunk"] * chunk_width,
                hidden_end=(args["chunk"] + 1) * chunk_width,
            )
        elif event_name == "nvlink.scatter":
            args["destination_mapping"] = "resolved_from_dispatch_pull"

        if event_name == "pipeline.wait":
            args["barrier"] = {
                2: "activation_stage_empty",
                3: "weight_stage_empty",
            }.get(warp_idx, "compute_stage_full")
        return args

    @staticmethod
    def _task_identity(args: Dict[str, Any]) -> str:
        if not args.get("coordinates_valid", True):
            return f"{args.get('phase', 'task')} coordinates=INVALID"
        phase = {
            "linear1": "L1",
            "linear2": "L2",
            "shared_linear1": "SL1",
            "shared_linear2": "SL2",
        }.get(args.get("phase"), str(args.get("phase", "task")))
        expert = (
            "shared" if args.get("expert_kind") == "shared"
            else f"E{args.get('global_expert', args.get('local_expert', '?'))}"
        )
        return (
            f"{phase} {expert} "
            f"M[{args.get('m_start', '?')},{args.get('m_end', '?')}) "
            f"N[{args.get('n_start', '?')},{args.get('n_end', '?')})"
        )

    def _annotate_scheduler_waits(
        self, track_events: List[Dict[str, Any]], *, truncated: bool
    ) -> None:
        pending: List[Dict[str, Any]] = []

        def finish(
            next_task: Optional[Dict[str, Any]], *, truncated_tail: bool = False
        ) -> None:
            if not pending:
                return
            if next_task is None:
                values = {
                    "next_task": (
                        "unknown_profiler_truncated"
                        if truncated_tail else "end_of_stream"
                    ),
                    "task_available": False,
                    "next_task_status": (
                        "profiler_truncated"
                        if truncated_tail else "end_of_stream"
                    ),
                }
            else:
                task_args = next_task["args"]
                values = {
                    "next_task": self._task_identity(task_args),
                    "task_available": True,
                    "next_task_status": "resolved",
                    "next_phase": task_args.get("phase"),
                    "next_global_expert": task_args.get("global_expert"),
                    "next_local_expert": task_args.get("local_expert"),
                    "next_m_start": task_args.get("m_start"),
                    "next_m_end": task_args.get("m_end"),
                    "next_n_start": task_args.get("n_start"),
                    "next_n_end": task_args.get("n_end"),
                }
            for wait_event in pending:
                wait_event["args"].update(values)
            pending.clear()

        for event in track_events:
            if event["name"] == "scheduler.wait":
                if event["phase"] == "B":
                    finish(None)
                pending.append(event)
            elif event["name"] == "task" and event["phase"] == "B":
                finish(event)
        finish(None, truncated_tail=truncated)

    def _correlate_scatter_routes(
        self, events: List[Dict[str, Any]]
    ) -> None:
        route_map: Dict[Tuple[int, int], Dict[str, int]] = {}
        for event in events:
            args = event["args"]
            if (
                event["name"] == "dispatch.pull"
                and event["phase"] == "B"
                and args.get("coordinates_valid", False)
            ):
                route_map[(args["dst_local_expert"], args["dst_m"])] = {
                    "expert_m": args["dst_m"],
                    "output_rank": args["src_rank"],
                    "output_token": args["src_token"],
                    "output_topk": args["src_topk"],
                }

        for event in events:
            if event["name"] != "nvlink.scatter":
                continue
            args = event["args"]
            if not args.get("coordinates_valid", True) or "m_start" not in args:
                args.update(
                    scatter_routes_valid=False,
                    scatter_route_status="invalid_task_coordinates",
                )
                continue
            task_m_start = args["m_start"]
            task_m_end = args["m_end"]
            warp_stripe_start = task_m_start + max(0, event["warp"] - 4) * 16
            scatter_m_start = min(task_m_end, warp_stripe_start)
            scatter_m_end = min(task_m_end, warp_stripe_start + 16)
            args.update(
                task_m_start=task_m_start,
                task_m_end=task_m_end,
                m_start=scatter_m_start,
                m_end=scatter_m_end,
                scatter_m_start=scatter_m_start,
                scatter_m_end=scatter_m_end,
            )
            routes: List[Dict[str, int]] = []
            missing = 0
            for expert_m in range(scatter_m_start, scatter_m_end):
                if args.get("expert_kind") == "shared":
                    routes.append(
                        {
                            "expert_m": expert_m,
                            "output_rank": self.rank,
                            "output_token": expert_m,
                            "output_topk": self.workload["num_topk"],
                        }
                    )
                else:
                    route = route_map.get((args["local_expert"], expert_m))
                    if route is None:
                        missing += 1
                    else:
                        routes.append(route)
            args.update(
                scatter_routes_valid=missing == 0,
                scatter_route_count=len(routes),
                scatter_missing_count=missing,
                output_ranks=",".join(
                    map(str, sorted({route["output_rank"] for route in routes}))
                ),
                output_tokens=",".join(
                    map(str, sorted({route["output_token"] for route in routes}))
                ),
                output_topk_slots=",".join(
                    map(str, sorted({route["output_topk"] for route in routes}))
                ),
            )
            if event["phase"] == "B":
                args["scatter_routes"] = ";".join(
                    f"m{route['expert_m']}->r{route['output_rank']}:"
                    f"t{route['output_token']}:k{route['output_topk']}"
                    for route in routes
                )

    def _display_name(self, event: Dict[str, Any]) -> str:
        name = event["name"]
        args = event["args"]
        role = event["role"]
        task = self._task_identity(args)
        if name == "kernel":
            return (
                f"kernel {role} T{args.get('num_tokens', '?')} "
                f"E{args.get('num_experts', '?')} H{args.get('hidden', '?')}"
            )
        if name == "setup":
            return f"setup {role} CTA{event['cta']}"
        if name in {"dispatch.count", "dispatch.pack"}:
            operation = "count" if name.endswith("count") else "pack"
            return (
                f"dispatch.{operation} tokens={args.get('token_count', '?')} "
                f"routes={args.get('route_count', '?')} "
                f"{args.get('dst_rank_route_counts', '')}"
            ).rstrip()
        if name == "dispatch.put":
            return (
                f"dispatch.put {args.get('workload_scope')} "
                f"{args.get('dst_rank_expert_counts', '')}"
            ).rstrip()
        if name == "dispatch.select":
            if not args.get("coordinates_valid", True):
                return "dispatch.select coordinates=INVALID"
            return (
                f"dispatch.select E{args.get('dst_global_expert', '?')} "
                f"M{args.get('dst_m', '?')}"
            )
        if name == "dispatch.pull":
            if not args.get("coordinates_valid", True):
                return "dispatch.pull coordinates=INVALID"
            return (
                f"dispatch.pull r{args['src_rank']}:t{args['src_token']}:"
                f"k{args['src_topk']} -> E{args['dst_global_expert']}:"
                f"M{args['dst_m']}"
            )
        if name == "grid_barrier":
            return f"grid_barrier dispatch CTAs={args.get('participants', '?')}"
        if name == "dispatch.nvlink":
            return f"dispatch.nvlink before_pull ranks={args.get('participating_ranks', '?')}"
        if name == "dispatch.cleanup":
            return f"dispatch.cleanup CTA{event['cta']} generation workspace"
        if name == "scheduler.wait":
            return f"scheduler.wait -> {args.get('next_task', 'unknown_task')}"
        if name == "task":
            return f"task {task}"
        if name == "l1_dependency.wait":
            return f"l1_dependency.wait {task}"
        if name == "pipeline.wait":
            return (
                f"pipeline.wait {args.get('barrier', '?')} {task} "
                f"K[{args.get('k_start', '?')},{args.get('k_end', '?')}) "
                f"S{args.get('stage', '?')}"
            )
        if name in {"tma.activation", "tma.activation_scale"}:
            return (
                f"{name} {task} K[{args.get('k_start', '?')},"
                f"{args.get('k_end', '?')}) S{args.get('stage', '?')}"
            )
        if name in {"tma.weight", "weight_scale.load"}:
            return (
                f"{name} {task} K[{args.get('k_start', '?')},"
                f"{args.get('k_end', '?')}) S{args.get('stage', '?')}"
            )
        if name == "mxfp4.decode":
            return (
                f"mxfp4.decode {task} K[{args.get('k_start', '?')},"
                f"{args.get('k_end', '?')}) Bslot{args.get('expanded_slot', '?')}"
            )
        if name in _WGMMA_PAYLOAD_EVENTS:
            return (
                f"{name} {task} K[{args.get('k_start', '?')},"
                f"{args.get('k_end', '?')}) S{args.get('stage', '?')} "
                f"Bslot{args.get('expanded_slot', '?')}"
            )
        if name == "scale.promote":
            return (
                f"scale.promote {task} K[{args.get('k_start', '?')},"
                f"{args.get('k_end', '?')}) SF{args.get('activation_sf_group', '?')}"
            )
        if name in {"epilogue.l1", "swiglu.quantize", "tma.store_l1"}:
            return (
                f"{name} {task} outN[{args.get('output_n_start', '?')},"
                f"{args.get('output_n_end', '?')})"
            )
        if name == "epilogue.l2":
            return f"epilogue.l2 {task}"
        if name == "nvlink.scatter":
            return (
                f"nvlink.scatter {task} -> ranks[{args.get('output_ranks', '?')}] "
                f"routes={args.get('scatter_route_count', '?')}"
            )
        if name == "combine.nvlink":
            return f"combine.nvlink before_reduce ranks={args.get('participating_ranks', '?')}"
        if name == "combine":
            return (
                f"combine tokens={args.get('token_start', '?')}+n*"
                f"{args.get('token_stride', '?')} H{args.get('hidden', '?')}"
            )
        if name in _COMBINE_PAYLOAD_EVENTS:
            slot = (
                f" slot{args['topk_slot']}" if "topk_slot" in args else ""
            )
            return (
                f"{name} token{args.get('token', '?')}{slot} "
                f"H[{args.get('hidden_start', '?')},{args.get('hidden_end', '?')})"
            )
        return f"{name} {role} CTA{event['cta']} W{event['warp']}"

    def _events_from_snapshot(
        self, snapshot: torch.Tensor
    ) -> List[Dict[str, Any]]:
        self._prepare_topk_routes()
        result: List[Dict[str, Any]] = []
        for cta_idx in range(self.num_ctas):
            for warp_idx, role in enumerate(MEGA_MOE_WARP_ROLES):
                task_args: Optional[Dict[str, Any]] = None
                route_args: Optional[Dict[str, Any]] = None
                track_events: List[Dict[str, Any]] = []
                attempted = int(snapshot[cta_idx, warp_idx, 0, 0].item())
                placement = int(snapshot[cta_idx, warp_idx, 0, 1].item())
                retained = min(attempted, self.capacity)
                sm_idx = placement & 0xFF
                for record_idx in range(retained):
                    timestamp = int(
                        snapshot[cta_idx, warp_idx, record_idx + 1, 0].item()
                    )
                    encoded = int(
                        snapshot[cta_idx, warp_idx, record_idx + 1, 1].item()
                    )
                    event_idx = encoded & 0xFF
                    event_type = (encoded >> 8) & 0x3
                    event_phase = _EVENT_PHASE.get(event_type, "i")
                    payload = (encoded >> 16) & 0xFFFFFFFFFFFF
                    if event_idx >= len(MEGA_MOE_EVENT_NAMES):
                        name = f"unknown.{event_idx}"
                    else:
                        name = MEGA_MOE_EVENT_NAMES[event_idx]
                    args = _decode_payload(name, payload)
                    if name == "dispatch.pull":
                        args = self._inherit_route_args(args, route_args)
                    args = self._enrich_event_args(
                        name, args, task_args, cta_idx, warp_idx
                    )
                    event = {
                        "timestamp_ns": timestamp,
                        "name": name,
                        "phase": event_phase,
                        "rank": self.rank,
                        "sm": sm_idx,
                        "cta": cta_idx,
                        "warp": warp_idx,
                        "role": role,
                        "args": args,
                    }
                    track_events.append(event)
                    if name == "task" and event["phase"] == "B":
                        task_args = args
                    elif name == "task" and event["phase"] == "E":
                        task_args = None
                    if name == "dispatch.select" and event["phase"] == "B":
                        route_args = args
                    elif name == "dispatch.pull" and event["phase"] == "E":
                        route_args = None
                self._annotate_scheduler_waits(
                    track_events, truncated=attempted > self.capacity)
                result.extend(track_events)
        self._correlate_scatter_routes(result)
        for event in result:
            event["display_name"] = self._display_name(event)
            event["args"]["workload_identity"] = event["display_name"]
        result.sort(key=lambda event: event["timestamp_ns"])
        return result

    def events(self) -> List[Dict[str, Any]]:
        """Return decoded events sorted on the device global-timer axis."""
        return self._events_from_snapshot(self._snapshot())

    def _truncation_from_snapshot(
        self, snapshot: torch.Tensor
    ) -> Dict[str, int]:
        attempted = snapshot[:, :, 0, 0].clamp_min(0)
        dropped = (attempted - self.capacity).clamp_min(0)
        truncated_tracks = int((dropped > 0).sum().item())
        return {
            "attempted": int(attempted.sum().item()),
            "retained": int(attempted.clamp_max(self.capacity).sum().item()),
            "dropped": int(dropped.sum().item()),
            "truncated_tracks": truncated_tracks,
            "counts_are_lower_bounds": truncated_tracks > 0,
        }

    def truncation(self) -> Dict[str, int]:
        return self._truncation_from_snapshot(self._snapshot())

    @staticmethod
    def _summary_from_events(
        events: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        stacks: Dict[tuple, List[int]] = defaultdict(list)
        durations: Dict[str, List[int]] = defaultdict(list)
        for event in events:
            key = (
                event["cta"],
                event["warp"],
                event["name"],
                event["args"]["payload"],
            )
            if event["phase"] == "B":
                stacks[key].append(event["timestamp_ns"])
            elif event["phase"] == "E" and stacks[key]:
                durations[event["name"]].append(
                    event["timestamp_ns"] - stacks[key].pop()
                )
        return {
            name: {
                "count": len(values),
                "total_us": sum(values) / 1000.0,
                "mean_us": sum(values) / len(values) / 1000.0,
                "max_us": max(values) / 1000.0,
            }
            for name, values in sorted(durations.items())
            if values
        }

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Aggregate paired range durations by stable event name."""
        return self._summary_from_events(self.events())

    @property
    def last_export_metadata(self) -> Dict[str, Any]:
        """Metadata computed from the most recent export snapshot."""
        if self._last_export_metadata is None:
            raise RuntimeError("no trace has been exported")
        return self._last_export_metadata

    def export_chrome_trace(
        self,
        path: Union[str, Path],
        *,
        host_time_range_ns: Optional[Tuple[int, int]] = None,
    ) -> Path:
        """Write a Chrome trace JSON accepted directly by Perfetto UI.

        For multi-rank traces, ``host_time_range_ns`` should bracket the kernel
        launch and synchronization using a host-wide monotonic clock. The
        midpoint maps the GPU global-timer domain onto that common clock. The
        resulting uncertainty is included in the trace metadata.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self._snapshot()
        events = self._events_from_snapshot(snapshot)
        kernel_starts: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        complete_kernel_ranges: List[Tuple[int, int]] = []
        for event in events:
            if event["name"] != "kernel":
                continue
            key = (event["cta"], event["warp"])
            if event["phase"] == "B":
                kernel_starts[key].append(event["timestamp_ns"])
            elif event["phase"] == "E" and kernel_starts[key]:
                complete_kernel_ranges.append((
                    kernel_starts[key].pop(), event["timestamp_ns"]))
        kernel_interval_complete = bool(complete_kernel_ranges)
        if kernel_interval_complete:
            first_timestamp = min(begin for begin, _ in complete_kernel_ranges)
            last_timestamp = max(end for _, end in complete_kernel_ranges)
        else:
            first_timestamp = min(
                (event["timestamp_ns"] for event in events), default=0)
            last_timestamp = max(
                (event["timestamp_ns"] for event in events), default=0)
        if host_time_range_ns is None:
            timestamp_offset_ns = -first_timestamp
            clock_alignment: Dict[str, Any] = {
                "mode": "rank_relative",
                "kernel_interval_complete": kernel_interval_complete,
            }
        else:
            host_begin_ns, host_end_ns = host_time_range_ns
            if host_end_ns < host_begin_ns:
                raise ValueError("host_time_range_ns must be ordered")
            gpu_midpoint_ns = (first_timestamp + last_timestamp) // 2
            host_midpoint_ns = (host_begin_ns + host_end_ns) // 2
            timestamp_offset_ns = host_midpoint_ns - gpu_midpoint_ns
            host_span_ns = host_end_ns - host_begin_ns
            gpu_span_ns = last_timestamp - first_timestamp
            clock_alignment = {
                "mode": (
                    "host_monotonic_midpoint"
                    if kernel_interval_complete else
                    "host_monotonic_midpoint_incomplete_kernel"
                ),
                "offset_ns": timestamp_offset_ns,
                "uncertainty_ns": (
                    max(0, host_span_ns - gpu_span_ns) // 2
                    if kernel_interval_complete else host_span_ns // 2
                ),
                "host_begin_ns": host_begin_ns,
                "host_end_ns": host_end_ns,
                "kernel_interval_complete": kernel_interval_complete,
            }
            if not kernel_interval_complete:
                clock_alignment["warning"] = (
                    "profiler did not retain a complete kernel Begin/End pair"
                )
        trace_events: List[Dict[str, Any]] = [
            {
                "name": "process_name",
                "ph": "M",
                "pid": self.rank,
                "tid": 0,
                "args": {"name": f"MegaMoE rank {self.rank}"},
            }
        ]
        for cta_idx in range(self.num_ctas):
            for warp_idx, role in enumerate(MEGA_MOE_WARP_ROLES):
                tid = cta_idx * self.num_warps + warp_idx
                trace_events.append(
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": self.rank,
                        "tid": tid,
                        "args": {"name": f"CTA {cta_idx} / {role}"},
                    }
                )
        for event in events:
            item = {
                "name": event["display_name"],
                "cat": event["name"],
                "ph": event["phase"],
                "pid": self.rank,
                "tid": event["cta"] * self.num_warps + event["warp"],
                "ts": (
                    event["timestamp_ns"] + timestamp_offset_ns
                ) / 1000.0,
                "args": {
                    "event_type": event["name"],
                    "warp_role": event["role"],
                    "rank": event["rank"],
                    "sm": event["sm"],
                    "cta": event["cta"],
                    "warp": event["warp"],
                    **event["args"],
                },
            }
            if event["phase"] == "i":
                item["s"] = "t"
            trace_events.append(item)

        metadata = {
            "format": "mega_moe_in_kernel_v5",
            "rank": self.rank,
            "num_ctas": self.num_ctas,
            "num_warps": self.num_warps,
            "capacity_per_track": self.capacity,
            "requested_event_types": self.requested_event_types,
            "enabled_event_types": self.enabled_event_types,
            "cta_indices": self.cta_indices,
            "warp_indices": self.warp_indices,
            "track_dependencies": self.track_dependencies,
            "enabled_track_count": int(
                (self._track_event_masks != 0).sum().item()),
            "workload": self.workload,
            "clock_alignment": clock_alignment,
            "truncation": self._truncation_from_snapshot(snapshot),
            "summary": self._summary_from_events(events),
        }
        payload = {
            "traceEvents": trace_events,
            "displayTimeUnit": "ns",
            "deep_gemm": metadata,
        }
        with output_path.open("w", encoding="utf-8") as output:
            json.dump(payload, output, separators=(",", ":"))
        self._last_export_metadata = metadata
        return output_path


def mega_moe_rank_trace_path(
    path: Union[str, Path], rank: int, num_ranks: int
) -> Path:
    path_string = str(path)
    if "{rank}" in path_string:
        return Path(path_string.format(rank=rank))
    output_path = Path(path_string)
    if num_ranks == 1:
        return output_path
    suffix = output_path.suffix or ".json"
    return output_path.with_name(
        f"{output_path.stem}.rank{rank}{suffix}")


def mega_moe_merged_trace_path(
    path: Union[str, Path], num_ranks: int
) -> Path:
    path_string = str(path)
    if num_ranks == 1:
        return mega_moe_rank_trace_path(path_string, 0, 1)
    if "{rank}" in path_string:
        return Path(path_string.format(rank="merged"))
    output_path = Path(path_string)
    return output_path if output_path.suffix else output_path.with_suffix(".json")


def merge_mega_moe_chrome_traces(
    paths: Sequence[Union[str, Path]], output_path: Union[str, Path]
) -> Path:
    """Merge host-aligned rank traces and normalize their shared time axis."""
    if not paths:
        raise ValueError("at least one rank trace is required")
    trace_events: List[Dict[str, Any]] = []
    rank_metadata: List[Dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as trace_file:
            payload = json.load(trace_file)
        trace_events.extend(payload["traceEvents"])
        rank_metadata.append(payload["deep_gemm"])

    if len(rank_metadata) > 1 and any(
        metadata["clock_alignment"]["mode"] != "host_monotonic_midpoint"
        for metadata in rank_metadata
    ):
        raise ValueError(
            "multi-rank traces require complete kernel intervals aligned "
            "with host_time_range_ns")

    timed_events = [event for event in trace_events if "ts" in event]
    origin_us = min((event["ts"] for event in timed_events), default=0.0)
    for event in timed_events:
        event["ts"] -= origin_us

    truncation_keys = (
        "attempted", "retained", "dropped", "truncated_tracks")
    merged_metadata = {
        "format": "mega_moe_in_kernel_merged_v5",
        "num_ranks": len(rank_metadata),
        "clock_alignment": "host_monotonic_midpoint",
        "max_uncertainty_ns": max(
            (
                metadata["clock_alignment"].get("uncertainty_ns", 0)
                for metadata in rank_metadata
            ),
            default=0,
        ),
        "truncation": {
            **{
                key: sum(
                    metadata["truncation"][key]
                    for metadata in rank_metadata
                )
                for key in truncation_keys
            },
            "counts_are_lower_bounds": any(
                metadata["truncation"].get(
                    "counts_are_lower_bounds", False
                )
                for metadata in rank_metadata
            ),
        },
        "ranks": rank_metadata,
    }
    merged_payload = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ns",
        "deep_gemm": merged_metadata,
    }
    merged_path = Path(output_path)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as output:
        json.dump(merged_payload, output, separators=(",", ":"))
    return merged_path


def create_mega_moe_profiler(
    capacity: int = 4096,
    *,
    rank: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    event_types: Optional[Sequence[str]] = None,
    cta_indices: Optional[Sequence[int]] = None,
    warp_indices: Optional[Sequence[int]] = None,
) -> MegaMoeProfiler:
    return MegaMoeProfiler(
        capacity=capacity,
        rank=rank,
        device=device,
        event_types=event_types,
        cta_indices=cta_indices,
        warp_indices=warp_indices,
    )
