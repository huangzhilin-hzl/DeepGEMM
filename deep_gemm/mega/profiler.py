"""In-kernel timeline profiling for the SM90 persistent MegaMoE kernel."""

from __future__ import annotations

import json
from collections import defaultdict
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
    "wgmma.issue",
    "wgmma.wait",
    "scale.promote",
}
_COMBINE_PAYLOAD_EVENTS = {
    "combine.tma_issue",
    "combine.tma_wait",
    "combine.reduce",
    "combine.tma_store",
}


def _decode_payload(event_name: str, payload: int) -> Dict[str, Any]:
    args: Dict[str, Any] = {"payload": payload}
    if event_name in _TASK_PAYLOAD_EVENTS:
        phase = payload & 0x7
        args.update(
            phase=_BLOCK_PHASE.get(phase, f"unknown_{phase}"),
            expert=(payload >> 3) & 0x1FF,
            m_block=(payload >> 12) & 0x3FF,
            n_block=(payload >> 22) & 0x3FF,
        )
    elif event_name in _PIPELINE_PAYLOAD_EVENTS:
        phase = payload & 0x7
        args.update(
            phase=_BLOCK_PHASE.get(phase, f"unknown_{phase}"),
            stage=(payload >> 3) & 0x3,
            k_block=(payload >> 5) & 0xFFF,
            auxiliary=(payload >> 17) & 0x7FFF,
        )
    elif event_name == "dispatch.pull":
        args["token"] = payload
    elif event_name in _COMBINE_PAYLOAD_EVENTS:
        args.update(
            token=payload & 0xFFFF,
            chunk=(payload >> 16) & 0xFF,
        )
        if event_name in {
            "combine.tma_issue",
            "combine.tma_wait",
            "combine.reduce",
        }:
            args["slot"] = (payload >> 24) & 0xFF
    return args


class MegaMoeProfiler:
    """Collect and export warp-level SM90 MegaMoE in-kernel events.

    ``capacity`` is the maximum number of events retained per CTA/warp track.
    The device still counts attempted events after the buffer fills, so exported
    traces report truncation instead of silently appearing complete.
    """

    num_warps = len(MEGA_MOE_WARP_ROLES)

    def __init__(
        self,
        capacity: int = 4096,
        *,
        rank: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if rank < 0:
            raise ValueError("rank must be non-negative")
        self.capacity = int(capacity)
        self.rank = int(rank)
        self.num_ctas = 2 * int(_C.get_num_sms())
        self.buffer = torch.zeros(
            (self.num_ctas, self.num_warps, self.capacity + 1, 2),
            dtype=torch.int64,
            device=device or "cuda",
        )
        self._last_export_metadata: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self.buffer.zero_()
        self._last_export_metadata = None

    def _snapshot(self) -> torch.Tensor:
        return self.buffer.detach().cpu()

    def _events_from_snapshot(
        self, snapshot: torch.Tensor
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for cta_idx in range(self.num_ctas):
            for warp_idx, role in enumerate(MEGA_MOE_WARP_ROLES):
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
                    payload = (encoded >> 16) & 0xFFFFFFFF
                    if event_idx >= len(MEGA_MOE_EVENT_NAMES):
                        name = f"unknown.{event_idx}"
                    else:
                        name = MEGA_MOE_EVENT_NAMES[event_idx]
                    result.append(
                        {
                            "timestamp_ns": timestamp,
                            "name": name,
                            "phase": _EVENT_PHASE.get(event_type, "i"),
                            "rank": self.rank,
                            "sm": sm_idx,
                            "cta": cta_idx,
                            "warp": warp_idx,
                            "role": role,
                            "args": _decode_payload(name, payload),
                        }
                    )
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
        return {
            "attempted": int(attempted.sum().item()),
            "retained": int(attempted.clamp_max(self.capacity).sum().item()),
            "dropped": int(dropped.sum().item()),
            "truncated_tracks": int((dropped > 0).sum().item()),
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
        first_timestamp = min(
            (event["timestamp_ns"] for event in events), default=0)
        last_timestamp = max(
            (event["timestamp_ns"] for event in events), default=0)
        if host_time_range_ns is None:
            timestamp_offset_ns = -first_timestamp
            clock_alignment: Dict[str, Any] = {"mode": "rank_relative"}
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
                "mode": "host_monotonic_midpoint",
                "offset_ns": timestamp_offset_ns,
                "uncertainty_ns": max(0, host_span_ns - gpu_span_ns) // 2,
                "host_begin_ns": host_begin_ns,
                "host_end_ns": host_end_ns,
            }
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
                "name": event["name"],
                "cat": event["role"],
                "ph": event["phase"],
                "pid": self.rank,
                "tid": event["cta"] * self.num_warps + event["warp"],
                "ts": (
                    event["timestamp_ns"] + timestamp_offset_ns
                ) / 1000.0,
                "args": {
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
            "format": "mega_moe_in_kernel_v2",
            "rank": self.rank,
            "num_ctas": self.num_ctas,
            "num_warps": self.num_warps,
            "capacity_per_track": self.capacity,
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
            "multi-rank traces must be exported with host_time_range_ns")

    timed_events = [event for event in trace_events if "ts" in event]
    origin_us = min((event["ts"] for event in timed_events), default=0.0)
    for event in timed_events:
        event["ts"] -= origin_us

    truncation_keys = (
        "attempted", "retained", "dropped", "truncated_tracks")
    merged_metadata = {
        "format": "mega_moe_in_kernel_merged_v2",
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
            key: sum(metadata["truncation"][key] for metadata in rank_metadata)
            for key in truncation_keys
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
) -> MegaMoeProfiler:
    return MegaMoeProfiler(capacity=capacity, rank=rank, device=device)
