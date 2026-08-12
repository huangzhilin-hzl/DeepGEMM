#pragma once

#include <cuda/std/cstdint>

#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::profile {

enum class EventType : uint32_t {
    Begin = 0,
    End = 1,
    Instant = 2,
};

// Keep this list synchronized with MEGA_MOE_EVENT_NAMES in
// deep_gemm/mega/profiler.py. IKET recommends a small stable event vocabulary;
// runtime details belong in the payload rather than dynamically named events.
enum class MegaMoeEvent : uint32_t {
    Kernel = 0,
    Setup,
    DispatchCount,
    DispatchPack,
    DispatchPut,
    DispatchPull,
    DispatchNvlink,
    DispatchCleanup,
    SchedulerWait,
    Task,
    TmaActivation,
    TmaActivationScale,
    TmaWeight,
    LoadWeightScale,
    PipelineWait,
    MXFP4Decode,
    Wgmma,
    ScalePromote,
    EpilogueL1,
    SwigluQuantize,
    TmaStoreL1,
    EpilogueL2,
    NvlinkScatter,
    L1DependencyWait,
    CombineNvlink,
    Combine,
    CombineTmaLoad,
    CombineReduce,
    CombineTmaStore,
    GridBarrier,
    WgmmaWait,
    CombineTmaWait,
    DispatchSelect,
    Count,
};

constexpr uint32_t kMegaMoeNumWarps = 8;

CUTLASS_DEVICE uint64_t globaltimer() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
    return value;
}

constexpr uint64_t kPayloadMask = (1ull << 48) - 1;

constexpr uint64_t encode_event(const MegaMoeEvent event,
                                const EventType type,
                                const uint64_t payload) {
    return static_cast<uint64_t>(event) |
           (static_cast<uint64_t>(type) << 8) |
           ((payload & kPayloadMask) << 16);
}

// Task payload: phase[2:0], expert[11:3], M block[27:12], N block[37:28],
// valid M[44:38], overflow[47]. K-block count is derived from phase/model shape.
constexpr uint64_t encode_task_payload(const uint32_t phase,
                                       const uint32_t expert,
                                       const uint32_t m_block,
                                       const uint32_t n_block,
                                       const uint32_t valid_m) {
    const bool overflow = phase > 0x7u or expert > 0x1ffu or
                          m_block > 0xffffu or n_block > 0x3ffu or
                          valid_m > 0x7fu;
    return static_cast<uint64_t>(phase & 0x7u) |
           (static_cast<uint64_t>(expert & 0x1ffu) << 3) |
           (static_cast<uint64_t>(m_block & 0xffffu) << 12) |
           (static_cast<uint64_t>(n_block & 0x3ffu) << 28) |
           (static_cast<uint64_t>(valid_m & 0x7fu) << 38) |
           (static_cast<uint64_t>(overflow) << 47);
}

// Pipeline payload: phase[2:0], stage[4:3], K block[16:5], auxiliary[47:17].
constexpr uint64_t encode_pipeline_payload(const uint32_t phase,
                                           const uint32_t stage,
                                           const uint32_t k_block,
                                           const uint32_t auxiliary = 0) {
    return static_cast<uint64_t>(phase & 0x7u) |
           (static_cast<uint64_t>(stage & 0x3u) << 3) |
           (static_cast<uint64_t>(k_block & 0xfffu) << 5) |
           (static_cast<uint64_t>(auxiliary & 0x7fffffffu) << 17);
}

// Dispatch selection payload: local expert[8:0], expert-local token[40:9],
// overflow[47]. Token coordinates retain the full uint32 workspace field.
constexpr uint64_t encode_dispatch_select_payload(const uint32_t local_expert,
                                                   const uint32_t expert_token) {
    const bool overflow = local_expert > 0x1ffu;
    return static_cast<uint64_t>(local_expert & 0x1ffu) |
           (static_cast<uint64_t>(expert_token) << 9) |
           (static_cast<uint64_t>(overflow) << 47);
}

// Dispatch pull payload: source token[31:0], source top-k slot[36:32],
// source rank[42:37], overflow[47]. Destination coordinates are inherited
// from the immediately preceding DispatchSelect event on the same warp track.
constexpr uint64_t encode_dispatch_pull_payload(const uint32_t source_rank,
                                                 const uint32_t source_token,
                                                 const uint32_t source_topk) {
    const bool overflow = source_topk > 0x1fu or source_rank > 0x3fu;
    return static_cast<uint64_t>(source_token) |
           (static_cast<uint64_t>(source_topk & 0x1fu) << 32) |
           (static_cast<uint64_t>(source_rank & 0x3fu) << 37) |
           (static_cast<uint64_t>(overflow) << 47);
}

// WGMMA payload: phase[2:0], stage[4:3], K block[16:5], K32 start[19:17],
// K32 count[22:20], expanded-B slot[24:23], accumulate[25].
constexpr uint64_t encode_wgmma_payload(const uint32_t phase,
                                        const uint32_t stage,
                                        const uint32_t k_block,
                                        const uint32_t k32_start,
                                        const uint32_t k32_count,
                                        const uint32_t expanded_slot = 0,
                                        const bool accumulate = false) {
    return static_cast<uint64_t>(phase & 0x7u) |
           (static_cast<uint64_t>(stage & 0x3u) << 3) |
           (static_cast<uint64_t>(k_block & 0xfffu) << 5) |
           (static_cast<uint64_t>(k32_start & 0x7u) << 17) |
           (static_cast<uint64_t>(k32_count & 0x7u) << 20) |
           (static_cast<uint64_t>(expanded_slot & 0x3u) << 23) |
           (static_cast<uint64_t>(accumulate) << 25);
}

// Combine payload: token[31:0], chunk[34:32], chunk count[37:35],
// top-k/shared slot[43:38], has slot[44], overflow[47].
constexpr uint64_t encode_combine_payload(const uint32_t token,
                                          const uint32_t chunk,
                                          const uint32_t num_chunks,
                                          const uint32_t slot = 0,
                                          const bool has_slot = false) {
    const bool overflow = chunk > 0x7u or num_chunks > 0x7u or slot > 0x3fu;
    return static_cast<uint64_t>(token) |
           (static_cast<uint64_t>(chunk & 0x7u) << 32) |
           (static_cast<uint64_t>(num_chunks & 0x7u) << 35) |
           (static_cast<uint64_t>(slot & 0x3fu) << 38) |
           (static_cast<uint64_t>(has_slot) << 44) |
           (static_cast<uint64_t>(overflow) << 47);
}

template <bool kEnabled>
class WarpProfiler {
public:
    CUTLASS_DEVICE WarpProfiler(uint64_t* buffer,
                               const uint32_t capacity,
                               const uint32_t cta_idx,
                               const uint32_t warp_idx,
                               const uint32_t lane_idx) {
        if constexpr (kEnabled) {
            active_ = buffer != nullptr and warp_idx < kMegaMoeNumWarps and
                      lane_idx == 0;
            capacity_ = capacity;
            if (active_) {
                const uint64_t track_stride =
                    static_cast<uint64_t>(capacity + 1) * 2;
                track_ = buffer +
                    (static_cast<uint64_t>(cta_idx) * kMegaMoeNumWarps + warp_idx) *
                        track_stride;
                // Python sets bit 63 when header word 1 contains a configured
                // event mask. This also allows a track to be disabled with an
                // empty mask; unconfigured direct/legacy buffers retain all events.
                constexpr uint64_t kAllEventMask =
                    (1ull << static_cast<uint32_t>(MegaMoeEvent::Count)) - 1;
                constexpr uint64_t kConfiguredMaskBit = 1ull << 63;
                const uint64_t configured_mask = track_[1];
                event_mask_ = (configured_mask & kConfiguredMaskBit) != 0
                                  ? configured_mask & kAllEventMask
                                  : kAllEventMask;
                // Header word 0 is retained count, or capacity + 1 as a
                // truncation sentinel. Word 1 identifies physical placement:
                // SM[7:0], CTA[31:8], warp[39:32].
                track_[0] = 0;
                track_[1] = static_cast<uint64_t>(ptx::get_sm_idx() & 0xffu) |
                            (static_cast<uint64_t>(cta_idx & 0xffffffu) << 8) |
                            (static_cast<uint64_t>(warp_idx & 0xffu) << 32);
            }
        }
    }

    CUTLASS_DEVICE void begin(const MegaMoeEvent event,
                              const uint64_t payload = 0) {
        record(event, EventType::Begin, payload);
    }

    CUTLASS_DEVICE void end(const MegaMoeEvent event,
                            const uint64_t payload = 0) {
        record(event, EventType::End, payload);
    }

    CUTLASS_DEVICE void mark(const MegaMoeEvent event,
                             const uint64_t payload = 0) {
        record(event, EventType::Instant, payload);
    }

private:
    CUTLASS_DEVICE void record(const MegaMoeEvent event,
                               const EventType type,
                               const uint64_t payload) {
        if constexpr (kEnabled) {
            const auto event_idx = static_cast<uint32_t>(event);
            if (not active_ or (event_mask_ & (1ull << event_idx)) == 0)
                return;

            // Once full, publish one truncation sentinel and stop reading the
            // global timer or writing the header on every subsequent event.
            if (count_ >= capacity_) {
                if (count_ == capacity_) {
                    ++ count_;
                    track_[0] = count_;
                }
                return;
            }
            asm volatile("" ::: "memory");
            const uint64_t timestamp = globaltimer();
            auto* record = track_ + 2 + static_cast<uint64_t>(count_) * 2;
            record[0] = timestamp;
            record[1] = encode_event(event, type, payload);
            ++ count_;
            track_[0] = count_;
            asm volatile("" ::: "memory");
        }
    }

    uint64_t* track_ = nullptr;
    uint32_t capacity_ = 0;
    uint32_t count_ = 0;
    uint64_t event_mask_ = ~0ull;
    bool active_ = false;
};

} // namespace deep_gemm::profile
