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
    Count,
};

constexpr uint32_t kMegaMoeNumWarps = 8;

CUTLASS_DEVICE uint64_t globaltimer() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
    return value;
}

constexpr uint64_t encode_event(const MegaMoeEvent event,
                                const EventType type,
                                const uint32_t payload) {
    return static_cast<uint64_t>(event) |
           (static_cast<uint64_t>(type) << 8) |
           (static_cast<uint64_t>(payload) << 16);
}

// Task payload: phase[2:0], expert[11:3], M block[21:12], N block[31:22].
constexpr uint32_t encode_task_payload(const uint32_t phase,
                                       const uint32_t expert,
                                       const uint32_t m_block,
                                       const uint32_t n_block) {
    return (phase & 0x7u) |
           ((expert & 0x1ffu) << 3) |
           ((m_block & 0x3ffu) << 12) |
           ((n_block & 0x3ffu) << 22);
}

// Pipeline payload: phase[2:0], stage[4:3], K block[16:5], auxiliary[31:17].
constexpr uint32_t encode_pipeline_payload(const uint32_t phase,
                                           const uint32_t stage,
                                           const uint32_t k_block,
                                           const uint32_t auxiliary = 0) {
    return (phase & 0x7u) |
           ((stage & 0x3u) << 3) |
           ((k_block & 0xfffu) << 5) |
           ((auxiliary & 0x7fffu) << 17);
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
                // Header word 0 is attempted event count. Word 1 identifies the
                // physical placement: SM[7:0], CTA[31:8], warp[39:32].
                track_[0] = 0;
                track_[1] = static_cast<uint64_t>(ptx::get_sm_idx() & 0xffu) |
                            (static_cast<uint64_t>(cta_idx & 0xffffffu) << 8) |
                            (static_cast<uint64_t>(warp_idx & 0xffu) << 32);
            }
        }
    }

    CUTLASS_DEVICE void begin(const MegaMoeEvent event,
                              const uint32_t payload = 0) {
        record(event, EventType::Begin, payload);
    }

    CUTLASS_DEVICE void end(const MegaMoeEvent event,
                            const uint32_t payload = 0) {
        record(event, EventType::End, payload);
    }

    CUTLASS_DEVICE void mark(const MegaMoeEvent event,
                             const uint32_t payload = 0) {
        record(event, EventType::Instant, payload);
    }

private:
    CUTLASS_DEVICE void record(const MegaMoeEvent event,
                               const EventType type,
                               const uint32_t payload) {
        if constexpr (kEnabled) {
            if (not active_)
                return;

            asm volatile("" ::: "memory");
            const uint64_t timestamp = globaltimer();
            const uint32_t event_idx = count_++;
            if (event_idx < capacity_) {
                auto* record = track_ + 2 + static_cast<uint64_t>(event_idx) * 2;
                record[0] = timestamp;
                record[1] = encode_event(event, type, payload);
            }
            // Recording attempted count lets the exporter diagnose truncation.
            track_[0] = count_;
            asm volatile("" ::: "memory");
        }
    }

    uint64_t* track_ = nullptr;
    uint32_t capacity_ = 0;
    uint32_t count_ = 0;
    bool active_ = false;
};

} // namespace deep_gemm::profile
