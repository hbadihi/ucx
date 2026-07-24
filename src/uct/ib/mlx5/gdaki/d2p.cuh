/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_D2P_CUH_
#define UCT_D2P_CUH_

#include "d2p.h"
#include "d2p_proto.h"
#include "common.cuh"

#include <uct/api/device/uct_device_types.h>
#include <ucs/sys/device_code.h>


UCS_F_DEVICE void uct_ib_d2p_store_relaxed_u64(volatile uint64_t *dst,
                                               uint64_t value)
{
#ifdef __NVCC__
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;"
                 :
                 : "l"(dst), "l"(value)
                 : "memory");
#else
    *dst = value;
#endif
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_ib_d2p_post_desc(uct_ib_d2p_gpu_ep_t *ep,
                                               unsigned channel_id,
                                               uint8_t opcode, uint32_t length,
                                               uint32_t lkey, uint64_t laddr,
                                               uint32_t rkey, uint64_t raddr,
                                               uint64_t add, uint16_t flags,
                                               uct_device_completion_t *tl_comp)
{
    ucs_status_t status = UCS_INPROGRESS;
    unsigned lane_id, num_lanes;

    uct_dev_exec_init<level>(lane_id, num_lanes);

    if (lane_id == 0) {
        auto iface = ep->iface;
        auto ch    = iface->channels + (channel_id & iface->channel_mask);
        const long long depth = UCS_BIT(iface->log_depth);
        unsigned long long pi;

        // uint64_t start_reserve_time, end_reserve_time, start_post_time, end_post_time;


        // start_reserve_time = clock64();
        pi = atomicAdd(ch->pi, 1ULL);
        unsigned long long cached_ci = ch->ci_shadow;
        while (static_cast<long long>(pi - cached_ci) >= depth) {
            cached_ci     = READ_ONCE(*ch->ci);
            ch->ci_shadow = cached_ci;
            if (static_cast<long long>(pi - cached_ci) < depth) {
                break;
            }
        //     printf("no resource\n");
        }

        if (tl_comp != nullptr) {
            tl_comp->d2p.ch = ch;
            tl_comp->d2p.pi = pi;
        }
        // end_reserve_time = clock64();
        // start_post_time = clock64();
        if (status == UCS_INPROGRESS) {
            const uint32_t slot = pi & UCS_MASK(iface->log_depth);
            auto queue_desc = reinterpret_cast<volatile uct_ib_d2p_desc_t*>(
                                      ch->queue_base) +
                              slot;
            uct_ib_d2p_desc_t desc = {};
            const uint8_t owner = (pi >> iface->log_depth) &
                                  UCT_IB_D2P_OWNER_MASK;

            uct_ib_d2p_desc_pack(&desc, owner, opcode, flags, length,
                                 ep->ep_idx, lkey, rkey, laddr, raddr, add);

            auto dst = reinterpret_cast<volatile uint64_t*>(queue_desc);

#pragma unroll
            for (unsigned i = 0; i < UCT_IB_D2P_DESC_SEG_COUNT; ++i) {
                uct_ib_d2p_store_relaxed_u64(dst + i, desc.segments[i]);
            }
        }
        // end_post_time = clock64();
        // printf("reserve time: %.3f us, post time: %.3f us\n",
        //        static_cast<double>(end_reserve_time - start_reserve_time) /
        //                2000.0,
        //        static_cast<double>(end_post_time - start_post_time) / 2000.0);
    }

    return static_cast<ucs_status_t>(
            uct_dev_bcast<level>(static_cast<int>(status), lane_id));
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_ib_d2p_ep_put(
        uct_device_ep_h tl_ep, const uct_device_mem_elem_t *src_uct_elem,
        const uct_device_mem_elem_t *tl_mem_elem, const void *address,
        uint64_t remote_address, size_t length, unsigned channel_id,
        uint64_t flags, uct_device_completion_t *comp)
{
    auto ep     = reinterpret_cast<uct_ib_d2p_gpu_ep_t*>(tl_ep);
    auto src_ib = reinterpret_cast<const uct_ib_md_device_mem_element_t*>(
            src_uct_elem);
    auto rem_ib = reinterpret_cast<const uct_ib_md_device_mem_element_t*>(
            tl_mem_elem);
    uint16_t desc_flags = (comp != nullptr ? UCT_IB_D2P_FLAG_CQ_UPDATE : 0) |
                          (flags & UCT_DEVICE_FLAG_NODELAY ?
                                   UCT_IB_D2P_FLAG_RING_DB :
                                   0);

    return uct_ib_d2p_post_desc<level>(ep, channel_id, UCT_IB_D2P_OP_RDMA_WRITE,
                                       length, src_ib->lkey,
                                       reinterpret_cast<uint64_t>(address),
                                       rem_ib->rkey, remote_address, 0,
                                       desc_flags, comp);
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_ib_d2p_ep_atomic_add(
        uct_device_ep_h tl_ep, const uct_device_mem_elem_t *tl_mem_elem,
        uint64_t inc_value, uint64_t remote_address, unsigned channel_id,
        uint64_t flags, uct_device_completion_t *comp)
{
    auto ep     = reinterpret_cast<uct_ib_d2p_gpu_ep_t*>(tl_ep);
    auto rem_ib = reinterpret_cast<const uct_ib_md_device_mem_element_t*>(
            tl_mem_elem);
    uint16_t desc_flags = (comp != nullptr ? UCT_IB_D2P_FLAG_CQ_UPDATE : 0) |
                          (flags & UCT_DEVICE_FLAG_NODELAY ?
                                   UCT_IB_D2P_FLAG_RING_DB :
                                   0);

    return uct_ib_d2p_post_desc<level>(ep, channel_id, UCT_IB_D2P_OP_ATOMIC_ADD,
                                       sizeof(uint64_t),
                                       ep->iface->atomic_result_lkey,
                                       ep->iface->atomic_result_va,
                                       rem_ib->rkey, remote_address, inc_value,
                                       desc_flags, comp);
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_ib_d2p_ep_check_completion(
        uct_device_ep_h, uct_device_completion_t *tl_comp)
{
    uct_ib_d2p_completion_t *comp = &tl_comp->d2p;
    unsigned long long ci         = READ_ONCE(*comp->ch->ci);

    return (static_cast<long long>(ci - comp->pi) > 0) ? UCS_OK :
                                                         UCS_INPROGRESS;
}

#endif /* UCT_D2P_CUH_ */
