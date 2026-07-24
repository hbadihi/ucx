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


UCS_F_DEVICE void uct_ib_d2p_store_relaxed_b128(volatile void *dst,
                                                const void *src)
{
    const uint64_t *src64 = reinterpret_cast<const uint64_t*>(src);
    const uint64_t val_lo = src64[0];
    const uint64_t val_hi = src64[1];

#ifdef __NVCC__
    asm volatile(R"(
        .reg .b128 _v%=;
        mov.b128 _v%=, {%1, %2};
        st.relaxed.sys.global.b128 [%0], _v%=;
    )" :: "l"(dst), "l"(val_lo), "l"(val_hi) : "memory");
#else
    volatile uint64_t *dst64 = reinterpret_cast<volatile uint64_t*>(dst);

    dst64[0] = val_lo;
    dst64[1] = val_hi;
#endif
}

UCS_F_DEVICE void uct_ib_d2p_store_release_b128(volatile void *dst,
                                                const void *src)
{
    const uint64_t *src64 = reinterpret_cast<const uint64_t*>(src);
    const uint64_t val_lo = src64[0];
    const uint64_t val_hi = src64[1];

#ifdef __NVCC__
    asm volatile(R"(
        .reg .b128 _v%=;
        mov.b128 _v%=, {%1, %2};
        st.release.sys.global.b128 [%0], _v%=;
    )" :: "l"(dst), "l"(val_lo), "l"(val_hi) : "memory");
#else
    volatile uint64_t *dst64 = reinterpret_cast<volatile uint64_t*>(dst);

    dst64[0] = val_lo;
    dst64[1] = val_hi;
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
        while (static_cast<long long>(pi - READ_ONCE(*ch->ci)) >= depth) {
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

            desc.owner  = (pi >> iface->log_depth) & 0x1;
            desc.opcode = opcode;
            desc.flags  = flags;
            desc.length = length;
            desc.ep_idx = ep->ep_idx;
            desc.lkey   = lkey;
            desc.rkey   = rkey;
            desc.laddr  = laddr;
            desc.raddr  = raddr;
            desc.add    = add;

            auto dst = reinterpret_cast<volatile uint8_t*>(queue_desc);
            auto src = reinterpret_cast<const uint8_t*>(&desc);

            uct_ib_d2p_store_relaxed_b128(dst + 16, src + 16);
            uct_ib_d2p_store_relaxed_b128(dst + 32, src + 32);
            uct_ib_d2p_store_release_b128(dst, src);
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
