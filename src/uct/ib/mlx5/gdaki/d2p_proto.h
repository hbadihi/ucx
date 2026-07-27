/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_D2P_PROTO_H_
#define UCT_D2P_PROTO_H_

#include <ucs/sys/compiler_def.h>

#include <stdint.h>


enum {
    UCT_IB_D2P_OP_RDMA_WRITE = 0,
    UCT_IB_D2P_OP_ATOMIC_ADD = 1,
};


enum {
    UCT_IB_D2P_FLAG_CQ_UPDATE = UCS_BIT(0),
    UCT_IB_D2P_FLAG_RING_DB   = UCS_BIT(1),
};


enum {
    UCT_IB_D2P_DESC_SEG_HEADER = 0,
    UCT_IB_D2P_DESC_SEG_EP_IDX,
    UCT_IB_D2P_DESC_SEG_KEYS,
    UCT_IB_D2P_DESC_SEG_LADDR,
    UCT_IB_D2P_DESC_SEG_RADDR,
    UCT_IB_D2P_DESC_SEG_ADD,
    UCT_IB_D2P_DESC_SEG_RESERVED0,
    UCT_IB_D2P_DESC_SEG_RESERVED1,
    UCT_IB_D2P_DESC_SEG_COUNT,
};


#define UCT_IB_D2P_OWNER_MASK         UINT64_C(0x1)
#define UCT_IB_D2P_SEG_PAYLOAD_MASK   (~UINT64_C(0) >> 1)
#define UCT_IB_D2P_SEG_PAYLOAD_SHIFT  1

#define UCT_IB_D2P_HDR_OPCODE_SHIFT   1
#define UCT_IB_D2P_HDR_FLAGS_SHIFT    9
#define UCT_IB_D2P_HDR_LENGTH_SHIFT   25
#define UCT_IB_D2P_HDR_EP_MSB_SHIFT   57
#define UCT_IB_D2P_HDR_KEYS_MSB_SHIFT 58
#define UCT_IB_D2P_HDR_LADDR_MSB_SHIFT 59
#define UCT_IB_D2P_HDR_RADDR_MSB_SHIFT 60
#define UCT_IB_D2P_HDR_ADD_MSB_SHIFT  61

#define UCT_IB_D2P_HDR_OPCODE_MASK    UINT64_C(0xff)
#define UCT_IB_D2P_HDR_FLAGS_MASK     UINT64_C(0xffff)
#define UCT_IB_D2P_HDR_LENGTH_MASK    UINT64_C(0xffffffff)


typedef union {
    uint64_t raw;
    struct {
        uint64_t owner     : 1;
        uint64_t opcode    : 8;
        uint64_t flags     : 16;
        uint64_t length    : 32;
        uint64_t ep_msb    : 1;
        uint64_t keys_msb  : 1;
        uint64_t laddr_msb : 1;
        uint64_t raddr_msb : 1;
        uint64_t add_msb   : 1;
        uint64_t reserved  : 2;
    } bits;
} uct_ib_d2p_header_segment_t;


typedef union {
    uint64_t raw;
    struct {
        uint64_t owner   : 1;
        uint64_t payload : 63;
    } bits;
} uct_ib_d2p_data_segment_t;


typedef union {
    struct {
        uct_ib_d2p_header_segment_t header;
        uct_ib_d2p_data_segment_t   ep_idx;
        uct_ib_d2p_data_segment_t   keys;
        uct_ib_d2p_data_segment_t   laddr;
        uct_ib_d2p_data_segment_t   raddr;
        uct_ib_d2p_data_segment_t   add;
        uct_ib_d2p_data_segment_t   reserved[2];
    } field;
    uint64_t segments[UCT_IB_D2P_DESC_SEG_COUNT];
} uct_ib_d2p_desc_t UCS_V_ALIGNED(64);


typedef struct {
    uint8_t  owner;
    uint8_t  opcode;
    uint16_t flags;
    uint32_t length;
    uint64_t ep_idx;
    uint32_t lkey;
    uint32_t rkey;
    uint64_t laddr;
    uint64_t raddr;
    uint64_t add;
} uct_ib_d2p_desc_fields_t;


#ifdef __NVCC__
#  define UCT_IB_D2P_F_ALWAYS_INLINE __host__ __device__ __forceinline__ static
#else
#  define UCT_IB_D2P_F_ALWAYS_INLINE static inline
#endif


UCT_IB_D2P_F_ALWAYS_INLINE uint64_t
uct_ib_d2p_make_segment(uint8_t owner, uint64_t value)
{
    return (owner & UCT_IB_D2P_OWNER_MASK) |
           ((value & UCT_IB_D2P_SEG_PAYLOAD_MASK) <<
            UCT_IB_D2P_SEG_PAYLOAD_SHIFT);
}


UCT_IB_D2P_F_ALWAYS_INLINE uint64_t
uct_ib_d2p_segment_value(uint64_t segment, uint64_t msb)
{
    return (segment >> UCT_IB_D2P_SEG_PAYLOAD_SHIFT) | (msb << 63);
}


UCT_IB_D2P_F_ALWAYS_INLINE void
uct_ib_d2p_desc_pack(uct_ib_d2p_desc_t *desc, uint8_t owner, uint8_t opcode,
                     uint16_t flags, uint32_t length, uint64_t ep_idx,
                     uint32_t lkey, uint32_t rkey, uint64_t laddr,
                     uint64_t raddr, uint64_t add)
{
    const uint64_t keys = ((uint64_t)rkey << 32) | lkey;
    const uint64_t owner_bit = owner & UCT_IB_D2P_OWNER_MASK;

    desc->segments[UCT_IB_D2P_DESC_SEG_HEADER] =
            owner_bit |
            ((uint64_t)opcode << UCT_IB_D2P_HDR_OPCODE_SHIFT) |
            ((uint64_t)flags << UCT_IB_D2P_HDR_FLAGS_SHIFT) |
            ((uint64_t)length << UCT_IB_D2P_HDR_LENGTH_SHIFT) |
            ((ep_idx >> 63) << UCT_IB_D2P_HDR_EP_MSB_SHIFT) |
            ((keys >> 63) << UCT_IB_D2P_HDR_KEYS_MSB_SHIFT) |
            ((laddr >> 63) << UCT_IB_D2P_HDR_LADDR_MSB_SHIFT) |
            ((raddr >> 63) << UCT_IB_D2P_HDR_RADDR_MSB_SHIFT) |
            ((add >> 63) << UCT_IB_D2P_HDR_ADD_MSB_SHIFT);
    desc->segments[UCT_IB_D2P_DESC_SEG_EP_IDX] =
            uct_ib_d2p_make_segment(owner, ep_idx);
    desc->segments[UCT_IB_D2P_DESC_SEG_KEYS] =
            uct_ib_d2p_make_segment(owner, keys);
    desc->segments[UCT_IB_D2P_DESC_SEG_LADDR] =
            uct_ib_d2p_make_segment(owner, laddr);
    desc->segments[UCT_IB_D2P_DESC_SEG_RADDR] =
            uct_ib_d2p_make_segment(owner, raddr);
    desc->segments[UCT_IB_D2P_DESC_SEG_ADD] =
            uct_ib_d2p_make_segment(owner, add);
    desc->segments[UCT_IB_D2P_DESC_SEG_RESERVED0] = owner_bit;
    desc->segments[UCT_IB_D2P_DESC_SEG_RESERVED1] = owner_bit;
}


UCT_IB_D2P_F_ALWAYS_INLINE void
uct_ib_d2p_desc_decode(uct_ib_d2p_desc_fields_t *fields,
                       const uct_ib_d2p_desc_t *desc)
{
    const uint64_t header = desc->segments[UCT_IB_D2P_DESC_SEG_HEADER];
    const uint64_t keys =
            uct_ib_d2p_segment_value(
                    desc->segments[UCT_IB_D2P_DESC_SEG_KEYS],
                    (header >> UCT_IB_D2P_HDR_KEYS_MSB_SHIFT) &
                    UCT_IB_D2P_OWNER_MASK);

    fields->owner  = header & UCT_IB_D2P_OWNER_MASK;
    fields->opcode = (header >> UCT_IB_D2P_HDR_OPCODE_SHIFT) &
                     UCT_IB_D2P_HDR_OPCODE_MASK;
    fields->flags  = (header >> UCT_IB_D2P_HDR_FLAGS_SHIFT) &
                     UCT_IB_D2P_HDR_FLAGS_MASK;
    fields->length = (header >> UCT_IB_D2P_HDR_LENGTH_SHIFT) &
                     UCT_IB_D2P_HDR_LENGTH_MASK;
    fields->ep_idx = uct_ib_d2p_segment_value(
            desc->segments[UCT_IB_D2P_DESC_SEG_EP_IDX],
            (header >> UCT_IB_D2P_HDR_EP_MSB_SHIFT) &
            UCT_IB_D2P_OWNER_MASK);
    fields->lkey   = (uint32_t)keys;
    fields->rkey   = (uint32_t)(keys >> 32);
    fields->laddr  = uct_ib_d2p_segment_value(
            desc->segments[UCT_IB_D2P_DESC_SEG_LADDR],
            (header >> UCT_IB_D2P_HDR_LADDR_MSB_SHIFT) &
            UCT_IB_D2P_OWNER_MASK);
    fields->raddr  = uct_ib_d2p_segment_value(
            desc->segments[UCT_IB_D2P_DESC_SEG_RADDR],
            (header >> UCT_IB_D2P_HDR_RADDR_MSB_SHIFT) &
            UCT_IB_D2P_OWNER_MASK);
    fields->add    = uct_ib_d2p_segment_value(
            desc->segments[UCT_IB_D2P_DESC_SEG_ADD],
            (header >> UCT_IB_D2P_HDR_ADD_MSB_SHIFT) &
            UCT_IB_D2P_OWNER_MASK);
}


#ifdef __cplusplus
static_assert(sizeof(uct_ib_d2p_header_segment_t) == sizeof(uint64_t),
              "D2P header segment must be 8 bytes");
static_assert(sizeof(uct_ib_d2p_data_segment_t) == sizeof(uint64_t),
              "D2P data segment must be 8 bytes");
static_assert(sizeof(uct_ib_d2p_desc_t) == 64,
              "D2P descriptor must be 64 bytes");
#else
typedef char uct_ib_d2p_header_segment_size_check[
        (sizeof(uct_ib_d2p_header_segment_t) == sizeof(uint64_t)) ? 1 : -1];
typedef char uct_ib_d2p_data_segment_size_check[
        (sizeof(uct_ib_d2p_data_segment_t) == sizeof(uint64_t)) ? 1 : -1];
typedef char uct_ib_d2p_desc_size_check[(sizeof(uct_ib_d2p_desc_t) == 64) ?
                                                1 : -1];
#endif

#undef UCT_IB_D2P_F_ALWAYS_INLINE

#endif /* UCT_D2P_PROTO_H_ */
