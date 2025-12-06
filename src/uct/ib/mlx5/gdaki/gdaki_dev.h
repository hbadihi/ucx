/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDAKI_DEV_H
#define UCT_GDAKI_DEV_H

#include <uct/api/device/uct_device_types.h>


typedef struct {
    uct_device_ep_t              super;
    void                         *atomic_va;
    uint32_t                     atomic_lkey;
    uint32_t                     pad[1];
    uint32_t                     sq_cq_dbrec[2];
    uint32_t                     rx_cq_dbrec[2];
    uint32_t                     sq_dbrec[2];
    uint32_t                     rx_dbrec[2];

    uint64_t                     sq_rsvd_index;
    uint64_t                     sq_ready_index;
    int                          sq_lock;

    uint8_t                      *sq_wqe_daddr;
    uint32_t                     *sq_dbrec_p;
    uint64_t                     *sq_db;
    uint8_t                      *sq_cqe_daddr;
    uint32_t                     sq_cqe_num;
    uint8_t                      *rx_cqe_daddr;
    uint32_t                     rx_cqe_num;
    uint32_t                     rx_cq_ci;
    uint32_t                     rx_wq_pi;
    uint32_t                     *rx_dbrec_p;
    uint64_t                     *rx_db;
    uint16_t                     sq_wqe_num;
    uint16_t                     rx_wqe_num;
    uint32_t                     sq_num;
    uint16_t                     sq_fc_mask;
} uct_rc_gdaki_dev_ep_t;

#define UCT_RC_GDAKI_SIGNALS_NUM 1024


typedef enum {
    UCT_RC_GDAKI_SIGNAL_OP_ADD = 0,
    UCT_RC_GDAKI_SIGNAL_OP_SET = 1
} uct_rc_gdaki_signal_op_t;


typedef struct uct_rc_gdaki_device_mem_element {
    uint32_t lkey;
    uint32_t rkey;
} uct_rc_gdaki_device_mem_element_t;

typedef struct {
    uint64_t wqe_idx;
} uct_rc_gda_completion_t;

#endif /* UCT_GDAKI_DEV_H */
