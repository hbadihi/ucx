/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gdaki.h"

#include <ucs/time/time.h>
#include <ucs/datastruct/string_buffer.h>
#include <uct/ib/mlx5/rc/rc_mlx5.h>
#include <uct/ib/mlx5/ib_mlx5.inl>
#include <uct/cuda/base/cuda_iface.h>

#include <cuda.h>


typedef struct {
    uct_rc_iface_common_config_t      super;
    uct_rc_mlx5_iface_common_config_t mlx5;
} uct_rc_gdaki_iface_config_t;

ucs_config_field_t uct_rc_gdaki_iface_config_table[] = {
    {UCT_IB_CONFIG_PREFIX, "", NULL,
     ucs_offsetof(uct_rc_gdaki_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_rc_iface_common_config_table)},

    {UCT_IB_CONFIG_PREFIX, "", NULL,
     ucs_offsetof(uct_rc_gdaki_iface_config_t, mlx5),
     UCS_CONFIG_TYPE_TABLE(uct_rc_mlx5_common_config_table)},

    {NULL}
};


ucs_status_t
uct_rc_gdaki_alloc(size_t size, size_t align, void **p_buf, CUdeviceptr *p_orig)
{
    unsigned int flag = 1;
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemAlloc(p_orig, size + align - 1));
    if (status != UCS_OK) {
        return status;
    }

    *p_buf = (void*)ucs_align_up_pow2_ptr(*p_orig, align);
    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                  (CUdeviceptr)*p_buf));
    if (status != UCS_OK) {
        goto err;
    }

    return UCS_OK;

err:
    cuMemFree(*p_orig);
    return status;
}

static void
uct_rc_gdaki_calc_dev_ep_layout(size_t sq_cq_umem_len, unsigned max_tx,
                                size_t rx_cq_umem_len,
                                size_t *sq_cq_umem_offset_p,
                                size_t *qp_umem_offset_p,
                                size_t *rx_cq_umem_offset_p,
                                size_t *dev_ep_size_p)
{
    size_t sq_wq_len;

    *sq_cq_umem_offset_p = ucs_align_up_pow2(sizeof(uct_rc_gdaki_dev_ep_t),
                                             ucs_get_page_size());
    *qp_umem_offset_p    = ucs_align_up_pow2(*sq_cq_umem_offset_p + sq_cq_umem_len,
                                             ucs_get_page_size());
    sq_wq_len            = max_tx * MLX5_SEND_WQE_BB;
    *rx_cq_umem_offset_p = ucs_align_up_pow2(*qp_umem_offset_p + sq_wq_len,
                                             ucs_get_page_size());
    *dev_ep_size_p       = *rx_cq_umem_offset_p + rx_cq_umem_len;
}

static UCS_CLASS_INIT_FUNC(uct_rc_gdaki_ep_t, const uct_ep_params_t *params)
{
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(params->iface,
                                                 uct_rc_gdaki_iface_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.super.super.super.md,
                                          uct_ib_mlx5_md_t);
    uct_ib_iface_init_attr_t init_attr = {};
    uct_ib_mlx5_cq_attr_t sq_cq_attr   = {};
    uct_ib_mlx5_cq_attr_t rx_cq_attr   = {};
    uct_ib_mlx5_qp_attr_t qp_attr      = {};
    ucs_status_t status;
    size_t dev_ep_size, rx_cq_umem_offset;
    uct_ib_mlx5_dbrec_t dbrec;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super.super.super);

    self->dev_ep_init = 0;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(iface->cuda_ctx));
    if (status != UCS_OK) {
        return status;
    }

    init_attr.cq_len[UCT_IB_DIR_TX] = 1;
    init_attr.cq_len[UCT_IB_DIR_RX] = iface->super.super.super.config.rx_max_batch;
    uct_ib_mlx5_cq_calc_sizes(&iface->super.super.super, UCT_IB_DIR_TX,
                              &init_attr, 0, &sq_cq_attr);
    uct_ib_mlx5_cq_calc_sizes(&iface->super.super.super, UCT_IB_DIR_RX,
                              &init_attr, 0, &rx_cq_attr);
    uct_rc_iface_fill_attr(&iface->super.super, &qp_attr.super,
                           iface->super.super.config.tx_qp_len,
                           iface->super.rx.srq.verbs.srq);
    uct_ib_mlx5_wq_calc_sizes(&qp_attr);

    sq_cq_attr.flags |= UCT_IB_MLX5_CQ_IGNORE_OVERRUN;

    qp_attr.mmio_mode = UCT_IB_MLX5_MMIO_MODE_DB;

    /* Disable inline scatter to TX CQE */
    qp_attr.super.max_inl_cqe[UCT_IB_DIR_TX] = 0;

    /*
     * dev_ep layout in GPU memory:
     * +---------------------------+------------+------------+------------+
     * | uct_rc_gdaki_dev_ep_t     | SQ CQ buf  | SQ WQ buf  | RX CQ buf  |
     * +---------------------------+------------+------------+------------+
     * 
     * uct_rc_gdaki_dev_ep_t struct contains:
     *   - Atomic operation metadata (atomic_va, atomic_lkey)
     *   - Doorbell records (32 bytes total):
     *       * sq_cq_dbrec[2]: SQ CQ doorbell record (8 bytes, 2 x uint32_t)
     *                         [0] = consumer counter
     *                         [1] = reserved/flags
     *       * rx_cq_dbrec[2]: RX CQ doorbell record (8 bytes, 2 x uint32_t)
     *                         [0] = consumer counter
     *                         [1] = reserved/flags
     *       * sq_dbrec[2]:    SQ doorbell record (8 bytes, 2 x uint32_t)
     *                         [0] = producer index
     *                         [1] = reserved/flags
     *       * rx_dbrec[2]:    SRQ doorbell record (8 bytes, 2 x uint32_t)
     *                         [0] = producer index
     *                         [1] = reserved/flags
     *   - Queue management:
     *       * sq_rsvd_index, sq_ready_index: producer/consumer tracking
     *       * sq_lock: synchronization for multi-thread access
     *   - Hardware pointers:
     *       * sq_wqe_daddr, sq_cqe_daddr: pointers to SQ WQ/CQ buffers
     *       * sq_dbrec_p, sq_db: SQ hardware doorbell pointers
     *       * rx_cqe_daddr: pointer to RX CQ buffer
     *       * rx_dbrec_p, rx_db: SRQ hardware doorbell pointers (rx_db is NULL)
     *   - Queue parameters: sq_cqe_num, rx_cqe_num, sq_wqe_num, sq_num, sq_fc_mask
     * 
     * All four sections (struct, SQ CQ buffer, SQ WQ buffer, RX CQ buffer) are
     * page-aligned for hardware DMA requirements.
     * 
     * Architecture notes:
     *   - SRQ (Shared Receive Queue) is at interface level with WQE buffer on CPU
     *   - SRQ doorbell record is on GPU (separate allocation at interface level)
     *   - RX CQ is per-endpoint on GPU for polling receive completions
     *   - SQ CQ is per-endpoint on GPU for polling send completions
     *   - CPU preposts WQEs at init, doorbell updated via cuMemcpyHtoD to GPU
     *   - GPU can post receives to SRQ via rx_dbrec_p (direct GPU memory access)
     *   - SRQ doesn't use UAR doorbell, only doorbell record updates
     */
    uct_rc_gdaki_calc_dev_ep_layout(sq_cq_attr.umem_len, qp_attr.max_tx,
                                    rx_cq_attr.umem_len,
                                    &sq_cq_attr.umem_offset,
                                    &qp_attr.umem_offset,
                                    &rx_cq_umem_offset,
                                    &dev_ep_size);

    status      = uct_rc_gdaki_alloc(dev_ep_size, ucs_get_page_size(),
                                     (void**)&self->ep_gpu, &self->ep_raw);
    if (status != UCS_OK) {
        goto err_ctx;
    }

    /* TODO add dmabuf_fd support */
    self->umem = mlx5dv_devx_umem_reg(md->super.dev.ibv_context, self->ep_gpu,
                                      dev_ep_size, IBV_ACCESS_LOCAL_WRITE);
    if (self->umem == NULL) {
        uct_ib_check_memlock_limit_msg(md->super.dev.ibv_context,
                                       UCS_LOG_LEVEL_ERROR,
                                       "mlx5dv_devx_umem_reg(ptr=%p size=%zu)",
                                       self->ep_gpu, dev_ep_size);
        status = UCS_ERR_NO_MEMORY;
        goto err_mem;
    }

    self->sq_cq.devx.mem.mem    = self->umem;
    self->rx_cq.devx.mem.mem    = self->umem;
    self->qp.super.devx.mem.mem = self->umem;

    dbrec.mem_id        = self->umem->umem_id;
    dbrec.offset        = ucs_offsetof(uct_rc_gdaki_dev_ep_t, sq_cq_dbrec);
    self->sq_cq.devx.dbrec = &dbrec;
    status = uct_ib_mlx5_devx_create_cq_common(&iface->super.super.super,
                                               UCT_IB_DIR_TX, &sq_cq_attr,
                                               &self->sq_cq, 0, 0);
    if (status != UCS_OK) {
        goto err_umem;
    }

    dbrec.offset           = ucs_offsetof(uct_rc_gdaki_dev_ep_t, rx_cq_dbrec);
    rx_cq_attr.umem_offset = rx_cq_umem_offset;
    self->rx_cq.devx.dbrec = &dbrec;
    status = uct_ib_mlx5_devx_create_cq_common(&iface->super.super.super,
                                               UCT_IB_DIR_RX, &rx_cq_attr,
                                               &self->rx_cq, 0, 0);
    if (status != UCS_OK) {
        goto err_sq_cq;
    }

    dbrec.offset              = ucs_offsetof(uct_rc_gdaki_dev_ep_t, sq_dbrec);
    self->qp.super.devx.dbrec = &dbrec;
    status = uct_ib_mlx5_devx_create_qp_common(&iface->super.super.super,
                                               &self->sq_cq, &self->rx_cq,
                                               &self->qp.super, &self->qp,
                                               &qp_attr);
    if (status != UCS_OK) {
        goto err_rx_cq;
    }

    (void)cuMemHostRegister(self->qp.reg->addr.ptr, UCT_IB_MLX5_BF_REG_SIZE * 2,
                            CU_MEMHOSTREGISTER_PORTABLE |
                            CU_MEMHOSTREGISTER_DEVICEMAP |
                            CU_MEMHOSTREGISTER_IOMEMORY);

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemHostGetDevicePointer((CUdeviceptr*)&self->sq_db,
                                      self->qp.reg->addr.ptr, 0));
    if (status != UCS_OK) {
        goto err_dev_ep;
    }

    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    return UCS_OK;

err_dev_ep:
    (void)cuMemHostUnregister(self->qp.reg->addr.ptr);
    uct_ib_mlx5_devx_destroy_qp_common(&self->qp.super);
err_rx_cq:
    uct_ib_mlx5_devx_destroy_cq_common(&self->rx_cq);
err_sq_cq:
    uct_ib_mlx5_devx_destroy_cq_common(&self->sq_cq);
err_umem:
    mlx5dv_devx_umem_dereg(self->umem);
err_mem:
    cuMemFree(self->ep_raw);
err_ctx:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_gdaki_ep_t)
{
    (void)cuMemHostUnregister(self->sq_db);
    uct_ib_mlx5_devx_destroy_qp_common(&self->qp.super);
    uct_ib_mlx5_devx_destroy_cq_common(&self->rx_cq);
    uct_ib_mlx5_devx_destroy_cq_common(&self->sq_cq);
    mlx5dv_devx_umem_dereg(self->umem);
    cuMemFree(self->ep_raw);
}

UCS_CLASS_DEFINE(uct_rc_gdaki_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_gdaki_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_gdaki_ep_t, uct_ep_t);

static ucs_status_t
uct_rc_gdaki_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    uct_rc_gdaki_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_gdaki_ep_t);
    uct_rc_mlx5_base_ep_address_t *rc_addr = (void*)addr;

    uct_ib_pack_uint24(rc_addr->qp_num, ep->qp.super.qp_num);
    return UCS_OK;
}

static ucs_status_t uct_rc_gdaki_iface_get_address(uct_iface_h tl_iface,
                                                   uct_iface_addr_t *addr)
{
    *(uint8_t*)addr = UCT_RC_MLX5_IFACE_ADDR_TYPE_BASIC;
    return UCS_OK;
}

static ucs_status_t
uct_rc_gdaki_ep_connect_to_ep_v2(uct_ep_h ep,
                                 const uct_device_addr_t *device_addr,
                                 const uct_ep_addr_t *ep_addr,
                                 const uct_ep_connect_to_ep_params_t *params)
{
    uct_rc_gdaki_ep_t *gdaki_ep     = ucs_derived_of(ep, uct_rc_gdaki_ep_t);
    uct_rc_gdaki_iface_t *iface     = ucs_derived_of(ep->iface,
                                                     uct_rc_gdaki_iface_t);
    const uct_ib_address_t *ib_addr = (void*)device_addr;
    const uct_rc_mlx5_base_ep_address_t *rc_addr = (void*)ep_addr;
    uint8_t path_index                           = 0;
    struct ibv_ah_attr ah_attr;
    enum ibv_mtu path_mtu;
    uint32_t dest_qp_num;
    ucs_status_t status;

    status = uct_ib_iface_fill_ah_attr_from_addr(&iface->super.super.super,
                                                 ib_addr, path_index, &ah_attr,
                                                 &path_mtu);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(path_mtu != UCT_IB_ADDRESS_INVALID_PATH_MTU);
    dest_qp_num = uct_ib_unpack_uint24(rc_addr->qp_num);

    return uct_rc_mlx5_iface_common_devx_connect_qp(
            &iface->super, &gdaki_ep->qp.super, dest_qp_num, &ah_attr, path_mtu,
            path_index, iface->super.super.config.max_rd_atomic);
}

int uct_rc_gdaki_ep_is_connected(uct_ep_h tl_ep,
                                 const uct_ep_is_connected_params_t *params)
{
    uct_rc_gdaki_ep_t *ep       = ucs_derived_of(tl_ep, uct_rc_gdaki_ep_t);
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                                 uct_rc_gdaki_iface_t);
    uint32_t addr_qp            = 0;
    uct_rc_mlx5_base_ep_address_t *rc_addr;
    ucs_status_t status;
    struct ibv_ah_attr ah_attr;
    uint32_t qp_num;
    union ibv_gid *rgid;
    const uct_ib_address_t *ib_addr;

    status = uct_ib_mlx5_query_qp_peer_info(&iface->super.super.super,
                                            &ep->qp.super, &ah_attr, &qp_num);
    if (status != UCS_OK) {
        return 0;
    }

    /* TODO unite code with uct_rc_mlx5_base_ep_is_connected */
    if (params->field_mask & UCT_EP_IS_CONNECTED_FIELD_EP_ADDR) {
        rc_addr = (uct_rc_mlx5_base_ep_address_t*)params->ep_addr;
        addr_qp = uct_ib_unpack_uint24(rc_addr->qp_num);
    }

    if ((addr_qp != 0) && (qp_num != addr_qp)) {
        return 0;
    }

    rgid    = (ah_attr.is_global) ? &ah_attr.grh.dgid : NULL;
    ib_addr = (const uct_ib_address_t*)params->device_addr;
    return uct_ib_iface_is_same_device(ib_addr, ah_attr.dlid, rgid);
}

static ucs_status_t
uct_rc_gdaki_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(tl_iface,
                                                 uct_rc_gdaki_iface_t);
    ucs_status_t status;

    status = uct_ib_iface_query(&iface->super.super.super, 0, iface_attr);
    if (status != UCS_OK) {
        return status;
    }

    /* TODO:
     *  - add UCT_IFACE_FLAG_PUT_BATCH
     *  - PENDING and PUT_ZCOPY will be needed to establish rma_bw lanes
     *  - As this lane does not really support PUT_ZCOPY and PENDING, this could be
     *    causing issue when trying to send standard PUT. Eventually we must probably
     *    introduce another type of lane (rma_batch#x).
     */
    iface_attr->cap.flags = UCT_IFACE_FLAG_CONNECT_TO_EP |
                            UCT_IFACE_FLAG_INTER_NODE |
                            UCT_IFACE_FLAG_DEVICE_EP |
                            UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;

    iface_attr->ep_addr_len    = sizeof(uct_rc_mlx5_base_ep_address_t);
    iface_attr->iface_addr_len = sizeof(uint8_t);
    iface_attr->overhead       = UCT_RC_MLX5_IFACE_OVERHEAD;

    iface_attr->cap.put.min_zcopy = 0;
    iface_attr->cap.put.max_zcopy =
            uct_ib_iface_port_attr(&iface->super.super.super)->max_msg_sz;
    return UCS_OK;
}

static ucs_status_t uct_rc_gdaki_iface_query_v2(uct_iface_h tl_iface,
                                                uct_iface_attr_v2_t *iface_attr)
{
    if (iface_attr->field_mask & UCT_IFACE_ATTR_FIELD_DEVICE_MEM_ELEMENT_SIZE) {
        iface_attr->device_mem_element_size = sizeof(
                uct_rc_gdaki_device_mem_element_t);
    }

    return UCS_OK;
}

ucs_status_t
uct_rc_gdaki_create_cq(uct_ib_iface_t *ib_iface, uct_ib_dir_t dir,
                       const uct_ib_iface_init_attr_t *init_attr,
                       int preferred_cpu, size_t inl)
{
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(ib_iface,
                                                 uct_rc_gdaki_iface_t);

    iface->super.cq[dir].type = UCT_IB_MLX5_OBJ_TYPE_NULL;
    return UCS_OK;
}

ucs_status_t
uct_rc_gdaki_ep_get_device_ep(uct_ep_h tl_ep, uct_device_ep_h *device_ep_p)
{
    uct_rc_gdaki_ep_t *ep        = ucs_derived_of(tl_ep, uct_rc_gdaki_ep_t);
    uct_rc_gdaki_iface_t *iface  = ucs_derived_of(ep->super.super.iface,
                                                  uct_rc_gdaki_iface_t);
    uct_rc_gdaki_dev_ep_t dev_ep = {};
    unsigned sq_cq_size, sq_cqe_size, rx_cq_size, rx_cqe_size, max_tx;
    size_t sq_cq_umem_offset, sq_cq_umem_len, qp_umem_offset;
    size_t rx_cq_umem_offset, rx_cq_umem_len, dev_ep_size;
    ucs_status_t status;

    pthread_mutex_lock(&iface->ep_init_lock);

    if (!ep->dev_ep_init) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(iface->cuda_ctx));
        if (status != UCS_OK) {
            goto out_unlock;
        }

        sq_cq_size     = UCS_BIT(ep->sq_cq.cq_length_log);
        sq_cqe_size    = UCS_BIT(ep->sq_cq.cqe_size_log);
        sq_cq_umem_len = sq_cqe_size * sq_cq_size;
        rx_cq_size     = UCS_BIT(ep->rx_cq.cq_length_log);
        rx_cqe_size    = UCS_BIT(ep->rx_cq.cqe_size_log);
        rx_cq_umem_len = rx_cqe_size * rx_cq_size;
        /* Reconstruct original max_tx from bb_max */
        max_tx         = ep->qp.bb_max + 2 * UCT_IB_MLX5_MAX_BB;

        uct_rc_gdaki_calc_dev_ep_layout(sq_cq_umem_len, max_tx, rx_cq_umem_len,
                                        &sq_cq_umem_offset, &qp_umem_offset,
                                        &rx_cq_umem_offset, &dev_ep_size);

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuMemsetD8((CUdeviceptr)ep->ep_gpu, 0, dev_ep_size));
        if (status != UCS_OK) {
            goto out_ctx;
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemsetD8(
                (CUdeviceptr)UCS_PTR_BYTE_OFFSET(ep->ep_gpu, sq_cq_umem_offset),
                0xff, sq_cq_umem_len));
        if (status != UCS_OK) {
            goto out_ctx;
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemsetD8(
                (CUdeviceptr)UCS_PTR_BYTE_OFFSET(ep->ep_gpu, rx_cq_umem_offset),
                0xff, rx_cq_umem_len));
        if (status != UCS_OK) {
            goto out_ctx;
        }

        dev_ep.atomic_va      = iface->atomic_buff;
        dev_ep.atomic_lkey    = htonl(iface->atomic_mr->lkey);
        dev_ep.sq_num         = ep->qp.super.qp_num;
        dev_ep.sq_wqe_daddr   = UCS_PTR_BYTE_OFFSET(ep->ep_gpu, qp_umem_offset);
        dev_ep.sq_dbrec_p     = &ep->ep_gpu->sq_dbrec[MLX5_SND_DBR];
        dev_ep.sq_wqe_num     = max_tx;
        /* FC mask is used to determine if WQE should be posted with completion.
         * max_tx must be a power of 2. */
        dev_ep.sq_fc_mask     = (max_tx >> 1) - 1;

        dev_ep.sq_cqe_daddr      = UCS_PTR_BYTE_OFFSET(ep->ep_gpu, sq_cq_umem_offset);
        dev_ep.sq_cqe_num        = sq_cq_size;
        dev_ep.sq_db          = ep->sq_db;

        dev_ep.rx_cqe_daddr = UCS_PTR_BYTE_OFFSET(ep->ep_gpu, rx_cq_umem_offset);
        dev_ep.rx_cqe_num   = rx_cq_size;

        /* Set SRQ doorbell pointers (SRQ is at interface level, doorbell on GPU) */
        dev_ep.rx_dbrec_p   = (uint32_t*)iface->srq_dbrec_gpu; /* Points to GPU! */
        dev_ep.rx_db        = NULL; /* SRQ doesn't use UAR doorbell */
        dev_ep.rx_wq_pi     = iface->super.rx.srq.sw_pi;

        status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemcpyHtoD(
                (CUdeviceptr)ep->ep_gpu, &dev_ep, sizeof(dev_ep)));
        if (status != UCS_OK) {
            goto out_ctx;
        }

        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));

        ep->dev_ep_init = 1;
    }

    *device_ep_p = &ep->ep_gpu->super;
    pthread_mutex_unlock(&iface->ep_init_lock);
    return UCS_OK;

out_ctx:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
out_unlock:
    pthread_mutex_unlock(&iface->ep_init_lock);
    return status;
}

ucs_status_t
uct_rc_gdaki_iface_mem_element_pack(const uct_iface_h tl_iface, uct_mem_h memh,
                                    uct_rkey_t rkey,
                                    uct_device_mem_element_t *mem_elem_p)
{
    uct_rc_gdaki_device_mem_element_t mem_elem;

    mem_elem.rkey = htonl(uct_ib_md_direct_rkey(rkey));
    if (memh == NULL) {
        mem_elem.lkey = UCT_IB_INVALID_MKEY;
    } else {
        mem_elem.lkey = htonl(((uct_ib_mem_t*)memh)->lkey);
    }

    return UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemcpyHtoD((CUdeviceptr)mem_elem_p, &mem_elem, sizeof(mem_elem)));
}

/* Update SRQ resources and write doorbell to GPU memory via cuMemcpyHtoD */
static UCS_F_ALWAYS_INLINE void
uct_rc_gdaki_iface_update_srq_res(uct_rc_gdaki_iface_t *iface,
                                  uct_ib_mlx5_srq_t *srq,
                                  uint16_t wqe_index, uint16_t count)
{
    uct_rc_iface_t *rc_iface = &iface->super.super;
    uint32_t doorbell_value;
    ucs_status_t status;
    
    ucs_assert(rc_iface->rx.srq.available >= count);
    
    if (count == 0) {
        return;
    }
    
    srq->ready_idx              = wqe_index;
    srq->sw_pi                 += count;
    rc_iface->rx.srq.available -= count;
    ucs_memory_cpu_store_fence();
    
    /* Write doorbell to GPU memory using cuMemcpyHtoD */
    doorbell_value = htonl(srq->sw_pi);
    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(iface->cuda_ctx));
    if (status == UCS_OK) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemcpyHtoD(iface->srq_dbrec_gpu, &doorbell_value, sizeof(uint32_t)));
        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    }
    
    if (status != UCS_OK) {
        ucs_error("Failed to update GPU doorbell: %s", ucs_status_string(status));
    }
}

/* Post receives to SRQ with GPU doorbell update */
static unsigned uct_rc_gdaki_iface_srq_post_recv(uct_rc_gdaki_iface_t *iface)
{
    uct_ib_mlx5_srq_t *srq             = &iface->super.rx.srq;
    uct_rc_mlx5_iface_common_t *mlx5_iface = &iface->super;
    uct_ib_mlx5_srq_seg_t *seg;
    uint16_t count, wqe_index, next_index;
    
    ucs_assert(UCS_CIRCULAR_COMPARE16(srq->ready_idx, <=, srq->free_idx));
    ucs_assert(mlx5_iface->super.rx.srq.available > 0);
    
    wqe_index = srq->ready_idx;
    for (;;) {
        next_index = wqe_index + 1;
        seg = uct_ib_mlx5_srq_get_wqe(srq, next_index);
        if (UCS_CIRCULAR_COMPARE16(next_index, >, srq->free_idx)) {
            if (!seg->srq.free) {
                break;
            }
            
            ucs_assert(next_index == (uint16_t)(srq->free_idx + 1));
            seg->srq.free  = 0;
            srq->free_idx  = next_index;
        }
        
        if (uct_rc_mlx5_iface_srq_set_seg(mlx5_iface, seg) != UCS_OK) {
            break;
        }
        
        wqe_index = next_index;
    }
    
    count = wqe_index - srq->sw_pi;
    uct_rc_gdaki_iface_update_srq_res(iface, srq, wqe_index, count);
    return count;
}

/* Prepost receives at initialization with GPU doorbell update */
static void uct_rc_gdaki_iface_prepost_recvs(uct_rc_gdaki_iface_t *iface)
{
    /* prepost recvs only if quota available (recvs were not preposted before) */
    if (iface->super.super.rx.srq.quota == 0) {
        return;
    }
    
    iface->super.super.rx.srq.available = iface->super.super.rx.srq.quota;
    iface->super.super.rx.srq.quota     = 0;
    uct_rc_gdaki_iface_srq_post_recv(iface);
}

static ucs_status_t
uct_rc_gdaki_iface_devx_init_rx(uct_rc_iface_t *rc_iface,
                                const uct_rc_iface_common_config_t *config)
{
    uct_rc_gdaki_iface_t *iface       = ucs_derived_of(rc_iface, uct_rc_gdaki_iface_t);
    uct_rc_mlx5_iface_common_t *mlx5_iface = &iface->super;
    uct_ib_mlx5_md_t *md              = uct_ib_mlx5_iface_md(&mlx5_iface->super.super);
    uct_ib_device_t *dev              = &md->super.dev;
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(create_rmp_in)]   = {};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_rmp_out)] = {};
    size_t dbrec_size;
    ucs_status_t status;
    void *rmpc, *wq;
    int len, max, stride, wq_type;

    /* Push CUDA context */
    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(iface->cuda_ctx));
    if (status != UCS_OK) {
        return status;
    }

    /* Allocate page-aligned GPU memory for doorbell record */
    dbrec_size = ucs_align_up(8, ucs_get_page_size()); /* 2 x uint32_t */
    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemAlloc(&iface->srq_dbrec_gpu, dbrec_size));
    if (status != UCS_OK) {
        goto err_ctx;
    }

    /* Initialize doorbell to zero */
    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemsetD8(iface->srq_dbrec_gpu, 0, dbrec_size));
    if (status != UCS_OK) {
        goto err_free_gpu;
    }

    /* Register GPU memory with DEVX */
    iface->srq_dbrec_umem = mlx5dv_devx_umem_reg(dev->ibv_context,
                                                 (void*)iface->srq_dbrec_gpu,
                                                 dbrec_size,
                                                 IBV_ACCESS_LOCAL_WRITE);
    if (iface->srq_dbrec_umem == NULL) {
        uct_ib_check_memlock_limit_msg(dev->ibv_context, UCS_LOG_LEVEL_ERROR,
                                       "mlx5dv_devx_umem_reg(gpu_dbrec ptr=%p size=%zu)",
                                       (void*)iface->srq_dbrec_gpu, dbrec_size);
        status = UCS_ERR_NO_MEMORY;
        goto err_free_gpu;
    }

    /* Calculate SRQ parameters */
    stride = uct_ib_mlx5_srq_stride(mlx5_iface->tm.mp.num_strides);
    max    = uct_ib_mlx5_srq_max_wrs(config->super.rx.queue_len,
                                     mlx5_iface->tm.mp.num_strides);
    max    = ucs_roundup_pow2(max);
    len    = max * stride;

    /* Allocate CPU memory for SRQ WQE buffer */
    status = uct_ib_mlx5_md_buf_alloc(md, len, 0, &mlx5_iface->rx.srq.buf,
                                      &mlx5_iface->rx.srq.devx.mem, 0, "srq buf");
    if (status != UCS_OK) {
        goto err_dereg_umem;
    }

    /* Create doorbell record structure pointing to GPU */
    iface->srq_dbrec_struct = ucs_calloc(1, sizeof(uct_ib_mlx5_dbrec_t),
                                         "srq gpu dbrec");
    if (iface->srq_dbrec_struct == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_wqe_buf;
    }

    iface->srq_dbrec_struct->mem_id = iface->srq_dbrec_umem->umem_id;
    iface->srq_dbrec_struct->offset = 0; /* Doorbell at start of allocation */
    iface->srq_dbrec_struct->md     = md;
    
    mlx5_iface->rx.srq.devx.dbrec = iface->srq_dbrec_struct;
    mlx5_iface->rx.srq.db = (volatile uint32_t*)iface->srq_dbrec_gpu;

    /* Build SRQ creation command */
    UCT_IB_MLX5DV_SET(create_rmp_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_RMP);
    rmpc = UCT_IB_MLX5DV_ADDR_OF(create_rmp_in, in, rmp_context);
    wq   = UCT_IB_MLX5DV_ADDR_OF(rmpc, rmpc, wq);

    UCT_IB_MLX5DV_SET(rmpc, rmpc, state, UCT_IB_MLX5_RMPC_STATE_RDY);

    /* Determine WQ type */
    if (mlx5_iface->config.srq_topo == UCT_RC_MLX5_SRQ_TOPO_CYCLIC) {
        wq_type = UCT_RC_MLX5_MP_ENABLED(mlx5_iface) ?
                  UCT_IB_MLX5_SRQ_TOPO_CYCLIC_MP_RQ :
                  UCT_IB_MLX5_SRQ_TOPO_CYCLIC;
    } else {
        wq_type = UCT_RC_MLX5_MP_ENABLED(mlx5_iface) ?
                  UCT_IB_MLX5_SRQ_TOPO_LIST_MP_RQ :
                  UCT_IB_MLX5_SRQ_TOPO_LIST;
    }

    /* Set WQ parameters */
    UCT_IB_MLX5DV_SET  (wq, wq, wq_type,       wq_type);
    UCT_IB_MLX5DV_SET  (wq, wq, log_wq_sz,     ucs_ilog2(max));
    UCT_IB_MLX5DV_SET  (wq, wq, log_wq_stride, ucs_ilog2(stride));
    UCT_IB_MLX5DV_SET  (wq, wq, pd,            uct_ib_mlx5_devx_md_get_pdn(md));
    
    /* Point to GPU doorbell */
    UCT_IB_MLX5DV_SET  (wq, wq, dbr_umem_id,   iface->srq_dbrec_umem->umem_id);
    UCT_IB_MLX5DV_SET64(wq, wq, dbr_addr,      0); /* Offset 0 in the UMEM */
    
    /* Point to CPU WQE buffer */
    UCT_IB_MLX5DV_SET  (wq, wq, wq_umem_id,    mlx5_iface->rx.srq.devx.mem.mem->umem_id);

    if (UCT_RC_MLX5_MP_ENABLED(mlx5_iface)) {
        int log_num_of_strides = ucs_ilog2(mlx5_iface->tm.mp.num_strides) - 9;
        UCT_IB_MLX5DV_SET(wq, wq, log_wqe_num_of_strides,
                          log_num_of_strides & 0xF);
        UCT_IB_MLX5DV_SET(wq, wq, log_wqe_stride_size,
                          (ucs_ilog2(mlx5_iface->super.super.config.seg_size) - 6));
    }

    /* Create SRQ object */
    mlx5_iface->rx.srq.devx.obj = uct_ib_mlx5_devx_obj_create(dev->ibv_context, in,
                                                               sizeof(in), out,
                                                               sizeof(out), "RMP",
                                                               UCS_LOG_LEVEL_ERROR);
    if (mlx5_iface->rx.srq.devx.obj == NULL) {
        status = UCS_ERR_IO_ERROR;
        goto err_free_dbrec_struct;
    }

    mlx5_iface->rx.srq.srq_num = UCT_IB_MLX5DV_GET(create_rmp_out, out, rmpn);
    mlx5_iface->rx.srq.type = UCT_IB_MLX5_OBJ_TYPE_DEVX;
    
    /* Initialize SRQ buffer metadata */
    uct_ib_mlx5_srq_buff_init(&mlx5_iface->rx.srq, 0, max - 1,
                              mlx5_iface->super.super.config.seg_size,
                              mlx5_iface->tm.mp.num_strides);
    mlx5_iface->super.rx.srq.quota = max - 1;
    
    /* Prepost receives: CPU prepares WQEs, doorbell updated to GPU via cuMemcpyHtoD */
    uct_rc_gdaki_iface_prepost_recvs(iface);

    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    return UCS_OK;

err_free_dbrec_struct:
    ucs_free(iface->srq_dbrec_struct);
err_free_wqe_buf:
    uct_ib_mlx5_md_buf_free(md, mlx5_iface->rx.srq.buf, &mlx5_iface->rx.srq.devx.mem);
err_dereg_umem:
    mlx5dv_devx_umem_dereg(iface->srq_dbrec_umem);
err_free_gpu:
    cuMemFree(iface->srq_dbrec_gpu);
err_ctx:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    return status;
}

static void uct_rc_gdaki_iface_devx_cleanup_rx(uct_rc_iface_t *rc_iface)
{
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(rc_iface, uct_rc_gdaki_iface_t);
    uct_ib_mlx5_md_t *md = uct_ib_mlx5_iface_md(&iface->super.super.super);
    
    /* Destroy SRQ object */
    uct_ib_mlx5_devx_obj_destroy(iface->super.rx.srq.devx.obj, "RMP");
    
    /* Free CPU WQE buffer */
    uct_ib_mlx5_md_buf_free(md, iface->super.rx.srq.buf, &iface->super.rx.srq.devx.mem);
    
    /* Deregister GPU UMEM */
    mlx5dv_devx_umem_dereg(iface->srq_dbrec_umem);
    
    /* Free GPU memory */
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPushCurrent(iface->cuda_ctx));
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemFree(iface->srq_dbrec_gpu));
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    
    /* Free metadata structure */
    ucs_free(iface->srq_dbrec_struct);
}

static ucs_status_t
uct_rc_gdaki_init_rx(uct_rc_iface_t *rc_iface,
                     const uct_rc_iface_common_config_t *rc_config)
{
    return uct_rc_gdaki_iface_devx_init_rx(rc_iface, rc_config);
}

static void uct_rc_gdaki_cleanup_rx(uct_rc_iface_t *rc_iface)
{
    uct_rc_gdaki_iface_devx_cleanup_rx(rc_iface);
}

static UCS_CLASS_DECLARE_NEW_FUNC(uct_rc_gdaki_iface_t, uct_iface_t, uct_md_h,
                                  uct_worker_h, const uct_iface_params_t*,
                                  const uct_iface_config_t*);

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_rc_gdaki_iface_t, uct_iface_t);

static UCS_CLASS_DECLARE_NEW_FUNC(uct_rc_gdaki_iface_t, uct_iface_t, uct_md_h,
                                  uct_worker_h, const uct_iface_params_t*,
                                  const uct_iface_config_t*);

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_rc_gdaki_iface_t, uct_iface_t);

static uct_rc_iface_ops_t uct_rc_gdaki_internal_ops = {
    .super = {
        .super = {
            .iface_query_v2         = uct_rc_gdaki_iface_query_v2,
            .iface_estimate_perf    = uct_ib_iface_estimate_perf,
            .iface_vfs_refresh      = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
            .iface_mem_element_pack = uct_rc_gdaki_iface_mem_element_pack,
            .ep_query               = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
            .ep_invalidate          = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported,
            .ep_connect_to_ep_v2    = uct_rc_gdaki_ep_connect_to_ep_v2,
            .iface_is_reachable_v2  = (uct_iface_is_reachable_v2_func_t)ucs_empty_function_return_one_int,
            .ep_is_connected        = uct_rc_gdaki_ep_is_connected,
            .ep_get_device_ep       = uct_rc_gdaki_ep_get_device_ep,
        },
        .create_cq  = uct_rc_gdaki_create_cq,
        .destroy_cq = (uct_ib_iface_destroy_cq_func_t)ucs_empty_function_return_success,
    },
    .init_rx    = uct_rc_gdaki_init_rx,
    .cleanup_rx = uct_rc_gdaki_cleanup_rx,
};

static uct_iface_ops_t uct_rc_gdaki_iface_tl_ops = {
    .ep_flush          = uct_base_ep_flush,
    .ep_fence          = uct_base_ep_fence,
    .ep_create         = UCS_CLASS_NEW_FUNC_NAME(uct_rc_gdaki_ep_t),
    .ep_destroy        = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_gdaki_ep_t),
    .ep_get_address    = uct_rc_gdaki_ep_get_address,
    .ep_connect_to_ep  = uct_base_ep_connect_to_ep,
    .ep_pending_purge  = (uct_ep_pending_purge_func_t)ucs_empty_function,
    .iface_close       = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_gdaki_iface_t),
    .iface_query       = uct_rc_gdaki_iface_query,
    .iface_get_address = uct_rc_gdaki_iface_get_address,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_base_iface_is_reachable,
    .iface_flush              = (uct_iface_flush_func_t)
            ucs_empty_function_return_success,
    .iface_fence              = (uct_iface_fence_func_t)
            ucs_empty_function_return_unsupported,
    .iface_progress_enable    = (uct_iface_progress_enable_func_t)
            ucs_empty_function_return_unsupported,
    .iface_progress_disable   = (uct_iface_progress_disable_func_t)
            ucs_empty_function_return_unsupported,
    .iface_progress           = (uct_iface_progress_func_t)
            ucs_empty_function_return_unsupported,
};


static UCS_CLASS_INIT_FUNC(uct_rc_gdaki_iface_t, uct_md_h tl_md,
                           uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_rc_gdaki_iface_config_t *config =
            ucs_derived_of(tl_config, uct_rc_gdaki_iface_config_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(tl_md, uct_ib_mlx5_md_t);
    uct_ib_iface_init_attr_t init_attr = {};
    UCS_STRING_BUFFER_ONSTACK(strb, 64);
    char *gpu_name, *ib_name;
    char pci_addr[UCS_SYS_BDF_NAME_MAX];
    ucs_status_t status;
    int cuda_id;

    status = uct_rc_mlx5_dp_ordering_ooo_init(md, &self->super,
                                              md->dp_ordering_cap_devx.rc,
                                              md->ddp_support_dv.rc,
                                              &config->mlx5, "rc_gda");
    if (status != UCS_OK) {
        return status;
    }

    ucs_string_buffer_appendf(&strb, "%s", params->mode.device.dev_name);
    gpu_name = ucs_string_buffer_next_token(&strb, NULL, "-");
    ib_name  = ucs_string_buffer_next_token(&strb, gpu_name, "-");

    if (memcmp(gpu_name, UCT_DEVICE_CUDA_NAME, UCT_DEVICE_CUDA_NAME_LEN)) {
        ucs_error("wrong device name: %s\n", gpu_name);
        return UCS_ERR_INVALID_PARAM;
    }

    /* Initialize CUDA context before SUPER_INIT, since init_rx needs it */
    cuda_id = atoi(gpu_name + UCT_DEVICE_CUDA_NAME_LEN);
    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetPCIBusId(
                    pci_addr, UCS_SYS_BDF_NAME_MAX, cuda_id));
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&self->cuda_dev, cuda_id));
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuDevicePrimaryCtxRetain(&self->cuda_ctx, self->cuda_dev));
    if (status != UCS_OK) {
        return status;
    }

    init_attr.seg_size = config->super.super.seg_size;
    init_attr.qp_type  = IBV_QPT_RC;
    init_attr.dev_name = ib_name;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_mlx5_iface_common_t,
                              &uct_rc_gdaki_iface_tl_ops,
                              &uct_rc_gdaki_internal_ops, tl_md, worker, params,
                              &config->super, &config->mlx5, &init_attr);

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(self->cuda_ctx));
    if (status != UCS_OK) {
        goto err_ctx_release;
    }

    status = uct_rc_gdaki_alloc(sizeof(uint64_t), sizeof(uint64_t),
                                (void**)&self->atomic_buff, &self->atomic_raw);
    if (status != UCS_OK) {
        goto err_ctx;
    }

    self->atomic_mr = ibv_reg_mr(md->super.pd, self->atomic_buff,
                                 sizeof(uint64_t),
                                 IBV_ACCESS_LOCAL_WRITE |
                                 IBV_ACCESS_REMOTE_WRITE |
                                 IBV_ACCESS_REMOTE_READ |
                                 IBV_ACCESS_REMOTE_ATOMIC);
    if (self->atomic_mr == NULL) {
        status = UCS_ERR_IO_ERROR;
        goto err_atomic;
    }

    if (pthread_mutex_init(&self->ep_init_lock, NULL) != 0) {
        status = UCS_ERR_IO_ERROR;
        goto err_lock;
    }

    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    return UCS_OK;

err_lock:
    ibv_dereg_mr(self->atomic_mr);
err_atomic:
    cuMemFree(self->atomic_raw);
err_ctx:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
err_ctx_release:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(self->cuda_dev));
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_gdaki_iface_t)
{
    pthread_mutex_destroy(&self->ep_init_lock);
    ibv_dereg_mr(self->atomic_mr);
    cuMemFree(self->atomic_raw);
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(self->cuda_dev));
}

UCS_CLASS_DEFINE(uct_rc_gdaki_iface_t, uct_rc_mlx5_iface_common_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_gdaki_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_gdaki_iface_t, uct_iface_t);

static ucs_status_t
uct_gdaki_md_check_uar(uct_ib_mlx5_md_t *md, CUdevice cuda_dev)
{
    struct mlx5dv_devx_uar *uar;
    ucs_status_t status;
    CUcontext cuda_ctx;
    unsigned flags;

    status = uct_ib_mlx5_devx_alloc_uar(md, 0, &uar);
    if (status != UCS_OK) {
        goto out;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuDevicePrimaryCtxRetain(&cuda_ctx, cuda_dev));
    if (status != UCS_OK) {
        goto out_free_uar;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(cuda_ctx));
    if (status != UCS_OK) {
        goto out_ctx_release;
    }

    flags  = CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP |
             CU_MEMHOSTREGISTER_IOMEMORY;
    status = UCT_CUDADRV_FUNC_LOG_DEBUG(
            cuMemHostRegister(uar->reg_addr, UCT_IB_MLX5_BF_REG_SIZE, flags));
    if (status == UCS_OK) {
        UCT_CUDADRV_FUNC_LOG_DEBUG(cuMemHostUnregister(uar->reg_addr));
    }

    UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
out_ctx_release:
    UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(cuda_dev));
out_free_uar:
    mlx5dv_devx_free_uar(uar);
out:
    return status;
}

static ucs_status_t
uct_gdaki_query_tl_devices(uct_md_h tl_md,
                           uct_tl_device_resource_t **tl_devices_p,
                           unsigned *num_tl_devices_p)
{
    static int uar_supported  = -1;
    static int peermem_loaded = -1;
    uct_ib_mlx5_md_t *md      = ucs_derived_of(tl_md, uct_ib_mlx5_md_t);
    unsigned num_tl_devices   = 0;
    uct_tl_device_resource_t *tl_devices;
    ucs_status_t status;
    CUdevice device;
    ucs_sys_device_t dev;
    ucs_sys_dev_distance_t dist;
    int i, num_gpus;

    /*
    * Save the result of peermem driver check in a global flag to avoid
    * printing diag message for each MD.
    */
    if (peermem_loaded == -1) {
        peermem_loaded = !!(md->super.reg_mem_types &
                            UCS_BIT(UCS_MEMORY_TYPE_CUDA));
        if (peermem_loaded == 0) {
            ucs_diag("GDAKI not supported, please load "
                        "Nvidia peermem driver by running "
                        "\"modprobe nvidia_peermem\"");
        }
    }

    if (peermem_loaded == 0) {
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetCount(&num_gpus));
    if (status != UCS_OK) {
        return status;
    }

    tl_devices = ucs_malloc(sizeof(*tl_devices) * num_gpus, "gdaki_tl_devices");
    if (tl_devices == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (i = 0; i < num_gpus; i++) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&device, i));
        if (status != UCS_OK) {
            goto err;
        }

        /*
         * Save the result of UAR support in a global flag since to avoid the
         * overhead of checking UAR support for each GPU and MD. Assume the
         * support is the same for all GPUs and MDs in the system.
         */
        if (uar_supported == -1) {
            status = uct_gdaki_md_check_uar(md, device);
            if (status == UCS_OK) {
                uar_supported = 1;
            } else {
                ucs_diag("GDAKI not supported, please add "
                         "NVreg_RegistryDwords=\"PeerMappingOverride=1;\" "
                         "option for nvidia kernel driver");
                uar_supported = 0;
            }
        }
        if (uar_supported == 0) {
            status = UCS_ERR_NO_DEVICE;
            goto err;
        }

        uct_cuda_base_get_sys_dev(device, &dev);
        status = ucs_topo_get_distance(dev, md->super.dev.sys_dev, &dist);
        if (status != UCS_OK) {
            goto err;
        }

        /* TODO this logic should be done in UCP */
        if (dist.latency > md->super.config.gda_max_sys_latency) {
            continue;
        }

        snprintf(tl_devices[num_tl_devices].name,
                 sizeof(tl_devices[num_tl_devices].name), "%s%d-%s:%d",
                 UCT_DEVICE_CUDA_NAME, device,
                 uct_ib_device_name(&md->super.dev), md->super.dev.first_port);
        tl_devices[num_tl_devices].type       = UCT_DEVICE_TYPE_NET;
        tl_devices[num_tl_devices].sys_device = dev;
        num_tl_devices++;
    }

    *num_tl_devices_p = num_tl_devices;
    *tl_devices_p     = tl_devices;
    return UCS_OK;

err:
    ucs_free(tl_devices);
out:
    return status;
}

UCT_TL_DEFINE_ENTRY(&uct_ib_component, rc_gda, uct_gdaki_query_tl_devices,
                    uct_rc_gdaki_iface_t, "RC_GDA_",
                    uct_rc_gdaki_iface_config_table,
                    uct_rc_gdaki_iface_config_t);

UCT_TL_INIT(&uct_ib_component, rc_gda, ctor, , )
