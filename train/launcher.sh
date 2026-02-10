#!/bin/bash
set -e

source venv/bin/activate
current_host=$(hostname)

base_cmd=(
    torchrun 
    --rdzv_backend=c10d 
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" 
    --nnodes="${NNODES}" 
    --nproc_per_node="${GPUS_PER_NODE}" 
    main.py 
    --config_path config/train-dist.yaml
)

log_info() {
    echo "[$(date '+%T') - ${current_host}] $*"
}

wait_for_master() {
    log_info "Waiting for master at ${MASTER_ADDR}:${MASTER_PORT}"
    
    # Bash built-in TCP check instead of nc
    while ! &>/dev/null </dev/tcp/${MASTER_ADDR}/${MASTER_PORT}; do
        sleep 1
    done
    
    log_info "Master ${MASTER_ADDR}:${MASTER_PORT} is ready"
}

if [ "${current_host}" = "${MASTER_ADDR}" ]; then
    log_info "Starting MASTER process"
    "${base_cmd[@]}"
else
    wait_for_master
    log_info "Starting WORKER process"
    "${base_cmd[@]}"
fi