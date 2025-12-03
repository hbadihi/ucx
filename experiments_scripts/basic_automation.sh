#!/bin/bash

# Configuration
UCX_BIN="/workspace/ucx/rfs/bin/ucx_perftest"
TEST_TYPE="ucp_put_multi_with_imm_lat"
MEM_TYPE="cuda"
ACCELERATOR="cuda"
START_SIZE=4
MAX_SIZE=$((4 * 1024 * 1024)) # 4 MiB in bytes

# Usage function
usage() {
    echo "Usage: $0 <mode> [server_ip]"
    echo "  mode:      'server' or 'client'"
    echo "  server_ip: Required only if mode is 'client'"
    echo ""
    echo "Examples:"
    echo "  $0 server"
    echo "  $0 client 10.52.48.40"
    exit 1
}

# Argument parsing
MODE=$1
SERVER_IP=$2

if [[ -z "$MODE" ]]; then
    usage
fi

if [[ "$MODE" == "client" && -z "$SERVER_IP" ]]; then
    echo "Error: Server IP is required for client mode."
    usage
fi

if [[ "$MODE" != "client" && "$MODE" != "server" ]]; then
    echo "Error: Invalid mode. Use 'client' or 'server'."
    usage
fi

# Main execution loop
current_size=$START_SIZE

echo "Starting UCX experiments..."
echo "Mode: $MODE"
echo "Range: $START_SIZE bytes to $MAX_SIZE bytes"
echo "------------------------------------------------"

while [ $current_size -le $MAX_SIZE ]; do
    
    # Human readable size for logging
    if [ $current_size -lt 1024 ]; then
        readable_size="${current_size} bytes"
    elif [ $current_size -lt 1048576 ]; then
        readable_size="$((current_size / 1024)) KB"
    else
        readable_size="$((current_size / 1048576)) MiB"
    fi

    echo "[$(date +'%H:%M:%S')] Running test with message size: $readable_size ($current_size)"

    if [[ "$MODE" == "server" ]]; then
        # Server command
        $UCX_BIN -t $TEST_TYPE -m $MEM_TYPE -a $ACCELERATOR -s $current_size
        
        # Check exit status
        if [ $? -ne 0 ]; then
            echo "Error: Server process failed at size $current_size"
            exit 1
        fi
    else
        # Client command
        $UCX_BIN -t $TEST_TYPE -m $MEM_TYPE -a $ACCELERATOR -s $current_size $SERVER_IP
        
        # Check exit status
        if [ $? -ne 0 ]; then
            echo "Error: Client process failed at size $current_size"
            exit 1
        fi

        # Delay to allow server to catch up/restart
        echo "Waiting 1 second..."
        sleep 1
    fi

    # Double the size
    current_size=$((current_size * 2))

done

echo "------------------------------------------------"
echo "All experiments completed successfully."
