#!/bin/bash
set -e

GPU="0"
CONFIG="arguments/configs/csi/exp1.yaml"

# Detected on this machine: 24 logical CPU threads.
# Use roughly the physical-core count to avoid CPU thread oversubscription while
# the CUDA renderer and PyTorch kernels are running.
CPU_THREADS="12"
MAX_GPU_MEM_GB="24"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --gpu)    GPU="$2"; shift 2 ;;
        --threads) CPU_THREADS="$2"; shift 2 ;;
        --max-gpu-mem-gb) MAX_GPU_MEM_GB="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ ! -f "$CONFIG" ]; then echo "Error: Config not found: $CONFIG"; exit 1; fi

export CUDA_VISIBLE_DEVICES="$GPU"

# CPU-side threading for NumPy/PyTorch BLAS/OpenMP operators.
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

# GPU-side acceleration and allocator settings.
# NVIDIA_TF32_OVERRIDE enables TensorFloat-32 matmul on supported RTX/Ampere+ GPUs.
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHONUNBUFFERED=1 python main_csi.py --config "$CONFIG" --max_gpu_mem_gb "$MAX_GPU_MEM_GB"
