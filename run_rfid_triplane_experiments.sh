#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-gsrf}"
DEFAULT_RATIO="${DEFAULT_RATIO:-0.8}"     # default RFID split used by the local GSRF config
SPARSE_RATIO="${SPARSE_RATIO:-0.03593}"  # 6123 * 0.03593 ~= 220 RFID training samples
SEED="${SEED:-8371}"
ITER_7K="${ITER_7K:-7000}"
ITER_SHORT="${ITER_SHORT:-10000}"
ITER_FULL="${ITER_FULL:-30000}"

BASE_CFG="arguments/configs/rfid/exp1.yaml"
TRI_CFG="arguments/configs/rfid/exp_triplane.yaml"
RUN_LOG_DIR="logs/rfid_experiment_runs"

ACTIVE_RATIO="${DEFAULT_RATIO}"
ACTIVE_TRAIN_INDEX="train_index_default_seed${SEED}.txt"
ACTIVE_TEST_INDEX="test_index_default_seed${SEED}.txt"

mkdir -p "${RUN_LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"

run_train() {
    local config="$1"
    local exp_name="$2"
    local iterations="$3"
    shift 3

    echo
    echo "========== TRAIN ${exp_name} (${iterations} iters) =========="
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n "${ENV_NAME}" python train_rfid.py \
        --config "${config}" \
        --exp_name "${exp_name}" \
        --iterations "${iterations}" \
        --ratio_train "${ACTIVE_RATIO}" \
        --train_index_path "${ACTIVE_TRAIN_INDEX}" \
        --test_index_path "${ACTIVE_TEST_INDEX}" \
        "$@" 2>&1 | tee "${RUN_LOG_DIR}/${exp_name}_train.log"
}

run_infer() {
    local config="$1"
    local exp_name="$2"
    local iteration="$3"

    echo
    echo "========== INFER ${exp_name} (${iteration}) =========="
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n "${ENV_NAME}" python inference_rfid.py \
        --config "${config}" \
        --exp_name "${exp_name}" \
        --iterations "${iteration}" \
        --ratio_train "${ACTIVE_RATIO}" \
        --train_index_path "${ACTIVE_TRAIN_INDEX}" \
        --test_index_path "${ACTIVE_TEST_INDEX}" \
        2>&1 | tee "${RUN_LOG_DIR}/${exp_name}_infer_${iteration}.log"
}

run_train_and_infer() {
    local config="$1"
    local exp_name="$2"
    local iterations="$3"
    shift 3
    run_train "${config}" "${exp_name}" "${iterations}" "$@"
    run_infer "${config}" "${exp_name}" "${iterations}"
}

use_default_split() {
    ACTIVE_RATIO="${DEFAULT_RATIO}"
    ACTIVE_TRAIN_INDEX="train_index_default_seed${SEED}.txt"
    ACTIVE_TEST_INDEX="test_index_default_seed${SEED}.txt"
}

use_sparse220_split() {
    ACTIVE_RATIO="${SPARSE_RATIO}"
    ACTIVE_TRAIN_INDEX="train_index_sparse220_seed${SEED}.txt"
    ACTIVE_TEST_INDEX="test_index_sparse220_seed${SEED}.txt"
}

main_default7k() {
    use_default_split
    run_train_and_infer "${BASE_CFG}" "baseline_default_7k" "${ITER_7K}"
    run_train_and_infer "${TRI_CFG}" "triplane_default_7k" "${ITER_7K}" \
        --triplane_warmup_iters 500
}

triplane_default7k() {
    use_default_split
    run_train_and_infer "${TRI_CFG}" "triplane_default_7k" "${ITER_7K}" \
        --triplane_warmup_iters 500
}

triplane_safe7k() {
    use_default_split
    run_train_and_infer "${TRI_CFG}" "triplane_safe_default_7k" "${ITER_7K}" \
        --triplane_decoder_type mlp \
        --triplane_hidden_dim 64 \
        --triplane_warmup_iters 300 \
        --triplane_offset_radius_scale 0.05 \
        --triplane_offset_l2 0.05 \
        --triplane_att_radius 0.1 \
        --triplane_att_l2 0.01 \
        --triplane_finalize_tx_samples 32
}

triplane_film7k() {
    use_default_split
    run_train_and_infer "${TRI_CFG}" "triplane_film_default_7k" "${ITER_7K}" \
        --triplane_decoder_type film_fourier \
        --triplane_hidden_dim 128 \
        --triplane_fourier_freqs 4 \
        --triplane_film_layers 3 \
        --triplane_warmup_iters 300 \
        --triplane_offset_radius_scale 0.05 \
        --triplane_offset_l2 0.05 \
        --triplane_att_radius 0.1 \
        --triplane_att_l2 0.01 \
        --triplane_finalize_tx_samples 32
}

triplane_null7k() {
    use_default_split
    run_train_and_infer "${TRI_CFG}" "triplane_null_default_7k" "${ITER_7K}" \
        --triplane_warmup_iters 0
}

main_default10k() {
    use_default_split
    run_train_and_infer "${BASE_CFG}" "baseline_default_10k" "${ITER_SHORT}"
    run_train_and_infer "${TRI_CFG}" "triplane_film_default_10k" "${ITER_SHORT}" \
        --triplane_decoder_type film_fourier \
        --triplane_hidden_dim 128 \
        --triplane_fourier_freqs 4 \
        --triplane_film_layers 3 \
        --triplane_warmup_iters 300 \
        --triplane_offset_radius_scale 0.05 \
        --triplane_offset_l2 0.05 \
        --triplane_att_radius 0.1 \
        --triplane_att_l2 0.01 \
        --triplane_finalize_tx_samples 32
}

triplane_safe10k() {
    use_default_split
    run_train_and_infer "${TRI_CFG}" "triplane_safe_default_10k" "${ITER_SHORT}" \
        --triplane_decoder_type mlp \
        --triplane_hidden_dim 64 \
        --triplane_warmup_iters 300 \
        --triplane_offset_radius_scale 0.05 \
        --triplane_offset_l2 0.05 \
        --triplane_att_radius 0.1 \
        --triplane_att_l2 0.01 \
        --triplane_finalize_tx_samples 32
}

triplane_film10k() {
    use_default_split
    run_train_and_infer "${TRI_CFG}" "triplane_film_default_10k" "${ITER_SHORT}" \
        --triplane_decoder_type film_fourier \
        --triplane_hidden_dim 128 \
        --triplane_fourier_freqs 4 \
        --triplane_film_layers 3 \
        --triplane_warmup_iters 300 \
        --triplane_offset_radius_scale 0.05 \
        --triplane_offset_l2 0.05 \
        --triplane_att_radius 0.1 \
        --triplane_att_l2 0.01 \
        --triplane_finalize_tx_samples 32
}

triplane_gate10k() {
    use_default_split
    run_train_and_infer "${TRI_CFG}" "triplane_gate_default_10k" "${ITER_SHORT}" \
        --triplane_output_mode gating_only \
        --triplane_decoder_type film_fourier \
        --triplane_hidden_dim 128 \
        --triplane_fourier_freqs 4 \
        --triplane_film_layers 3 \
        --triplane_warmup_iters 300 \
        --triplane_offset_radius_scale 0.0 \
        --triplane_offset_l2 0.0 \
        --triplane_att_radius 0.05 \
        --triplane_att_l2 0.02 \
        --triplane_finalize_tx_samples 32
}

triplane_gate_mlp10k() {
    use_default_split
    run_train_and_infer "${TRI_CFG}" "triplane_gate_mlp_default_10k" "${ITER_SHORT}" \
        --triplane_output_mode gating_only \
        --triplane_decoder_type mlp \
        --triplane_hidden_dim 128 \
        --triplane_warmup_iters 300 \
        --triplane_offset_radius_scale 0.0 \
        --triplane_offset_l2 0.0 \
        --triplane_att_radius 0.05 \
        --triplane_att_l2 0.02 \
        --triplane_finalize_tx_samples 32
}

triplane_gate_haar10k() {
    use_default_split
    run_train_and_infer "${TRI_CFG}" "triplane_gate_haar_default_10k" "${ITER_SHORT}" \
        --triplane_output_mode gating_only \
        --triplane_field_type haar \
        --triplane_wavelet_levels 2 \
        --triplane_wavelet_l1 0.001 \
        --triplane_wavelet_c2f \
        --triplane_decoder_type film_fourier \
        --triplane_hidden_dim 128 \
        --triplane_fourier_freqs 4 \
        --triplane_film_layers 3 \
        --triplane_warmup_iters 300 \
        --triplane_offset_radius_scale 0.0 \
        --triplane_offset_l2 0.0 \
        --triplane_att_radius 0.05 \
        --triplane_att_l2 0.02 \
        --triplane_finalize_tx_samples 32
}

triplane_gate_haar30k() {
    use_default_split
    run_train_and_infer "${TRI_CFG}" "triplane_gate_haar_default_30k" "${ITER_FULL}" \
        --triplane_output_mode gating_only \
        --triplane_field_type haar \
        --triplane_wavelet_levels 2 \
        --triplane_wavelet_l1 0.001 \
        --triplane_wavelet_c2f \
        --triplane_decoder_type film_fourier \
        --triplane_hidden_dim 128 \
        --triplane_fourier_freqs 4 \
        --triplane_film_layers 3 \
        --triplane_warmup_iters 300 \
        --triplane_offset_radius_scale 0.0 \
        --triplane_offset_l2 0.0 \
        --triplane_att_radius 0.05 \
        --triplane_att_l2 0.02 \
        --triplane_finalize_tx_samples 32
}

triplane_null10k() {
    use_default_split
    run_train_and_infer "${TRI_CFG}" "triplane_null_default_10k" "${ITER_SHORT}" \
        --triplane_warmup_iters 0
}

main_sparse7k() {
    use_sparse220_split
    run_train_and_infer "${BASE_CFG}" "baseline_sparse220_7k" "${ITER_7K}"
    run_train_and_infer "${TRI_CFG}" "triplane_sparse220_7k" "${ITER_7K}" \
        --triplane_warmup_iters 500
}

warmup_ablation7k() {
    use_default_split
    for warmup in 100 500 1000; do
        run_train_and_infer "${TRI_CFG}" "triplane_warmup${warmup}_default_7k" "${ITER_7K}" \
            --triplane_warmup_iters "${warmup}"
    done
}

resolution_ablation7k() {
    use_default_split
    for res in 32 64 128; do
        run_train_and_infer "${TRI_CFG}" "triplane_res${res}_default_7k" "${ITER_7K}" \
            --triplane_resolution "${res}" \
            --triplane_warmup_iters 500
    done
}

offset_ablation7k() {
    use_default_split
    for scale in 0.25 0.5 1.0; do
        exp_scale="${scale/./p}"
        run_train_and_infer "${TRI_CFG}" "triplane_offset${exp_scale}_default_7k" "${ITER_7K}" \
            --triplane_offset_radius_scale "${scale}" \
            --triplane_warmup_iters 500
    done
}

final_default30k() {
    use_default_split
    run_train_and_infer "${BASE_CFG}" "baseline_default_30k" "${ITER_FULL}"
    run_train_and_infer "${TRI_CFG}" "triplane_default_30k" "${ITER_FULL}" \
        --triplane_warmup_iters 500
}

final_sparse30k() {
    use_sparse220_split
    run_train_and_infer "${BASE_CFG}" "baseline_sparse220_30k" "${ITER_FULL}"
    run_train_and_infer "${TRI_CFG}" "triplane_sparse220_30k" "${ITER_FULL}" \
        --triplane_warmup_iters 500
}

case "${1:-main_default7k}" in
    main_default7k)
        main_default7k
        ;;
    main_default10k)
        main_default10k
        ;;
    triplane_default7k)
        triplane_default7k
        ;;
    triplane_safe7k)
        triplane_safe7k
        ;;
    triplane_film7k)
        triplane_film7k
        ;;
    triplane_safe10k)
        triplane_safe10k
        ;;
    triplane_film10k)
        triplane_film10k
        ;;
    triplane_gate10k)
        triplane_gate10k
        ;;
    triplane_gate_mlp10k)
        triplane_gate_mlp10k
        ;;
    triplane_gate_haar10k)
        triplane_gate_haar10k
        ;;
    triplane_gate_haar30k)
        triplane_gate_haar30k
        ;;
    triplane_null10k)
        triplane_null10k
        ;;
    triplane_null7k)
        triplane_null7k
        ;;
    main_sparse7k)
        main_sparse7k
        ;;
    warmup_ablation7k)
        warmup_ablation7k
        ;;
    resolution_ablation7k)
        resolution_ablation7k
        ;;
    offset_ablation7k)
        offset_ablation7k
        ;;
    final_default30k)
        final_default30k
        ;;
    final_sparse30k)
        final_sparse30k
        ;;
    all7k)
        main_default7k
        warmup_ablation7k
        resolution_ablation7k
        offset_ablation7k
        ;;
    *)
        echo "Usage: $0 {main_default7k|main_default10k|triplane_default7k|triplane_safe7k|triplane_film7k|triplane_null7k|triplane_safe10k|triplane_film10k|triplane_gate10k|triplane_gate_mlp10k|triplane_gate_haar10k|triplane_gate_haar30k|triplane_null10k|main_sparse7k|warmup_ablation7k|resolution_ablation7k|offset_ablation7k|final_default30k|final_sparse30k|all7k}"
        exit 2
        ;;
esac
