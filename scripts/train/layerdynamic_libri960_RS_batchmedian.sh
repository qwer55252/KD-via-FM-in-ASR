export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
export PRJ_NAME="FlowMatching_KD"
export EXP_NAME="layerdynamic_RS_batchmedian"

# 1) 출력 디렉토리 생성
OUTPUT_DIR="./outputs/$PRJ_NAME/$EXP_NAME"
mkdir -p "$OUTPUT_DIR"

# 2) 학습 실행 및 로그 저장
CUDA_VISIBLE_DEVICES=2 python asr_train.py \
--output_dir "$OUTPUT_DIR" \
--data_dir "/workspace/KD-via-FM-in-ASR/data/train_100" \
--data_config_name train_100 \
--data_train_split train.clean.100 \
--data_val_split dev.clean \
--data_test_split test.clean \
--batch_size 32 \
--epochs 100 \
--use_ctc True \
--use_logit_distillation True \
--use_layerwise_distillation False \
--use_flow_matching True \
--kd_temperature 1 \
--kd_alpha 0.1 \
--layer_kd_alpha 1.0 \
--use_dynamic_steps True \
--router_strategy "batch_median" \
--router_weight 1.0 \
--router_temperature 1.0 \
--router_max_sampling_steps 16 \

# > "$OUTPUT_DIR/output_log.txt" 2>&1
