export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
export PRJ_NAME="FlowMatching_KD"
export EXP_NAME="libri960_logitkd"

# 1) 출력 디렉토리 생성
OUTPUT_DIR="./outputs/$PRJ_NAME/$EXP_NAME"
mkdir -p "$OUTPUT_DIR"

# 2) 학습 실행 및 로그 저장
CUDA_VISIBLE_DEVICES=2 python asr_train.py \
--output_dir "$OUTPUT_DIR" \
--data_dir "/workspace/KD-via-FM-in-ASR/data/all" \
--data_config_name all \
--data_train_split "train.clean.100+train.clean.360+train.other.500" \
--data_val_split dev.clean \
--data_test_split test.clean \
--batch_size 32 \
--epochs 100 \
--use_ctc True \
--use_logit_distillation True \
--use_layerwise_distillation False \
--use_flow_matching False \

# > "$OUTPUT_DIR/output_log.txt" 2>&1
