export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
export PRJ_NAME="FlowMatching_KD"
export EXP_NAME="DS_ted1_student"

# 1) 출력 디렉토리 생성
OUTPUT_DIR="./outputs/$PRJ_NAME/$EXP_NAME"
mkdir -p "$OUTPUT_DIR"

# 2) 학습 실행 및 로그 저장
CUDA_VISIBLE_DEVICES=0 python asr_train.py \
--output_dir "$OUTPUT_DIR" \
--data_dir "/workspace/KD-via-FM-in-ASR/data/tedlium1" \
--data_script_path ./tedlium_asr.py \
--data_config_name release1 \
--data_train_split train \
--data_val_split validation \
--data_test_split test \
--batch_size 32 \
--epochs 100 \
--use_ctc True \
--use_logit_distillation False \
--use_layerwise_distillation False \
--use_flow_matching False \

# > "$OUTPUT_DIR/output_log.txt" 2>&1