export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
export PRJ_NAME="FlowMatching_KD"
export EXP_NAME="DS_GS_teacher"

# 1) 출력 디렉토리 생성
OUTPUT_DIR="./outputs/$PRJ_NAME/$EXP_NAME"
mkdir -p "$OUTPUT_DIR"

# 2) 학습 실행 및 로그 저장
CUDA_VISIBLE_DEVICES=1 python asr_train_GS.py \
--output_dir "$OUTPUT_DIR" \
--data_dir "/workspace/KD-via-FM-in-ASR/data/gigaspeech" \
--data_script_path ./gigaspeech.py \
--data_config_name xs \
--data_train_split train \
--data_val_split validation \
--data_test_split test \
--batch_size 16 \
--epochs 100 \
--use_ctc True \
--use_logit_distillation False \
--use_layerwise_distillation False \
--use_flow_matching False \
--train_teacher True \

# > "$OUTPUT_DIR/output_log.txt" 2>&1