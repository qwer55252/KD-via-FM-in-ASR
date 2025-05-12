export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
export EXP_NAME="LogitKD_CTCLoss"

# 1) 출력 디렉토리 생성
OUTPUT_DIR="./outputs/$EXP_NAME/temp1_alpha1"
mkdir -p "$OUTPUT_DIR"

# 2) 학습 실행 및 로그 저장
python asr_train.py \
--output_dir "$OUTPUT_DIR" \
--data_config_name train_100 \
--data_train_split train.clean.100 \
--data_val_split dev.clean \
--data_test_split test.clean \
--batch_size 32 \
--epochs 100 \
--logit_distillation True \
--kd_temperature 1 \
--kd_alpha 1.0
# > "$OUTPUT_DIR/output_log.txt" 2>&1
