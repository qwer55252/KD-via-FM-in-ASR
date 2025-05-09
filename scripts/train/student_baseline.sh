export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"

python asr_train.py \
--data_config_name train_100 \
--data_train_split train.clean.100 \
--data_val_split dev.clean \
--data_test_split test.clean \
--batch_size 32