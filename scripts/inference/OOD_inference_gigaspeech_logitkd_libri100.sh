
python asr_inference.py \
  --ckpt_path "/workspace/outputs/LogitKD_CTCLoss/temp1_alpha01/LogitKD_CTCLoss/dqcpjya2/checkpoints/epoch=99-step=89200.ckpt" \
  --use_ctc True \
  --use_logit_distillation True \
  --use_layerwise_distillation False \
  --use_diffkd False \
  --use_flow_matching False \
  --data_dir /workspace/KD-via-FM-in-ASR/data/gigaspeech \
  --eval_data gigaspeech