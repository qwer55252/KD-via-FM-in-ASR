
python asr_inference.py \
  --ckpt_path "/workspace/KD-via-FM-in-ASR/outputs/FlowMatching_KD/libri960_teacher/checkpoints/last.ckpt" \
  --use_ctc True \
  --use_logit_distillation False \
  --use_layerwise_distillation False \
  --use_diffkd False \
  --use_flow_matching False \
  --data_dir /workspace/KD-via-FM-in-ASR/data/gigaspeech \
  --eval_data gigaspeech \
  --is_teacher True
  