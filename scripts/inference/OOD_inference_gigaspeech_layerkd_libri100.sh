
python asr_inference.py \
  --ckpt_path "/workspace/outputs/FlowMatching_KD/real_LayerKD_alpha01/checkpoints/last.ckpt" \
  --use_ctc True \
  --use_logit_distillation True \
  --use_layerwise_distillation True \
  --use_diffkd False \
  --use_flow_matching False \
  --data_dir /workspace/KD-via-FM-in-ASR/data/gigaspeech \
  --eval_data gigaspeech