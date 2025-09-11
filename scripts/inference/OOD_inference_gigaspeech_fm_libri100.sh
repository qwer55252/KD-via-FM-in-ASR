
python asr_inference.py \
  --ckpt_path "/workspace/KD-via-FM-in-ASR/outputs/FlowMatching_KD/layerwise_flowkd_mlp_linear_sampling8/checkpoints/last.ckpt" \
  --use_ctc True \
  --use_logit_distillation True \
  --use_layerwise_distillation False \
  --use_diffkd False \
  --use_flow_matching True \
  --data_dir /workspace/KD-via-FM-in-ASR/data/gigaspeech \
  --eval_data gigaspeech