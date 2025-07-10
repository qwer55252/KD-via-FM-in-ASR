
python asr_inference.py \
  --ckpt_path "/workspace/KD-via-FM-in-ASR/outputs/FlowMatching_KD/only_flow_sampling4/checkpoints/last.ckpt" \
  --meta_encoder_type "mlp" \
  --flow_steps 4 \
  --flow_schedule "rectified" \
  --flow_weight 1.0