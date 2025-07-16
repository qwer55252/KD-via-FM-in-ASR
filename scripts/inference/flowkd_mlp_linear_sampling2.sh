
python asr_inference.py \
  --ckpt_path "/workspace/outputs/FlowMatching_KD/flowkd_mlp_linear_sampling2/checkpoints/last-v3.ckpt" \
  --meta_encoder_type "mlp" \
  --flow_steps 2 \
  --flow_schedule "rectified" \
  --flow_weight 1.0 \
  --use_ctc True \
  --use_logit_distillation True \
  --use_layerwise_distillation False \
  --use_flow_matching True \
