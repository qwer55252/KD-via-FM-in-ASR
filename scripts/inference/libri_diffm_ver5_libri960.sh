
python asr_inference_diffm.py \
  --ckpt_path "/workspace/KD-via-FM-in-ASR/outputs/FlowMatching_KD/diffm_ver5_libri960/checkpoints/last.ckpt" \
  --meta_encoder_type "mlp" \
  --flow_steps 2 \
  --flow_schedule "rectified" \
  --flow_weight 1.0 \
  --use_ctc True \
  --use_logit_distillation True \
  --use_layerwise_distillation False \
  --use_flow_matching False \
  --model_ver 5 \
