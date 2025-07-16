
python asr_inference.py \
  --ckpt_path "/workspace/outputs/LogitKD_CTCLoss/temp2_alpha1/LogitKD_CTCLoss/22stcvq1/checkpoints/epoch=99-step=89200.ckpt" \
  --use_ctc True \
  --use_logit_distillation True \
  --use_layerwise_distillation False \
  --use_flow_matching False \
  --kd_temperature 1.0 \
  --kd_alpha 0.1 \
  --layer_kd_alpha 1.0 \
