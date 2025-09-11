
python asr_inference.py \
  --ckpt_path "/workspace/outputs/baseline_epochs100/baseline_epochs100/xeswp4uk/checkpoints/epoch=99-step=89200.ckpt" \
  --use_ctc True \
  --use_logit_distillation False \
  --use_layerwise_distillation False \
  --use_diffkd False \
  --use_flow_matching False \
  --data_dir /workspace/KD-via-FM-in-ASR/data/gigaspeech \
  --eval_data gigaspeech