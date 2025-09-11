
python asr_inference.py \
  --ckpt_path "/workspace/outputs/teacher_baseline_epochs100/teacher_baseline_epochs100/iod7s4bk/checkpoints/epoch=99-step=89200.ckpt" \
  --use_ctc True \
  --use_logit_distillation False \
  --use_layerwise_distillation False \
  --use_diffkd False \
  --use_flow_matching False \
  --data_dir /workspace/KD-via-FM-in-ASR/data/gigaspeech \
  --eval_data gigaspeech \
  --is_teacher True \
  