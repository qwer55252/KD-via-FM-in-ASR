export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"

python asr_inference_CV.py \
  --ckpt_path "/workspace/KD-via-FM-in-ASR/outputs/FlowMatching_KD/only_ctc/checkpoints/last.ckpt" \
  --data_dir "/workspace/KD-via-FM-in-ASR/data/commonvoice" \
  --use_ctc True \
  --use_logit_distillation False \
  --use_layerwise_distillation False \
  --use_flow_matching False