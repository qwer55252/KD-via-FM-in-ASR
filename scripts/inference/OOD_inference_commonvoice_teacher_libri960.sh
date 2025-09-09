export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"

CUDA_VISIBLE_DEVICES=3 python asr_inference_CV.py \
  --ckpt_path "/workspace/KD-via-FM-in-ASR/outputs/FlowMatching_KD/libri960_teacher/checkpoints/last.ckpt" \
  --data_dir "/workspace/KD-via-FM-in-ASR/data/commonvoice" \
  --use_ctc True \
  