export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"

CUDA_VISIBLE_DEVICES=3 python asr_inference_CV.py \
  --ckpt_path "/workspace/KD-via-FM-in-ASR/outputs/FlowMatching_KD/layerwise_flowkd_mlp_linear_sampling8_libri960/checkpoints/last.ckpt" \
  --data_dir "/workspace/KD-via-FM-in-ASR/data/commonvoice" \
  --hf_token "$HF_TOKEN" \
  --meta_encoder_type "mlp" \
  --flow_steps 2 \
  --flow_schedule "rectified" \
  --flow_weight 1.0 \
  --use_ctc True \
  --use_logit_distillation True \
  --use_layerwise_distillation False \
  --use_flow_matching True