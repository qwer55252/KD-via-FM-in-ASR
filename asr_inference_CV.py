#!/usr/bin/env python3
# asr_inference_commonvoice.py
"""
OOD evaluation on Common Voice using a Libri-trained NeMo CTC (optionally Flow-Matching KD) model.

- Loads Common Voice via HF Datasets (requires HF token).
- Builds NeMo-style JSON manifests from Common Voice splits.
- Runs evaluation per split and prints WER / loss.
"""

import os
import re
import glob
import json
import argparse
import torch
import lightning as pl
import nemo.collections.asr as nemo_asr
from copy import deepcopy
from datasets import load_dataset, DownloadConfig, config as hf_config

# --- import utilities from your training script ---
from asr_train import (
    release_nemoAPI,
    make_student_config,
    DistilFlowMatchingCTCModelBPE,
)

# ---------------------------
# utils
# ---------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def clean_text_commonvoice(s: str) -> str:
    """
    HF 권장: 양쪽 인용부호 제거, 문장부호 없으면 마침표 추가.
    필요 시 소문자화 등 추가 가능.
    """
    s = s.strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    if len(s) > 0 and s[-1] not in [".", "?", "!"]:
        s = s + "."
    # 여분 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_manifest_from_commonvoice(ds, manifest_path: str, cache_dir: str):
    """
    Common Voice HF Dataset -> NeMo manifest
    fields:
      audio_filepath, duration, text
    Note: CV 오디오는 대개 mp3(48kHz). NeMo가 mp3를 읽으려면 audioread 등 백엔드가 필요.
    """
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    extract_root = os.path.join(cache_dir, "extracted")

    with open(manifest_path, "w", encoding="utf-8") as fout:
        for sample in ds:
            audio = sample["audio"]
            # HF 스크립트에서 'audio.path'는 로컬 절대경로로 설정되는 편이지만,
            # 드물게 상대가 남아있을 수 있어 fallback 검색을 둠.
            orig_path = audio.get("path", None)
            if not orig_path or not os.path.isfile(orig_path):
                # 파일명으로 cache/extracted 아래 재귀 검색
                filename = os.path.basename(sample["path"]) if sample.get("path") else None
                if filename is None:
                    # 마지막 수단: audio.path가 있다면 그걸 파일명으로 사용
                    filename = os.path.basename(orig_path) if orig_path else None
                if filename:
                    matches = glob.glob(os.path.join(extract_root, "**", filename), recursive=True)
                    if matches:
                        orig_path = matches[0]
                # 그래도 못 찾으면 audio.path 를 그대로 사용(스트리밍/원격 가능)
            if not orig_path:
                # 스트리밍 모드 등: path 가 없을 수 있음 -> 이 경우 스킵
                # (원한다면 bytes를 임시파일로 쓸 수도 있음)
                continue

            sr = audio.get("sampling_rate", 16000)
            arr = audio.get("array", None)
            if arr is not None:
                duration = float(len(arr)) / float(sr)
            else:
                # 배열이 없으면 대략 추정 불가 -> 0.0으로 둠(NeMo는 없어도 읽음)
                duration = 0.0

            text = clean_text_commonvoice(sample["sentence"])
            entry = {
                "audio_filepath": orig_path,
                "duration": duration,
                "text": text,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ---------------------------
# args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Common Voice OOD evaluation for a Libri-trained model")
    # ckpt / runtime
    p.add_argument("--ckpt_path", type=str, required=True, help="trained checkpoint (.ckpt)")
    p.add_argument("--data_dir", type=str, default="data", help="root dir to place cache/manifests")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--data_sample_rate", type=int, default=16000, help="model sample rate (NeMo preprocessor)")

    # model / KD knobs (keep parity with training flags you used)
    p.add_argument("--use_ctc", type=str2bool, default=True)
    p.add_argument("--use_logit_distillation", type=str2bool, default=False)
    p.add_argument("--use_layerwise_distillation", type=str2bool, default=False)
    p.add_argument("--kd_temperature", type=float, default=1.0)
    p.add_argument("--kd_alpha", type=float, default=0.1)
    p.add_argument("--layer_kd_alpha", type=float, default=1.0)

    # Flow-Matching options
    p.add_argument("--use_flow_matching", type=str2bool, default=False)
    p.add_argument("--meta_encoder_type", type=str, default="mlp", choices=["mlp", "cnn", "swin", "conformer", "unet"])
    p.add_argument("--flow_steps", type=int, default=2)
    p.add_argument("--flow_schedule", type=str, default="rectified", choices=["rectified", "vp_ode", "ve_ode"])
    p.add_argument("--flow_weight", type=float, default=1.0)

    # HF dataset (Common Voice)
    p.add_argument("--cv_dataset_name", type=str, default="mozilla-foundation/common_voice_7_0",
                   help="HF dataset id for Common Voice (e.g., mozilla-foundation/common_voice_7_0 or _8_0)")
    p.add_argument("--cv_lang", type=str, default="en", help="language code (e.g., en, de, fr, ...)")
    p.add_argument("--cv_splits", type=str, default="validation,test",
                   help="comma-separated CV splits to evaluate (e.g., validation,test,train,other)")
    p.add_argument("--hf_token", type=str, default=None, help="HF token (required by Common Voice)")

    return p.parse_args()

# ---------------------------
# main
# ---------------------------
def main():
    args = parse_args()

    # 0) dirs & cache
    manifest_dir = os.path.join(args.data_dir, "manifests")
    cache_dir = os.path.join(args.data_dir, "cache")
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    hf_config.HF_DATASETS_CACHE = cache_dir

    # 1) trainer
    trainer = pl.Trainer(accelerator="gpu", devices=args.gpus)

    # 2) teacher (for KD wrappers)
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name="stt_en_conformer_ctc_small",
        map_location="cuda:0",
        trainer=trainer,
    )
    teacher.eval()
    release_nemoAPI(teacher)

    # 3) student cfg (re-use your train/val/test manifests if present; not strictly used at test time)
    train_m = os.path.join(manifest_dir, "train.json")
    val_m   = os.path.join(manifest_dir, "val.json")
    test_m  = os.path.join(manifest_dir, "test.json")
    # synthesize a minimal config using teacher cfg as template
    class _ArgsForCfg:
        data_sample_rate = args.data_sample_rate
        batch_size = args.batch_size
    cfg_args = _ArgsForCfg()
    student_cfg = make_student_config(
        teacher_model=teacher, args=cfg_args,
        train_manifest=train_m, val_manifest=val_m, test_manifest=test_m,
    )

    # 4) build model & load checkpoint
    flow_cfg = {
        "meta_encoder_type": args.meta_encoder_type,
        "feature_dim": student_cfg.encoder.d_model,
        "time_embed_dim": 32,
        "hidden_dim": 128,
        "training_sampling": args.flow_steps,
        "inference_sampling": args.flow_steps,
        "weight": args.flow_weight,
        "noise_schedule": args.flow_schedule,
        "loss": "mse",
        "shape_transform": "linear",
        "student_dim": student_cfg.encoder.d_model,
        "teacher_dim": teacher.cfg.encoder.d_model,
        "student_head_num": student_cfg.encoder.n_heads,
        "teacher_head_num": teacher.cfg.encoder.n_heads,
    }

    model = DistilFlowMatchingCTCModelBPE(
        cfg=student_cfg,
        trainer=trainer,
        teacher_model=teacher,
        use_ctc=args.use_ctc,
        use_logit_distillation=args.use_logit_distillation,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
        use_layerwise_distillation=args.use_layerwise_distillation,
        layer_kd_alpha=args.layer_kd_alpha,
        use_flow_matching=args.use_flow_matching,
        flow_cfg=flow_cfg,
    )
    model.eval()

    # load state dict
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    model.load_state_dict(state, strict=False)

    # 5) evaluate per CV split
    dl_cfg = DownloadConfig(
        cache_dir=cache_dir,
        token=args.hf_token,           # Common Voice requires auth token
        force_download=False,
        resume_download=True,
        max_retries=10,
        disable_tqdm=True,
        extract_compressed_file=True,
        delete_extracted=False,
    )

    splits = ["validation", "test"]
    all_metrics = {}

    for split in splits:
        print(f"\n===== Evaluating Common Voice ({args.cv_lang}) :: {split} =====")
        ds = load_dataset(
            "./commonvoice_asr.py",
            "en",
            split=split,
            trust_remote_code=True,
            download_config=dl_cfg,
            cache_dir=cache_dir,
        )

        # build manifest
        json_name = f"commonvoice_{args.cv_lang}_{split.replace('.', '_')}.json"
        manifest_i = os.path.join(manifest_dir, json_name)
        build_manifest_from_commonvoice(ds, manifest_i, cache_dir)

        # setup test loader
        test_cfg = deepcopy(model.cfg.test_ds)
        test_cfg.manifest_filepath = manifest_i
        test_cfg.sample_rate = args.data_sample_rate
        test_cfg.shuffle = False
        test_cfg.batch_size = args.batch_size
        model.setup_test_data(test_cfg)
        dl = model.test_dataloader()

        # run test
        results = trainer.test(model=model, dataloaders=[dl], verbose=True)
        res  = results[0]
        wer  = res.get("test_wer", res.get("wer", None))
        loss = res.get("test_loss", res.get("loss", None))
        print(f" → {split} | loss={loss:.4f} | WER={wer:.2%}")
        all_metrics[split] = {"loss": loss, "wer": wer}

    # summary
    print("\n===== Final OOD Summary (Common Voice) =====")
    for split, m in all_metrics.items():
        print(f"{split:12s} → Loss: {m['loss']:.4f}, WER: {m['wer']:.2%}")


if __name__ == "__main__":
    main()
