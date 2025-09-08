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
import shutil
import aiohttp
import argparse
import torch
import lightning as pl
import nemo.collections.asr as nemo_asr
from copy import deepcopy
from datasets import load_dataset, DownloadConfig, config as hf_config

# --- import utilities from your training script ---
from asr_train import (
    release_nemoAPI,
    DistilFlowMatchingCTCModelBPE,
)
from huggingface_hub import HfFolder
import uuid
import soundfile as sf  # pip install soundfile
import unicodedata
import regex as re  # pip install regex

PUNCT_MAP = {
    "\u2047": " ",  # DOUBLE QUESTION MARK → 제거(공백)
    "“": '"', "”": '"', "„": '"',
    "‘": "'", "’": "'",
    "–": "-", "—": "-",
    "…": " ", "‹": " ", "›": " ", "«": " ", "»": " ",
}

def normalize_text_cv(s: str, keep_punct: bool = False) -> str:
    # 0) 유니코드 정규화 + 소문자
    s = unicodedata.normalize("NFKC", s or "").strip().lower()

    # 1) 흔한 특수문자 매핑 및 제거
    for k, v in PUNCT_MAP.items():
        s = s.replace(k, v)

    # 2) 바깥 큰따옴표만 한 쌍이면 제거
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]

    # 3) CV 특유의 공백+아포스트로피 정리: "men 's" → "men's"
    s = re.sub(r"\s+'\s*s\b", "'s", s)

    # 4) 평가용: 구두점 제거 권장(문자/숫자/공백/아포스트로피/하이픈만 유지)
    if not keep_punct:
        s = re.sub(r"[^\p{L}\p{N}\s'\-]", " ", s)

    # 5) 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_student_config(teacher_model, args, train_manifest, val_manifest, test_manifest):
    # teacher_model.cfg 를 복사한 student_cfg
    student_cfg = deepcopy(teacher_model.cfg)

    # 새로운 manifest 경로로 덮어쓰기
    student_cfg.train_ds.is_tarred = False # manifest_filepath 기반으로 데이터 Load하기 위한 설정
    student_cfg.train_ds.tarred_audio_filepaths = None
    student_cfg.train_ds.manifest_filepath      = train_manifest
    student_cfg.train_ds.sample_rate            = args.data_sample_rate
    student_cfg.train_ds.batch_size             = args.batch_size

    student_cfg.validation_ds.is_tarred = False
    student_cfg.validation_ds.tarred_audio_filepaths = None
    student_cfg.validation_ds.manifest_filepath = val_manifest
    student_cfg.validation_ds.sample_rate       = args.data_sample_rate
    student_cfg.validation_ds.batch_size        = args.batch_size
    
    
    student_cfg.test_ds.is_tarred = False
    student_cfg.test_ds.tarred_audio_filepaths = None
    student_cfg.test_ds.manifest_filepath       = test_manifest
    student_cfg.test_ds.sample_rate             = args.data_sample_rate
    student_cfg.test_ds.batch_size              = args.batch_size
    
    student_cfg.encoder.d_model = teacher_model.cfg.encoder.d_model // 2
    student_cfg.encoder.n_heads = teacher_model.cfg.encoder.n_heads // 2
    student_cfg.decoder.feat_in = teacher_model.cfg.decoder.feat_in // 2
    
    return student_cfg

def build_manifest_from_hf(ds, manifest_path: str, cache_dir: str):
    """
    HuggingFace Dataset 객체(ds)를 순회하며
    NeMo 형식의 JSON manifest를 생성
    """
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    # 기본 HF_DATASETS_CACHE (원본 오디오가 풀리던 위치)
    default_root = hf_config.HF_DATASETS_CACHE
    extract_marker = os.path.join("extracted")

    # with open(manifest_path, "w") as fout:
    #     for sample in ds:
    #         audio = sample["audio"]
    #         orig_path = audio["path"] 
    #         # sample["audio"]["path"] : '/workspace/data/cache/extracted/28e1f76d85906acbe5672f913bb405be336b2a2aa63d4db4a3d1546fd2728272/2277-149896-0000.flac'
    #         # 실제 데이터 경로 : '/workspace/data/cache/extracted/28e1f76d85906acbe5672f913bb405be336b2a2aa63d4db4a3d1546fd2728272/LibriSpeech/dev-clean/2277/149896/2277-149896-0000.flac'
            

    #         duration = len(audio["array"]) / audio["sampling_rate"]
    #         entry = {
    #             "audio_filepath": orig_path,  # 실제로 존재하는 절대/상대 경로
    #             "duration": duration,
    #             "text": sample["text"].lower().strip(),
    #         }
    #         fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    # HF가 flac을 풀어놓는 최상위 디렉토리
    extract_root = os.path.join(cache_dir, "extracted")

    with open(manifest_path, "w") as fout:
        for sample in ds:
            audio     = sample["audio"]
            orig_path = audio["path"]  # HF가 알려준 경로 (존재하지 않을 수도 있음)

            # 1) 첫 시도: orig_path 에 파일이 실제로 존재하는지
            if not os.path.isfile(orig_path):
                filename = os.path.basename(orig_path)
                # 2) fallback: extract_root 이하를 재귀 검색
                pattern = os.path.join(extract_root, "**", filename)
                matches = glob.glob(pattern, recursive=True)
                if not matches:
                    raise FileNotFoundError(
                        f"Audio 파일을 찾을 수 없습니다: {filename} \n"
                        f"원경로: {orig_path}\n"
                        f"검색경로: {pattern}"
                    )
                # 검색 결과 중 첫 번째를 사용
                orig_path = matches[0]

            duration = len(audio["array"]) / audio["sampling_rate"]
            entry = {
                "audio_filepath": orig_path,
                "duration":        duration,
                # "text":            sample["sentence"].lower().strip(),
                "text": normalize_text_cv(sample["sentence"], keep_punct=False),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_manifest_streaming(ds, manifest_path: str, tmp_audio_dir: str):
    """
    Streaming HF dataset에서 샘플을 읽어 임시 WAV로 저장하고
    NeMo 형식 manifest를 생성한다.
    """
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    os.makedirs(tmp_audio_dir, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as fout:
        for sample in ds:
            audio = sample["audio"]  # {"array": np.ndarray, "sampling_rate": int, "path": str or None}
            arr = audio["array"]
            sr  = audio["sampling_rate"]
            # 임시 wav 저장
            wav_path = os.path.join(tmp_audio_dir, f"cv_{uuid.uuid4().hex}.wav")
            sf.write(wav_path, arr, sr)

            # 텍스트 정리
            text = clean_text_commonvoice(sample["sentence"])
            duration = float(len(arr)) / float(sr)

            fout.write(json.dumps({
                "audio_filepath": wav_path,
                "duration": duration,
                "text": text,
            }, ensure_ascii=False) + "\n")


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
    # manifest 경로 설정
    manifest_dir = os.path.join(args.data_dir, "manifests")
    cache_dir    = os.path.join(args.data_dir, "cache")
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    token = args.hf_token or os.getenv("HF_TOKEN") or HfFolder().get_token()
    # train_manifest = os.path.join(args.data_dir, "manifests", "train-clean-100.json")
    # val_manifest = os.path.join(args.data_dir, "manifests", "validation.json")
    train_manifest = os.path.join(manifest_dir, "train.json")
    val_manifest = os.path.join(manifest_dir, "validation.json")
    test_manifest = os.path.join(manifest_dir, "test.json")

    # 1) HuggingFace LibriSpeech 로드
    print("Datasets cache dir:", hf_config.HF_DATASETS_CACHE)
    
    cache_dir = os.path.join(args.data_dir, "cache")
    hf_config.HF_DATASETS_CACHE = cache_dir
    dl_cfg = DownloadConfig(
        cache_dir=cache_dir,
        force_download=False,
        resume_download=True,
        max_retries=10,
        disable_tqdm=False,
        download_desc="Downloading LibriSpeech ASR",
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=7200)}},
        delete_extracted=False,
        extract_compressed_file=True,
        force_extract=True,            
    )
    
    train_ds = load_dataset(
        "./commonvoice_asr.py",
        "en",
        split="train",
        trust_remote_code=True,
        download_config=dl_cfg,
        cache_dir=cache_dir,
    )
    val_ds = load_dataset(
        "./commonvoice_asr.py",
        "en",
        split="validation",
        trust_remote_code=True,
        download_config=dl_cfg,
        cache_dir=cache_dir,
    )
    test_ds = load_dataset(
        "./commonvoice_asr.py",
        "en",
        split="test",
        trust_remote_code=True,
        download_config=dl_cfg,
        cache_dir=cache_dir,
    )
    # 2) NeMo manifest 생성

    print("building manifest files...")
    if not os.path.isfile(train_manifest):
        build_manifest_from_hf(train_ds, train_manifest, cache_dir)
        print(f"train_manifest DONE: {train_manifest}")
    if not os.path.isfile(val_manifest):
        build_manifest_from_hf(val_ds, val_manifest, cache_dir)
        print(f"val_manifest DONE: {val_manifest}")
    if not os.path.isfile(test_manifest):
        build_manifest_from_hf(test_ds, test_manifest, cache_dir)
        print(f"test_manifest DONE: {test_manifest}")
    print("manifest files built.")

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
    # synthesize a minimal config using teacher cfg as template
    class _ArgsForCfg:
        data_sample_rate = args.data_sample_rate
        batch_size = args.batch_size
    cfg_args = _ArgsForCfg()
    student_cfg = make_student_config(
        teacher_model=teacher, args=cfg_args,
        train_manifest=train_manifest, val_manifest=val_manifest, test_manifest=test_manifest,
    )
    # student_cfg = make_student_config(
    #     teacher_model=teacher, args=cfg_args,
    #     val_manifest=val_m, test_manifest=test_m,
    # )

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

    splits = ["validation", "test", "other", "invalidated"]
    all_metrics = {}

    for split in splits:
        print(f"\n===== Evaluating Common Voice ({args.cv_lang}) :: {split} =====")

        # 공식 데이터셋 + streaming=True (번들 전체 다운로드 안 함)
        # ds = load_dataset(
        #     args.cv_dataset_name,
        #     args.cv_lang,
        #     split=split,
        #     streaming=True,
        #     use_auth_token=token,
        # )
        ds = load_dataset(
            "./commonvoice_asr.py",
            "en",
            split=split,
            trust_remote_code=True,
            download_config=dl_cfg,
            cache_dir=cache_dir,
        )

        # 매니페스트 생성 (임시 wav 저장)
        json_name = split.replace(".", "_") + ".json"
        manifest_i = os.path.join(manifest_dir, json_name)
        if os.path.isfile(manifest_i):
            print(f"manifest exists: {manifest_i}")
        else:
            build_manifest_from_hf(ds, manifest_i, cache_dir)
        
        
        # json_name   = f"commonvoice_{args.cv_lang}_{split}.json"
        # manifest_i  = os.path.join(manifest_dir, json_name)
        # tmp_audio_i = os.path.join(args.data_dir, "tmp_audio", f"{args.cv_lang}_{split}")
        # build_manifest_streaming(ds, manifest_i, tmp_audio_i)

        # 테스트 설정 & 평가
        test_cfg = deepcopy(model.cfg.test_ds)
        test_cfg.manifest_filepath = manifest_i
        test_cfg.sample_rate = args.data_sample_rate
        test_cfg.shuffle = False
        test_cfg.batch_size = args.batch_size
        model.setup_test_data(test_cfg)
        dl = model.test_dataloader()

        results = trainer.test(model=model, dataloaders=[dl], verbose=True)
        res  = results[0]
        wer  = res.get("test_wer", res.get("wer", None))
        loss = res.get("test_loss", res.get("loss", None))
        print(f" → {split} | loss={loss:.4f} | WER={wer:.2%}")
        all_metrics[split] = {"loss": loss, "wer": wer}

    print("\n===== Final OOD Summary (Common Voice) =====")
    for split, m in all_metrics.items():
        print(f"{split:12s} → Loss: {m['loss']:.4f}, WER: {m['wer']:.2%}")


if __name__ == "__main__":
    main()
