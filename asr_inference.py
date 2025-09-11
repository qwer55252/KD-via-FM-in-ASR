#!/usr/bin/env python3
# inference_splits.py

import os
import re
import regex as re_u
import json
import uuid
import argparse
import glob
import unicodedata
import csv
import torch
import lightning as pl
import soundfile as sf
import nemo.collections.asr as nemo_asr
from datasets import load_dataset, DownloadConfig, config as hf_config
from copy import deepcopy
from asr_train import (
    release_nemoAPI,
    make_student_config,
    build_manifest_from_hf,
    DistilFlowMatchingCTCModelBPE,
)

def normalize_text_cv(s: str, keep_punct: bool = False) -> str:
    # 0) 유니코드 정규화 + 소문자
    s = unicodedata.normalize("NFKC", s or "").strip().lower()

    # 1) 흔한 특수문자 매핑 및 제거
    for k, v in {"\u2047": " ","“": '"', "”": '"', "„": '"',"‘": "'", "’": "'","–": "-", "—": "-","…": " ", "‹": " ", "›": " ", "«": " ", "»": " ",}.items():   # DOUBLE QUESTION MARK → 제거(공백)
        s = s.replace(k, v)

    # 2) 바깥 큰따옴표만 한 쌍이면 제거
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]

    # 3) CV 특유의 공백+아포스트로피 정리: "men 's" → "men's"
    s = re.sub(r"\s+'\s*s\b", "'s", s)

    # 4) 평가용: 구두점 제거 권장(문자/숫자/공백/아포스트로피/하이픈만 유지)
    if not keep_punct:
        s = re_u.sub(r"[^\p{L}\p{N}\s'\-]", " ", s)

    # 5) 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_manifest_from_hf_gigaspeech(ds, manifest_path: str, cache_dir: str):
    """
    HF Dataset -> NeMo manifest(JSONL)
      { "audio_filepath": ..., "duration": ..., "text": ... }
    - path 우선 사용, 안되면 cache/extracted 재귀탐색,
      그래도 안되면 bytes/array를 임시 파일로 저장
    """
    import io
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    extract_root  = os.path.join(cache_dir, "extracted")
    tmp_audio_dir = os.path.join(cache_dir, "tmp_manifest_audio")
    os.makedirs(tmp_audio_dir, exist_ok=True)

    # [ADDED] 레퍼런스 내 특수 태그 목록 (대소문자 무시)
    BANNED_TAGS = {
        "<MUSIC>", "<COMMA>", "<NOISE>", "<VOCALIZED_NOISE>", "<LAUGHTER>",
        "<SPOKEN_NOISE>", "<PERIOD>", "<QUESTION_MARK>", "<EXCLAMATION_MARK>",
        "<SEMICOLON>", "<COLON>", "<DASH>", "<ELLIPSIS>", "<SIL>", "<OTHER>"
    }
    # [ADDED] 태그 스트립용 정규식 (대소문자 무시)
    import re as _re
    _TAGS_RE = _re.compile(r"(?:%s)" % "|".join(_re.escape(t) for t in BANNED_TAGS), _re.IGNORECASE)

    # [ADDED] 태그만 제거하되, 제거 후 빈 문자열이면 "태그만 존재"로 판단
    def _strip_special_tags(text: str) -> tuple[str, bool]:
        """
        Returns: (tags_removed_text, is_tag_only)
        - 텍스트에서 BANNED_TAGS에 해당하는 토큰을 모두 제거
        - 제거 결과가 빈 문자열이면 '태그만 있는' 케이스로 간주
        """
        if not text:
            return "", True
        no_tags = _TAGS_RE.sub(" ", text)
        no_tags = _re.sub(r"\s+", " ", no_tags).strip()
        is_tag_only = (len(no_tags) == 0)
        return no_tags, is_tag_only

    def _resolve_audio_path(sample) -> tuple[str, float]:
        """오디오 파일 실제 경로와 duration(sec) 반환"""
        audio = sample["audio"]
        # 1) path가 실제 파일이면 그대로 사용
        orig_path = audio.get("path", None)
        if isinstance(orig_path, str) and os.path.isfile(orig_path):
            # duration이 없으면 soundfile로 계산
            arr = audio.get("array", None)
            sr  = audio.get("sampling_rate", 16000)
            if arr is not None:
                dur = float(len(arr)) / float(sr)
            else:
                try:
                    info = sf.info(orig_path)
                    dur = float(info.frames) / float(info.samplerate)
                except Exception:
                    dur = 0.0
            return orig_path, dur

        # 2) cache/extracted 아래에서 파일명으로 재귀 검색
        cand_name = None
        if isinstance(orig_path, str) and len(orig_path) > 0:
            cand_name = os.path.basename(orig_path)
        elif isinstance(sample.get("path", None), str):
            cand_name = os.path.basename(sample["path"])

        if cand_name:
            matches = glob.glob(os.path.join(extract_root, "**", cand_name), recursive=True)
            if matches:
                found = matches[0]
                try:
                    info = sf.info(found)
                    dur = float(info.frames) / float(info.samplerate)
                except Exception:
                    # array가 있으면 fallback
                    arr = audio.get("array", None)
                    sr  = audio.get("sampling_rate", 16000)
                    dur = float(len(arr)) / float(sr) if arr is not None else 0.0
                return found, dur

        # 3) 원본 파일 바이트가 있으면 그대로 저장(확장자 추정)
        if audio.get("bytes", None) is not None:
            # 확장자 추정: path에서 따오거나 기본 .wav
            ext = ".wav"
            if isinstance(orig_path, str) and "." in os.path.basename(orig_path):
                ext = os.path.splitext(orig_path)[1] or ".wav"
            out_path = os.path.join(tmp_audio_dir, f"hf_{uuid.uuid4().hex}{ext}")
            with open(out_path, "wb") as f:
                f.write(audio["bytes"])
            # duration 계산 시도
            try:
                info = sf.info(out_path)
                dur = float(info.frames) / float(info.samplerate)
            except Exception:
                # array가 있으면 fallback
                arr = audio.get("array", None)
                sr  = audio.get("sampling_rate", 16000)
                dur = float(len(arr)) / float(sr) if arr is not None else 0.0
            return out_path, dur

        # 4) 마지막 수단: array+sr로 WAV 저장
        arr = audio.get("array", None)
        sr  = audio.get("sampling_rate", 16000)
        if arr is not None:
            out_path = os.path.join(tmp_audio_dir, f"hf_{uuid.uuid4().hex}.wav")
            sf.write(out_path, arr, sr)
            dur = float(len(arr)) / float(sr)
            return out_path, dur

        raise FileNotFoundError("오디오 경로/바이트/배열 중 어느 것도 사용할 수 없습니다.")

    n_written, n_skipped_short, n_skipped_tagonly = 0, 0, 0  # [CHANGED] 카운터 추가
    min_sec = 1.0
    with open(manifest_path, "w", encoding="utf-8") as fout:
        for sample in ds:
            audio_path, duration = _resolve_audio_path(sample)

            # 너무 짧은 샘플 스킵
            if duration < min_sec:
                n_skipped_short += 1
                continue

            # GigaSpeech는 보통 'text', 일부 스크립트는 'sentence'
            raw_text = sample.get("sentence", None)
            if raw_text is None:
                raw_text = sample.get("text", "")

            # [CHANGED] "태그가 있으면 스킵" → "태그는 제거만, 태그만 있으면 스킵"
            tag_stripped, is_tag_only = _strip_special_tags(raw_text)
            if is_tag_only:
                n_skipped_tagonly += 1
                continue

            # 이후 일반 정규화
            text = normalize_text_cv(tag_stripped, keep_punct=False)

            fout.write(json.dumps({
                "audio_filepath": audio_path,
                "duration": float(duration),
                "text": text,
            }, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[manifest] wrote {n_written} lines "
          f"(skipped {n_skipped_short} < {min_sec}s, "
          f"skipped {n_skipped_tagonly} tag-only refs) → {manifest_path}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no",  "false","f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def parse_args():
    parser = argparse.ArgumentParser(
        description="LibriSpeech 4 splits에 대해 flow-matching 모델 inference 및 평가"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True,
        help="학습된 체크포인트 경로 (예: outputs/.../checkpoints/last.ckpt)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="훈련 시 사용한 data root (manifests 폴더가 있어야 함)"
    )
    parser.add_argument(
        "--gpus", type=int, default=1,
        help="사용할 GPU 개수"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="추론 시 배치 크기"
    )
    parser.add_argument(
        "--data_script_path", type=str, default="./librispeech_asr.py",
        help="HF LibriSpeech 데이터 스크립트 경로, [./librispeech_asr.py, ./tedlium_asr.py, ./commonvoice_asr.py]"
    )
    parser.add_argument(
        "--data_config_name", type=str, default="train_100",
        help="HF 데이터 config 이름 (train_100 등)"
    )
    parser.add_argument(
        "--meta_encoder_type", type=str, default="mlp",
        choices=["mlp", "cnn", "swin"],
        help="flow-matching 메타 인코더 타입 (학습과 동일하게)"
    )
    parser.add_argument(
        "--flow_steps", type=int, default=4,
        help="flow-matching 샘플링 단계 수 (학습과 동일하게)"
    )
    parser.add_argument(
        "--flow_schedule", type=str, default="rectified",
        choices=["rectified", "vp_ode", "ve_ode"],
        help="flow-matching noise schedule (학습과 동일하게)"
    )
    parser.add_argument(
        "--flow_weight", type=float, default=1.0,
        help="flow-matching loss 가중치 (학습과 동일하게)"
    )
    parser.add_argument(
        "--use_ctc",
        type=str2bool,
        default=False,
        help="CTC loss 사용 여부 (True: CTC, False: CrossEntropy)"
    )
    parser.add_argument(
        "--use_logit_distillation",
        type=str2bool,
        default=False,
        help="CTC loss 외에 teacher logits 와의 KL-divergence loss 를 추가"
    )
    parser.add_argument(
        "--use_layerwise_distillation", 
        type=str2bool, 
        default=False,
        help="레이어 단위 KD 실행 여부"
    )
    parser.add_argument(
        "--use_diffkd", 
        type=str2bool, 
        default=False,
        help="DiffKD 기법 사용 여부"
    )
    parser.add_argument(
        "--use_flow_matching",
        type=str2bool,
        default=False,
        help="Flow Matching 기법 사용 여부"
    )
    parser.add_argument(
        "--kd_temperature", type=float, default=1.0,
        help="Knowledge Distillation 온도 (logit distillation 시)"
    )
    parser.add_argument(
        "--kd_alpha", type=float, default=0.1,
        help="Knowledge Distillation 가중치 (logit distillation 시)"
    )
    parser.add_argument(
        "--layer_kd_alpha", type=float, default=1.0,
        help="레이어 단위 Knowledge Distillation 가중치"
    )
    parser.add_argument(
        "--eval_data", type=str, default="librispeech",
        help="평가할 데이터셋 (librispeech 또는 tedlium2)"
    )
    parser.add_argument("--data_sample_rate", type=int, default=16000, help="샘플링 주파수")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Trainer 세팅
    trainer = pl.Trainer(accelerator="gpu", devices=args.gpus)

    # 2) Teacher 모델 로드
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name="stt_en_conformer_ctc_small",
        map_location="cuda:0",
        trainer=trainer,
    )
    release_nemoAPI(teacher)

    # 3) Student config 생성
    manifest_dir = os.path.join(args.data_dir, "manifests")
    cache_dir = os.path.join(args.data_dir, "cache")
    hf_config.HF_DATASETS_CACHE = cache_dir
    train_m = os.path.join(manifest_dir, "train.json")
    val_m   = os.path.join(manifest_dir, "validation.json")
    test_m  = os.path.join(manifest_dir, "test.json")

    student_cfg = make_student_config(
        teacher_model=teacher,
        args=args,
        train_manifest=train_m,
        val_manifest=val_m,
        test_manifest=test_m,
    )

    # 4) FlowMatching 설정
    flow_cfg = {
        "meta_encoder_type": args.meta_encoder_type,
        "feature_dim":      student_cfg.encoder.d_model,
        "time_embed_dim":   32,
        "hidden_dim":       128,
        "training_sampling": args.flow_steps,
        "inference_sampling": args.flow_steps,
        "weight":           args.flow_weight,
        "noise_schedule":   args.flow_schedule,
        "loss":             "mse",
        "shape_transform":  "linear",
        "student_dim":      student_cfg.encoder.d_model,
        "teacher_dim":      teacher.cfg.encoder.d_model,
        "student_head_num": student_cfg.encoder.n_heads,
        "teacher_head_num": teacher.cfg.encoder.n_heads,
    }
    diffkd_cfg = {
        "diffusion_steps": 9,
        "student_dim": student_cfg.encoder.d_model,
        "teacher_dim": teacher.cfg.encoder.d_model,
        "latent_dim": 88, # student 모델의 latent dim과 같게
        
        # 필요에 따라 추가 하이퍼파라미터
    }


    # (1) 모델 인스턴스 생성
    model = DistilFlowMatchingCTCModelBPE(
        cfg=student_cfg,
        trainer=trainer,
        teacher_model=teacher,
        use_ctc=args.use_ctc,
        use_logit_distillation=args.use_logit_distillation,
        kd_alpha=0.1,
        kd_temperature=1.0,
        use_layerwise_distillation=args.use_layerwise_distillation,
        layer_kd_alpha=1.0,
        use_flow_matching=args.use_flow_matching,
        flow_cfg=flow_cfg,
        use_diffkd=args.use_diffkd,
        diffkd_cfg=diffkd_cfg
    )
    model.eval()

    # (2) 체크포인트 로드
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False,)
    state = ckpt["state_dict"]
    # # layer_proj.* 키만 사전에서 제거
    # for k in list(state.keys()):
    #     if k.startswith("layer_proj."):
    #         state.pop(k)
    model.load_state_dict(state, strict=False)
    # model.to("cuda:0")  # 필요하다면

    # 6) 4개 split에 대해 평가
    dl_cfg = DownloadConfig(
        cache_dir=cache_dir,
        force_download=False,
        resume_download=True,
        max_retries=10,
        disable_tqdm=True,
        extract_compressed_file=True,
        delete_extracted=False,
    )
    
    if args.eval_data == "tedlium2":
        split_names = ["validation", "test"]
        script_path = "./tedlium_asr.py"
        config_name = "release2"
    elif args.eval_data == "librispeech":
        split_names = ["dev.clean", "dev.other", "test.clean", "test.other"]
        script_path = "./librispeech_asr.py"
        config_name = "train_100"
    elif args.eval_data == "commonvoice":
        split_names = ["validation", "test"]
        script_path = "./commonvoice_asr.py"
        config_name = "en"
    elif args.eval_data == "gigaspeech":
        split_names = ["validation", "test"]
        script_path = "./gigaspeech.py"
        config_name = "dev"
    else:
        raise ValueError("지원하는 eval_data: librispeech, tedlium2")
    
    all_metrics = {}
    for split in split_names:
        print(f"\n===== Evaluating split: {split} =====")
        ds = load_dataset(
            script_path,
            config_name,
            split=split,
            trust_remote_code=True,
            download_config=dl_cfg,
            cache_dir=cache_dir,
        )
        json_name = split.replace(".", "_") + ".json"
        manifest_i = os.path.join(args.data_dir, "manifests", json_name)
        if args.eval_data == "gigaspeech":
            build_manifest_from_hf_gigaspeech(ds, manifest_i, cache_dir)
        else:
            build_manifest_from_hf(ds, manifest_i, cache_dir)

        test_cfg = deepcopy(model.cfg.test_ds)
        test_cfg.manifest_filepath = manifest_i
        test_cfg.shuffle = False
        model.setup_test_data(test_cfg)
        dl = model.test_dataloader()

        results = trainer.test(
            model=model,
            dataloaders=[dl],
            ckpt_path=args.ckpt_path,
            verbose=False,
        )
        print(f'results: {results}')
        res  = results[0]
        wer  = res.get("test_wer", res.get("wer", None))
        loss = res.get("test_loss", res.get("loss", None))
        print(f"→ {split} | loss = {loss:.4f} | WER = {wer:.2%}")
        
        all_metrics[split] = {"loss": loss, "wer": wer}
    print("\n===== Final Summary =====")
    for split, m in all_metrics.items():
        print(f"{split:10s} → Loss: {m['loss']:.4f}, WER: {m['wer']:.2%}")


if __name__ == "__main__":
    main()
