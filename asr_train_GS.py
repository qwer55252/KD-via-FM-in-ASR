#!/usr/bin/env python3
"""
train_conformer_small.py

Transformers의 load_dataset으로 LibriSpeech 100h 불러와
halved-dimension Conformer CTC 모델 구조(student)를 NeMo로 생성,
Weights & Biases 로깅 포함
"""

import os
import regex as re
import json
import uuid
import soundfile as sf
import shutil
import argparse
import unicodedata
import torch.nn as nn
from ruamel.yaml import YAML
import nemo.collections.asr as nemo_asr
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from datasets import load_dataset, DownloadConfig, config
import aiohttp
from omegaconf import OmegaConf
from copy import deepcopy
from nemo.utils.app_state import AppState
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
import glob
import torch
import random
import torch.nn.functional as F

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
    for k, v in {"\u2047": " ","“": '"', "”": '"', "„": '"',"‘": "'", "’": "'","–": "-", "—": "-","…": " ", "‹": " ", "›": " ", "«": " ", "»": " ",}.items():   # DOUBLE QUESTION MARK → 제거(공백)
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

def release_nemoAPI(teacher_model):
    # 1) .nemo 실제 경로 조회
    # 실제 .nemo 파일 경로는 teacher_model._cfg.restore_from 혹은 
    # teacher_model._pretrained_model_path 에 있습니다.
    meta = AppState().get_model_metadata_from_guid(teacher_model.model_guid)
    nemo_file = meta.restoration_path
    # 2) 압축 풀기
    connector = SaveRestoreConnector()
    connector._unpack_nemo_file(nemo_file, out_folder="/workspace/outputs/nemo_archive")
     # 3) 다음 복원 때 재활용할 디렉토리 지정
    teacher_model._save_restore_connector.model_extracted_dir = "/workspace/outputs/nemo_archive"
    AppState().nemo_file_folder = "/workspace/outputs/nemo_archive"

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

def make_teacher_config(teacher_model, args, train_manifest, val_manifest, test_manifest):
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
    
    return student_cfg

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no",  "false","f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

class DiffKDModule(nn.Module):
    """
    DiffKD: teacher feature를 선형 오토인코더로 잠재 공간으로 압축,
    student feature를 같은 잠재 공간으로 프로젝션한 뒤
    반복적 디노이저로 노이즈 제거 → 최종 디노이즈된 student latent와
    teacher latent 간 MSE 손실을 계산.
    """
    def __init__(self, cfg):
        super().__init__()
        # 파라미터
        self.steps       = cfg.get("steps", 5)
        self.teacher_dim = cfg["teacher_dim"]
        self.student_dim = cfg["student_dim"]
        # 잠재 차원 (원하면 cfg로 따로 지정 가능)
        self.latent_dim  = cfg.get("latent_dim", min(self.teacher_dim, self.student_dim))
        
        # 1) Linear Autoencoder (채널 차원 압축 & 재구성)
        #   - encoder: (B, teacher_dim, T) → (B, latent_dim, T)
        #   - decoder: (B, latent_dim, T) → (B, teacher_dim, T)
        self.encoder = nn.Conv1d(self.teacher_dim, self.latent_dim, kernel_size=1)
        self.decoder = nn.Conv1d(self.latent_dim, self.teacher_dim, kernel_size=1)
        
        # 2) Student → latent 프로젝션
        #   (B, student_dim, T) → (B, latent_dim, T)
        self.proj    = nn.Conv1d(self.student_dim, self.latent_dim, kernel_size=1)
        
        # 3) 디노이저 네트워크 (간단한 1D CNN 블록)
        self.denoiser = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1),
        )
        
        # 4) 손실 함수
        self.recon_loss   = nn.MSELoss()
        self.distill_loss = nn.MSELoss()

    def forward(self, stu_feat, tch_feat, sampling_steps=None):
        """
        Args:
          stu_feat: Tensor [B, T, student_dim]
          tch_feat: Tensor [B, T, teacher_dim]

        Returns:
          diffkd_loss: scalar tensor = AE loss + distill loss
        """
        # --- (1) Teacher feature 압축 & 재구성으로 오토인코더 학습 ---
        # teacher latent (gradient 차단)
        
        stu_feat = stu_feat.permute(0, 2, 1)  # [B, T, student_dim] → [B, student_dim, T]
        tch_feat = tch_feat.permute(0, 2, 1)  # [B, T, teacher_dim] → [B, teacher_dim, T]
        z_t = self.encoder(tch_feat).detach()            # [B, latent_dim, T]
        rec = self.decoder(z_t)                          # [B, teacher_dim, T]
        ae_loss = self.recon_loss(rec, tch_feat)         # Eq.(6)
        
        # --- (2) Student feature를 latent로 프로젝션 ---
        z_s = self.proj(stu_feat)                        # [B, latent_dim, T]
        
        # --- (3) 반복적 디노이징 (간단한 Euler 업데이트 형태) ---
        x = z_s
        for _ in range(self.steps):
            pred_noise = self.denoiser(x)                # 노이즈 예측
            x = x - pred_noise / self.steps              # 한 스텝 디노이징
        denoised = x                                     # ˆZ(stu)
        
        # --- (4) 최종 KD 손실 계산 ---
        diffkd_loss = self.distill_loss(denoised, z_t)   # Eq.(7)
        
        return ae_loss + diffkd_loss

class DistilEncDecCTCModelBPE(nemo_asr.models.EncDecCTCModelBPE):
    def __init__(self, cfg, trainer, teacher_model, use_logit_distillation=False, kd_alpha=0.1, kd_temperature=1.0, use_layerwise_distillation=False, layer_kd_alpha=1.0):
        super().__init__(cfg=cfg, trainer=trainer)
        self.teacher = teacher_model
        self.use_logit_distillation = use_logit_distillation
        self.kd_alpha = kd_alpha
        self.temperature = kd_temperature
        self.use_layerwise_distillation = use_layerwise_distillation
        self.layer_kd_alpha = layer_kd_alpha
        self.stu_feats = []
        self.tch_feats = []
        
        # projection 레이어를 lazy 초기화하기 위한 placeholder
        self.layer_proj = None
        
    def _init_layer_proj(self, stu_feat: torch.Tensor, tch_feat: torch.Tensor):
        """
        stu_feat: (B, H_s, T_s), tch_feat: (B, H_t, T_t)
        """
        H_s = stu_feat.size(1)
        H_t = tch_feat.size(1)
        # student→teacher projection
        self.layer_proj = nn.Linear(H_s, H_t).to(stu_feat.device)

    def _capture_stu_feat(self, module, input, output): # output: Tensor [B, H, T]
        self.stu_feats.append(output)
    def _capture_tch_feat(self, module, input, output):
        self.tch_feats.append(output)

    def forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        # ————— layerwise KD용 버퍼 초기화 —————
        if self.use_layerwise_distillation or self.use_flow_matching or self.use_diffkd:
            self.stu_feats.clear()
            self.tch_feats.clear()
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoder_output = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoder_output[0]
        encoded_len = encoder_output[1]
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return (
            log_probs,
            encoded_len,
            greedy_predictions,
        )
        
    def training_step(self, batch, batch_idx):
        # layerwise KD용 teacher buffer 초기화
        if self.use_layerwise_distillation:
            self.stu_feats.clear()
            self.tch_feats.clear()
        # 1) base class와 동일하게 입력 unpack
        #    (signal, signal_length, transcript, transcript_length)
        signal, signal_length, transcript, transcript_length = batch

        # 2) student forward
        log_probs, encoded_len, _ = self.forward(
            input_signal=signal, input_signal_length=signal_length
        )
        # 3) CTC loss 계산
        ctc_loss = self.loss(
            log_probs=log_probs,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=transcript_length,
        )
        
        # logit distillation loss
        if self.use_logit_distillation:
            with torch.no_grad():
                tch_log_probs, tch_encoded_len, tch_logits = self.teacher.forward(
                    input_signal=signal,
                    input_signal_length=signal_length,
                )
            T = self.temperature
            stu_logp = F.log_softmax(log_probs / T, dim=-1)
            tch_p    = F.softmax(tch_log_probs  / T, dim=-1)
            logit_kd_loss  = F.kl_div(stu_logp, tch_p, reduction="batchmean") * (T*T)
            self.log("train_kd_loss", logit_kd_loss, prog_bar=False, on_step=True, on_epoch=True)
        else:
            logit_kd_loss = torch.tensor(0.0, device=log_probs.device)

        # layerwise distillation loss
        layer_kd_loss = torch.tensor(0.0, device=log_probs.device)
        if self.use_layerwise_distillation:
            for i, (s, t) in enumerate(zip(self.stu_feats, self.tch_feats)):
                B, Hs, T = s.size()
                Ht = t.size(1)
                # (B, Hs, T) → (B*T, Hs)
                s_flat = s.transpose(1,2).reshape(-1, Hs)
                # project → (B*T, Ht) → (B, Ht, T)
                p_flat = self.layer_proj(s_flat)
                s_proj = p_flat.reshape(B, T, Ht).transpose(1,2)
                layer_loss = F.mse_loss(s_proj, t)
                layer_kd_loss += layer_loss
            layer_kd_loss /= len(self.stu_feats)  # 평균화
            self.tch_feats.clear()
        else:
            layer_loss = torch.tensor(0.0, device=log_probs.device)

        # 종합 loss
        loss = ctc_loss \
             + (self.kd_alpha * logit_kd_loss if self.use_logit_distillation else 0.0) \
             + (self.layer_kd_alpha * layer_kd_loss if self.use_layerwise_distillation else 0.0)


        # logging
        self.log("train_loss",     loss,     prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ctc_loss", ctc_loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, sig_len, transcript, transcript_len, sample_id = batch
        log_probs, encoded_len, predictions, _ = self.forward(input_signal=signal, input_signal_length=sig_len)
        transcribed = self.wer.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=log_probs, decoder_lengths=encoded_len, return_hypotheses=False
        )
        if isinstance(sample_id, torch.Tensor):
            sample_id = sample_id.cpu().numpy()
        return list(zip(sample_id, transcribed))

class DistilGenerativeCTCModelBPE(nemo_asr.models.EncDecCTCModelBPE):
    def __init__(
        self,
        cfg,
        trainer,
        teacher_model,
        use_ctc=True,
        use_logit_distillation=True,
        kd_alpha=0.1,
        kd_temperature=1.0,
        use_layerwise_distillation=False,
        layer_kd_alpha=1.0,
        use_flow_matching=False,
        flow_cfg=None,
        use_diffkd=False,
        diffkd_cfg=None,
    ):
        super().__init__(cfg=cfg, trainer=trainer)
        self.teacher = teacher_model
        self.use_ctc = use_ctc
        self.use_logit_distillation = use_logit_distillation
        self.kd_alpha = kd_alpha
        self.temperature = kd_temperature
        self.use_layerwise_distillation = use_layerwise_distillation
        self.layer_kd_alpha = layer_kd_alpha
        self.use_flow_matching = use_flow_matching
        self.layer_sampling_steps = None
        self.use_diffkd = use_diffkd
        self.router_max_sampling_steps = flow_cfg.get("router_max_sampling_steps", 16) if use_flow_matching else None
        self.flow_cfg = flow_cfg
        # Lazy init for layer projection
        
        self.flow_matching = None
        if use_flow_matching:
            assert flow_cfg is not None
            self.sampling_steps_per_layer = flow_cfg.get("sampling_steps_per_layer", None)
            assert self.sampling_steps_per_layer is None or len(self.sampling_steps_per_layer) == len(self.encoder.layers), \
                "sampling_steps_per_layer 길이는 encoder.layers 수와 같아야 합니다."
            self.flow_matching = FlowMatchingModule(flow_cfg)
            # ----- Dynamic router 설정 -----
            self.use_dynamic_steps = flow_cfg.get("use_dynamic_steps", True)
            self.router_strategy = flow_cfg.get("router_strategy", "batch_mode")
            self.router_weight = flow_cfg.get("router_weight", 1.0)
            self.router = DynamicStepRouter(
                max_steps=flow_cfg.get("router_max_sampling_steps", 16), min_steps=1,
                stu_dim=flow_cfg["student_dim"], tch_dim=flow_cfg["teacher_dim"],
                use_layer_id=True, num_layers=len(self.encoder.layers), layer_emb_dim=32,
                temperature=flow_cfg.get("router_temperature", 1.0),                 # 초기에 크게(예: 2.0~4.0) → 점점 낮추기
                budget_target=8.0, budget_weight=0.05,
                entropy_weight=0.001,
            )
        if use_diffkd:
            assert diffkd_cfg is not None, "diffkd_cfg가 None입니다. DiffKD를 사용하려면 diffkd_cfg를 제공해야 합니다."
            self.sampling_steps_per_layer = diffkd_cfg.get("sampling_steps_per_layer", None)
            assert self.sampling_steps_per_layer is None or len(self.sampling_steps_per_layer) == len(self.encoder.layers), \
                "sampling_steps_per_layer 길이는 encoder.layers 수와 같아야 합니다."
            self.diffkd = DiffKDModule(diffkd_cfg)
        self.layer_proj = None
        self.stu_feats = []
        self.tch_feats = []
        if self.use_layerwise_distillation or self.use_flow_matching or self.use_diffkd:
            self.layer_proj = nn.Linear(flow_cfg["student_dim"], flow_cfg["teacher_dim"])
            assert len(self.teacher.encoder.layers) == len(self.encoder.layers), \
                "student 모델과 teacher 모델의 layer 수가 같아야 합니다."
            for layer in self.encoder.layers:
                layer.register_forward_hook(self._capture_stu_feat)
            for layer in self.teacher.encoder.layers:
                layer.register_forward_hook(self._capture_tch_feat)
            
    
    def _capture_stu_feat(self, module, input, output): # output: Tensor [B, H, T]
        self.stu_feats.append(output)
    def _capture_tch_feat(self, module, input, output):
        self.tch_feats.append(output)

    def forward(self, input_signal=None, input_signal_length=None, 
                processed_signal=None, processed_signal_length=None):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        # ————— layerwise KD용 버퍼 초기화 —————
        if self.use_layerwise_distillation or self.use_flow_matching or self.use_diffkd:
            self.stu_feats.clear()
            self.tch_feats.clear()
        
        # preprocess
        has_input = input_signal is not None and input_signal_length is not None
        has_processed = processed_signal is not None and processed_signal_length is not None
        if (has_input ^ has_processed) is False:
            raise ValueError("Arguments `input_signal`/`input_signal_length` and `processed_signal`/`processed_signal_length` are mutually exclusive")
        if not has_processed:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        # spec augmentation
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
        # encode
        encoder_out, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        # encoder_out.shape : torch.Size([32, 88, 179])
        # prepare teacher feature for training
        tch_feat = None
        # if self.use_flow_matching and self.training:
        if self.use_flow_matching or self.use_diffkd:
            with torch.no_grad():
                proc_t, len_t = self.teacher.preprocessor(input_signal=input_signal, length=input_signal_length)
                tch_feat, _ = self.teacher.encoder(audio_signal=proc_t, length=len_t)
        # flow matching (training & inference)
        total_loss = torch.tensor(0.0, device=encoder_out.device)
        if self.use_flow_matching:
            total_flow_loss = torch.tensor(0.0, device=encoder_out.device)
            total_router_loss = torch.tensor(0.0, device=encoder_out.device)
            batch_mean_sampling_steps = []
            for i, (stu_feat, tch_feat) in enumerate(zip(self.stu_feats, self.tch_feats)):
                # ------ Layer별 sampling step 결정 ------
                # 1) 라우터로 동적 결정 (훈련/추론 공통)
                if self.use_dynamic_steps:
                    steps_batch, router_loss, aux = self.router(stu_feat, tch_feat, layer_id=i)
                    # steps: (B,) 예: tensor([3,5,4,7, ...])
                    # rloss: 스칼라 라우터 정규화 손실
                    # aux["probs"]: (B,K), aux["expected_steps"]: (B,)
                    batch_mean_sampling_steps.append(steps_batch.float().mean())
                    total_router_loss += router_loss
                    # 1-1) 배치 단위로 최빈값 sampling step을 선택
                    if self.router_strategy == "batch_mode":
                        layer_sampling_step = int(torch.mode(steps_batch).values.item())
                        flow_loss, fm_encoder_out= self.flow_matching(stu_feat, tch_feat, layer_sampling_step=layer_sampling_step, layer_id=i)
                        total_flow_loss += flow_loss
                    # 1-2) 배치 단위로 평균 sampling step을 선택
                    elif self.router_strategy == "batch_avg":
                        avg_val = torch.round(steps_batch.float().mean()).clamp(1, self.router_max_sampling_steps)
                        layer_sampling_step = int(avg_val.item())
                        flow_loss, fm_encoder_out= self.flow_matching(stu_feat, tch_feat, layer_sampling_step=layer_sampling_step, layer_id=i)
                        total_flow_loss += flow_loss
                    # 1-3) 배치 단위로 중앙값 sampling step을 선택
                    elif self.router_strategy == "batch_median":
                        med_val = torch.median(steps_batch.float())
                        layer_sampling_step = int(med_val.clamp(1, self.router_max_sampling_steps).item())
                        flow_loss, fm_encoder_out= self.flow_matching(stu_feat, tch_feat, layer_sampling_step=layer_sampling_step, layer_id=i)
                        total_flow_loss += flow_loss
                    # 1-4) 배치 내에서 sampling step이 같은 group을 묶어서 처리
                    elif self.router_strategy == "group":                        
                        unique_steps = torch.unique(steps_batch)
                        fm_encoder_out = torch.zeros_like(stu_feat)  # (B, T, C) 임시 버퍼인데, TODO: 이거 맞아? fm_encoder_out 이 원래 stu_feat이랑 shape이 같나?
                        for s in unique_steps.tolist():
                            idx = (steps_batch == s)
                            if idx.any():
                                flow_loss_s, fm_encoder_out_s = self.flow_matching(stu_feat[idx], tch_feat[idx], layer_sampling_step = int(s), layer_id = i)
                                fm_encoder_out[idx] = fm_encoder_out_s
                                total_flow_loss += flow_loss_s
                    else:
                        raise ValueError(f"Unknown router strategy: {self.router_strategy}")
                else:
                    # 2) 고정 리스트가 주어졌으면 우선 사용
                    if self.sampling_steps_per_layer is not None:
                        layer_sampling_step = int(self.sampling_steps_per_layer[i])
                    else:
                        # 3) 고정값 사용 (훈련/추론 공통)
                        layer_sampling_step = self.flow_cfg.get("training_sampling", 8)
                    flow_loss, fm_encoder_out= self.flow_matching(stu_feat, tch_feat, layer_sampling_step=layer_sampling_step, layer_id=i)
                    total_flow_loss += flow_loss
                
                
                # print(f'i: {i}, flow_loss: {flow_loss}')
            total_loss += total_router_loss * self.router_weight
            total_loss += total_flow_loss
            
            # print(f"Total flow matching loss: {total_flow_loss.item()}")
            # 로그
            self.log("train_flow_matching_loss", total_flow_loss, prog_bar=False, on_step=True, on_epoch=True)
            if self.use_dynamic_steps:
                self.log("train_router_loss", self.router_weight * total_router_loss, prog_bar=False, on_step=True, on_epoch=True)
                if len(batch_mean_sampling_steps) > 0:
                    mean_exp = torch.stack([v if v.ndim == 0 else v.mean() for v in batch_mean_sampling_steps]).mean()
                    self.log("router/batch_mean_sampling_steps_mean", mean_exp, on_step=True, on_epoch=True, prog_bar=False)

            
            
            # flow matching 결과 fm_encoder_out: (B, T, C) 
            # → decoder가 기대하는 (B, C, T) 순으로 non-inplace 변환
            encoder_out = fm_encoder_out.transpose(1, 2) # TODO: 이거 있는게 나은지 없는게 나은지 실험해보자 -> kobie-175에서    (일단은 있이 ㄱㄱ)
        # decode: positional → kwargs 수정
        
        # encoder_out.shape : torch.Size([batch, 88, seq_len]) 이어야 함
        log_probs = self.decoder(encoder_output=encoder_out)
        greedy_preds = log_probs.argmax(dim=-1, keepdim=False)
        if self.training:
            return log_probs, encoded_len, greedy_preds, total_loss, encoder_out
        else:
            return log_probs, encoded_len, greedy_preds
    
    def training_step(self, batch, batch_idx):
        # layerwise KD용 teacher buffer 초기화
        if self.use_layerwise_distillation:
            self.stu_feats.clear()
            self.tch_feats.clear()
        
        signal, sig_len, transcript, transcript_len = batch

        '''
        # 1) Student: preprocess + encode 한 번
        proc_s, len_s = self.preprocessor(input_signal=signal, length=sig_len)ㅋ
        stu_feat, encoded_len = self.encoder(audio_signal=proc_s, length=len_s)

        # 2) Teacher: preprocess+encode 한 번, decoder 한 번 (no grad)
        with torch.no_grad():
            proc_t, len_t = self.teacher.preprocessor(
                input_signal=signal, length=sig_len
            )
            tch_feat, _ = self.teacher.encoder(
                audio_signal=proc_t, length=len_t
            )
            tch_logp = self.teacher.decoder(encoder_output=tch_feat)

        # 3) Flow matching (uses stu_feat ⭢ new encoder_out) 
        if self.use_flow_matching:
            flow_loss, encoder_out = self.flow_matching(stu_feat, tch_feat)
        else:
            flow_loss = torch.tensor(0.0, device=stu_feat.device)
            encoder_out = stu_feat

        # 4) Decode student once (with matched features)
        log_probs = self.decoder(encoder_output=encoder_out)
        greedy_preds = log_probs.argmax(dim=-1)
        '''
        log_probs, encoded_len, greedy_preds, total_flow_loss, encoder_out = self.forward(input_signal=signal, input_signal_length=sig_len)

        # 5) CTC loss
        if self.use_ctc:
            ctc_loss = self.loss(
                log_probs=log_probs,
                targets=transcript,
                input_lengths=encoded_len,
                target_lengths=transcript_len,
            )
        else:
            ctc_loss = torch.tensor(0.0, device=log_probs.device)

        # 6) Logit distillation (KL-divergence)
        if self.use_logit_distillation:
            with torch.no_grad():
                tch_logp = self.teacher.decoder(encoder_output=self.tch_feats[-1].permute(0, 2, 1))
                tch_p = F.softmax(tch_logp   / self.temperature, dim=-1)
            stu_logp = F.log_softmax(log_probs / self.temperature, dim=-1)
            logit_kd_loss  = F.kl_div(stu_logp, tch_p, reduction="batchmean") \
                       * (self.temperature ** 2)
        else:
            logit_kd_loss = torch.tensor(0.0, device=log_probs.device)

        # 7) Layerwise distillation (student feature vs. teacher feature)
        layer_kd_loss = torch.tensor(0.0, device=log_probs.device)
        if self.use_layerwise_distillation:
            for i, (s, t) in enumerate(zip(self.stu_feats, self.tch_feats)):
                B, Hs, T = s.size()
                Ht = t.size(1)
                # (B, Hs, T) → (B*T, Hs)
                s_flat = s.transpose(1,2).reshape(-1, Hs)
                # project → (B*T, Ht) → (B, Ht, T)
                p_flat = self.layer_proj(s_flat)
                s_proj = p_flat.reshape(B, T, Ht).transpose(1,2)
                layer_loss = F.mse_loss(s_proj, t)
                layer_kd_loss += layer_loss
            layer_kd_loss /= len(self.stu_feats)  # 평균화
            self.tch_feats.clear()

        # 8) DiffKD (if enabled)
        diffkd_loss = torch.tensor(0.0, device=log_probs.device)
        diffkd_sampling_steps = 5
        if self.use_diffkd:
            # 모든 layer feature 사용
            for i, (stu_feat, tch_feat) in enumerate(zip(self.stu_feats, self.tch_feats)):
                diffkd_loss += self.diffkd(stu_feat, tch_feat)
                self.log(f"train_diffkd_loss_{i}", diffkd_loss, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train_diffkd_loss", diffkd_loss, prog_bar=False, on_step=True, on_epoch=True)
        
        # 8) Total loss & logging
        total_loss = (
            ctc_loss
            + (self.kd_alpha * logit_kd_loss if self.use_logit_distillation else 0.0)
            + (self.layer_kd_alpha * layer_kd_loss if self.use_layerwise_distillation else 0.0)
            + (total_flow_loss * 1.0 if self.use_flow_matching else 0.0)
            + diffkd_loss
        )
        self.log("train_ctc_loss", ctc_loss, on_step=True, on_epoch=True)
        if self.use_logit_distillation:
            self.log("train_logit_kd_loss", logit_kd_loss, on_step=True, on_epoch=True)
        if self.use_layerwise_distillation:
            self.log("train_layer_kd_loss", layer_kd_loss, on_step=True, on_epoch=True)
        if self.use_flow_matching:
            self.log("train_flow_matching_loss", total_flow_loss, on_step=True, on_epoch=True)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True)

        return total_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, sig_len, transcript, transcript_len, sample_id = batch
        log_probs, encoded_len, predictions, _ = self.forward(input_signal=signal, input_signal_length=sig_len)
        transcribed = self.wer.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=log_probs, decoder_lengths=encoded_len, return_hypotheses=False
        )
        if isinstance(sample_id, torch.Tensor):
            sample_id = sample_id.cpu().numpy()
        return list(zip(sample_id, transcribed))
    
def rectified_flow_schedule(t):
    alpha_t = t
    sigma_t = 1 - t
    return alpha_t, sigma_t
def vp_ode_schedule(t, a=19.9, b=0.1):
    alpha_t = torch.exp(-0.25 * a * (1 - t) ** 2 - 0.5 * b * (1 - t))
    sigma_t = torch.sqrt(1 - alpha_t ** 2)
    return alpha_t, sigma_t
def ve_ode_schedule(t, a=0.02, b=100):
    alpha_t = a * (b / a) ** t
    sigma_t = torch.ones_like(t)
    return alpha_t, sigma_t
def rectified_flow_schedule_deriv(t):
    # alpha_t = t, sigma_t = 1 - t
    dalpha_dt = torch.ones_like(t)
    dsigma_dt = -torch.ones_like(t)
    return dalpha_dt, dsigma_dt
def vp_ode_schedule_deriv(t, a=19.9, b=0.1):
    # alpha_t = exp(-0.25 * a * (1-t)^2 - 0.5 * b * (1-t))
    # d(alpha_t)/dt = alpha_t * [0.5*a*(1-t) + 0.5*b]
    alpha_t = torch.exp(-0.25 * a * (1 - t) ** 2 - 0.5 * b * (1 - t))
    dalpha_dt = alpha_t * (0.5 * a * (1 - t) + 0.5 * b)
    # sigma_t = sqrt(1 - alpha_t^2)
    sigma_t = torch.sqrt(1 - alpha_t ** 2)
    dsigma_dt = -alpha_t * dalpha_dt / sigma_t
    return dalpha_dt, dsigma_dt
def ve_ode_schedule_deriv(t, a=0.02, b=100):
    # alpha_t = a * (b/a)^t
    # d(alpha_t)/dt = alpha_t * log(b/a)
    alpha_t = a * (b / a) ** t
    dalpha_dt = alpha_t * torch.log(torch.tensor(b / a, device=t.device, dtype=t.dtype))
    # sigma_t = 1, dsigma_dt = 0
    dsigma_dt = torch.zeros_like(t)
    return dalpha_dt, dsigma_dt

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        if num_layers == 2:
            layers = [
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            ]
        elif num_layers == 1:
            layers = [
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            ]
        self.encoder = nn.Sequential(*layers)
    def forward(self, x):
        return self.encoder(x)
class SwinTransformerEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.attn   = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        # x: (B, C, T)
        # 1) (B, C, T) → (T, B, C), 시퀀스 길이 기준으로 MHA 호출
        x_seq = x.permute(2, 0, 1)               # [T, B, C]
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        # 2) 다시 (B, C, T) 로 되돌리기
        attn_out = attn_out.permute(1, 2, 0)     # [B, C, T]
        # 3) 채널 차원에 대해 pointwise FFN
        #    (B, C, T) → (B, T, C)
        h = attn_out.permute(0, 2, 1)            # [B, T, C]
        h = self.linear1(h)                     # [B, T, out_dim]
        h = self.relu(h)
        h = self.linear2(h)                     # [B, T, out_dim]
        # 4) (B, out_dim, T) 로 돌려서 Conv1d 분기와 동일한 포맷으로
        return h.permute(0, 2, 1)               # [B, out_dim, T]
class CNNEncoder(nn.Module):
    # for image classification
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 1)
        )
    def forward(self, x):
        return self.block(x)
class UNet1D(nn.Module):
    def __init__(self, in_ch, base_ch, out_ch, num_layers=4):
        super().__init__()
        self.down_channels = []
        ch = in_ch
        self.downs = nn.ModuleList()
        for i in range(num_layers):
            outc = base_ch * (2**i)
            self.downs.append(nn.Conv1d(ch, outc, 4, 2, 1))
            self.down_channels.append(outc)
            ch = outc
        self.bottleneck = nn.Conv1d(ch, ch, 3, 1, 1)
        # up path
        self.ups = nn.ModuleList()
        # reversed skip channels
        for skip_c in reversed(self.down_channels):
            in_c = ch + skip_c
            out_c = skip_c
            self.ups.append(nn.ConvTranspose1d(in_c, out_c, 4, 2, 1))
            ch = out_c
        self.final = nn.Conv1d(ch, out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up in self.ups:
            skip = skips.pop()
            # 만약 길이가 안 맞으면 크롭/패딩
            if x.size(2) != skip.size(2):
                diff = skip.size(2) - x.size(2)
                x = F.pad(x, (0, diff))  # 또는 x = x[..., :skip.size(2)]
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        return self.final(x)
# ---------------- Conformer Modules ----------------
class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
class ConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * expansion_factor, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            dim * expansion_factor,
            dim * expansion_factor,
            kernel_size=kernel_size,
            groups=dim * expansion_factor,
            padding=kernel_size // 2
        )
        self.batch_norm = nn.BatchNorm1d(dim * expansion_factor)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(dim * expansion_factor, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) for conv
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return self.dropout(x)
class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_mult=4, conv_expansion=2, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, mult=ff_mult, dropout=dropout)
        self.norm_ff1 = nn.LayerNorm(dim)
        self.mha_layer = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.conv_module = ConvModule(dim, expansion_factor=conv_expansion, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, mult=ff_mult, dropout=dropout)
        self.norm_ff2 = nn.LayerNorm(dim)
        self.norm_final = nn.LayerNorm(dim)

    def forward(self, x):
        # Feed Forward module 1 (half-step)
        residual = x
        x = self.norm_ff1(x)
        x = self.ff1(x)
        x = residual + 0.5 * x

        # Multi-Head Self-Attention
        residual = x
        x = self.mha_layer(x)
        x,_ = self.mha(x, x, x)
        x = residual + x

        # Convolution Module
        residual = x
        x = self.conv_module(x)
        x = residual + x

        # Feed Forward module 2 (half-step)
        residual = x
        x = self.norm_ff2(x)
        x = self.ff2(x)
        x = residual + 0.5 * x

        # Final layer norm
        return self.norm_final(x)
class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, encoder_dim, num_heads, ff_mult=4, conv_expansion_factor=2, num_layers=4, dropout=0.1):
        super().__init__()
        # initial projection if needed
        self.input_proj = nn.Linear(input_dim, encoder_dim) if input_dim != encoder_dim else nn.Identity()
        self.layers = nn.ModuleList([
            ConformerBlock(
                dim=encoder_dim,
                heads=num_heads,
                ff_mult=ff_mult,
                conv_expansion=conv_expansion_factor,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (B, T, input_dim)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        # return (B, T, encoder_dim)
        return x

class DynamicStepRouter(nn.Module):
    """
    Flow-Matching KD에서 sampling step 수(1..max_steps)를 정하는 라우터.

    입력:
      - stu_feat: (B, Hs, T)  or (B, T, Hs) 도 허용
      - tch_feat: (B, Ht, T)  or (B, T, Ht) 도 허용
      - layer_id: (B,) 또는 스칼라 int (옵션, use_layer_id=True일 때)

    출력:
      - steps: (B,) 선택된 정수 스텝 (학습: 샘플, 평가: argmax)
      - router_loss: 스칼라 (budget + entropy 등)
      - aux: dict (probs, logits, expected_steps 등 디버그용)

    옵션 특징:
      - use_layer_id: 레이어 ID 임베딩을 추가 특징으로 사용
      - budget_target: 목표 평균 스텝(실수). 없으면 budget 항 비활성화
      - budget_weight: budget 정규화 가중치
      - entropy_weight: 엔트로피 정규화(탐색 유도)
      - temperature: Gumbel-Softmax 온도(학습 중 점진적 감소 권장)
      - min_steps, max_steps: 허용 구간(기본 1..max_steps)
      - feature_reduce: 'gap' | 'mean' | 'last' 등 시간축 축약 방식
    """
    def __init__(
        self,
        max_steps: int = 16,
        min_steps: int = 1,
        stu_dim: int = None,
        tch_dim: int = None,
        hidden_dim: int = 128,
        proj_dim: int = 128,
        use_layer_id: bool = False,
        num_layers: int = None,       # use_layer_id=True일 때 필요(임베딩 크기 결정)
        layer_emb_dim: int = 32,
        feature_reduce: str = "gap",  # 'gap'|'mean'|'last'
        temperature: float = 1.0,
        budget_target: float | None = None,
        budget_weight: float = 0.1,
        entropy_weight: float = 0.0,
        allow_channel_last: bool = True,
    ):
        super().__init__()
        assert 1 <= min_steps <= max_steps
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.K = max_steps  # 카테고리 수(1..K)
        self.temperature = temperature
        self.use_layer_id = use_layer_id
        self.feature_reduce = feature_reduce
        self.allow_channel_last = allow_channel_last

        # 입력 차원 명시 필요
        assert stu_dim is not None and tch_dim is not None, "stu_dim/tch_dim을 지정하세요."

        # (Hs)->proj_dim, (Ht)->proj_dim 로 투영
        self.stu_proj = nn.Sequential(
            nn.Linear(stu_dim, proj_dim),
            nn.ReLU(inplace=True)
        )
        self.tch_proj = nn.Sequential(
            nn.Linear(tch_dim, proj_dim),
            nn.ReLU(inplace=True)
        )

        if use_layer_id:
            assert num_layers is not None and num_layers > 0
            self.layer_emb = nn.Embedding(num_layers, layer_emb_dim)
            router_in = proj_dim * 2 + layer_emb_dim
        else:
            self.layer_emb = None
            router_in = proj_dim * 2

        # 작은 MLP 라우터 → K개의 logits
        self.router = nn.Sequential(
            nn.Linear(router_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.K)
        )

        # 정규화 관련 하이퍼
        self.budget_target = budget_target  # 목표 평균 스텝(예: 6.0)
        self.budget_weight = budget_weight
        self.entropy_weight = entropy_weight

        # 선택적: 불가 구간 마스킹(여기선 min_steps>1일 때 전처리용 bias)
        mask = torch.full((self.K,), 0.0)
        if self.min_steps > 1:
            mask[: self.min_steps - 1] = float("-inf")
        self.register_buffer("logit_mask", mask)

    @staticmethod
    def _to_channel_first(x):
        # 허용 모드: (B, C, T) 또는 (B, T, C). 후자는 변환
        if x.dim() != 3:
            raise ValueError("Feature must be 3D (B, C, T) or (B, T, C).")
        B, A, B_or_T = x.shape
        # 간단 휴리스틱: 시간축이 더 길 가능성이 높음
        # 하지만 allow_channel_last=True면 (B,T,C)로 간주 가능
        return x.transpose(1, 2) if A > B_or_T else x

    def _reduce_time(self, x):  # x: (B, C, T)
        if self.feature_reduce == "gap":
            return x.mean(dim=-1)  # (B, C)
        elif self.feature_reduce == "mean":
            return x.mean(dim=-1)
        elif self.feature_reduce == "last":
            return x[..., -1]      # (B, C)
        else:
            raise ValueError(f"unknown feature_reduce: {self.feature_reduce}")

    def forward(self, stu_feat, tch_feat, layer_id=None, temperature: float | None = None, train_mode: bool | None = None):
        if train_mode is None:
            train_mode = self.training
        if temperature is None:
            temperature = self.temperature

        # 입력 정형화
        if self.allow_channel_last:
            # (B,T,C)면 (B,C,T)로 바꾸기
            if stu_feat.shape[1] != stu_feat.shape[2]:  # 단순 휴리스틱
                stu_feat = stu_feat.transpose(1, 2)
            if tch_feat.shape[1] != tch_feat.shape[2]:
                tch_feat = tch_feat.transpose(1, 2)
        else:
            stu_feat = self._to_channel_first(stu_feat)
            tch_feat = self._to_channel_first(tch_feat)

        # 시간 축 축약 → (B, Hs), (B, Ht)
        stu_vec = self._reduce_time(stu_feat)  # (B, Hs)
        tch_vec = self._reduce_time(tch_feat)  # (B, Ht)

        # 공통 차원으로 투영
        stu_h = self.stu_proj(stu_vec)  # (B, P)
        tch_h = self.tch_proj(tch_vec)  # (B, P)

        # layer id 임베딩(옵션)
        if self.use_layer_id:
            if layer_id is None:
                raise ValueError("use_layer_id=True 이면 layer_id가 필요합니다.")
            if isinstance(layer_id, int):
                layer_id = torch.full((stu_h.size(0),), layer_id, dtype=torch.long, device=stu_h.device)
            lyr = self.layer_emb(layer_id)  # (B, E)
            h = torch.cat([stu_h, tch_h, lyr], dim=-1)
        else:
            h = torch.cat([stu_h, tch_h], dim=-1)

        # 라우터 로짓
        logits = self.router(h)  # (B, K)

        # min_steps 제한용 마스크 적용
        if self.min_steps > 1:
            logits = logits + self.logit_mask  # -inf가 앞쪽에 더해져 선택 불가

        # 확률
        probs = F.softmax(logits, dim=-1)  # (B, K)
        expected_steps = (probs * torch.arange(1, self.K + 1, device=probs.device)).sum(dim=-1)  # (B,)

        if train_mode:
            # Gumbel-Softmax (straight-through)
            y_soft = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
            # hard one-hot: straight-through trick
            index = y_soft.argmax(dim=-1)
            y_hard = torch.zeros_like(y_soft).scatter_(1, index.unsqueeze(1), 1.0)
            y = (y_hard - y_soft).detach() + y_soft  # (B,K) gradient는 y_soft, fwd는 y_hard

            # 샘플된 step (1..K)
            steps = index + 1

            # 정규화 손실
            losses = []

            # (1) Budget: 평균 스텝이 budget_target에 가깝도록
            if self.budget_target is not None and self.budget_weight > 0:
                batch_mean = steps.float().mean()
                budget_loss = (batch_mean - self.budget_target) ** 2
                losses.append(self.budget_weight * budget_loss)

            # (2) Entropy: 분포가 너무 예리/평평하지 않도록
            if self.entropy_weight > 0:
                # 평균 엔트로피(큰 값일수록 탐색)
                entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=-1).mean()
                # 일반적으론 엔트로피를 '높이고' 싶으면 -entropy를 최소화 항에 넣음
                # 즉 loss += -w * entropy
                losses.append(-self.entropy_weight * entropy)

            router_loss = sum(losses) if len(losses) else logits.new_zeros([])
        else:
            # 평가: argmax
            index = probs.argmax(dim=-1)
            steps = index + 1
            router_loss = logits.new_zeros([])

        aux = {
            "logits": logits,
            "probs": probs,
            "expected_steps": expected_steps,  # 연속 기대값(분포 기반 계획에 참고)
        }
        return steps, router_loss, aux

class FlowMatchingModule(nn.Module):
    def __init__(self, flow_cfg, router: nn.Module = None, router_weight: float = 0.1):
        super().__init__()
        # 파라미터 파싱
        self.meta_encoder_type = flow_cfg.get("meta_encoder_type", "mlp")
        time_embed_dim = flow_cfg.get("time_embed_dim", 32)
        hidden_dim = flow_cfg.get("hidden_dim", 128)
        self.training_sampling = flow_cfg.get("training_sampling", 8)
        self.inference_sampling = flow_cfg.get("inference_sampling", 8)
        self.weight = flow_cfg.get("weight", 1.0)
        self.feature_dim = flow_cfg.get("student_dim", 88)
        self.teacher_dim = flow_cfg.get("teacher_dim", 176)
        self.student_head_num = flow_cfg.get("student_head_num", 4)
        self.teacher_head_num = flow_cfg.get("teacher_head_num", 8)
        
        # router 설정
        self.router = router  # DepthRouter or None
        self.router_weight = router_weight
        
        # time embedding
        self.time_embed = nn.Linear(1, time_embed_dim)

        # meta_encoder 자동 생성
        self.meta_encoder = None
        if self.meta_encoder_type == "mlp":
            self.meta_encoder = nn.Sequential(
                nn.Linear(self.feature_dim + time_embed_dim, hidden_dim),
                # nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.feature_dim)
            )
        elif self.meta_encoder_type == "cnn":
            # 예시: 1D conv
            self.meta_encoder = nn.Sequential(
                nn.Conv1d(self.feature_dim+time_embed_dim, self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
            )
        elif self.meta_encoder_type == "swin":
            self.meta_encoder = SwinTransformerEncoder(self.feature_dim + time_embed_dim, self.feature_dim, self.student_head_num)
        elif self.meta_encoder_type == "conformer":
            # Conformer 기반 flow 네트워크
            self.meta_encoder = ConformerEncoder(
                input_dim=self.feature_dim + time_embed_dim,
                encoder_dim=self.feature_dim,
                num_heads=self.student_head_num,
                ff_mult=4,
                conv_expansion_factor=2,
                num_layers=4
            )
        elif self.meta_encoder_type == "unet":
            # 1D U-Net 기반 flow 네트워크
            self.meta_encoder = UNet1D(
                in_ch=self.feature_dim + time_embed_dim,
                base_ch=hidden_dim,
                out_ch=self.feature_dim,
                num_layers=4
            )
        else:
            raise ValueError(f"Unknown meta_encoder type: {self.meta_encoder_type}")

        # shape_transformation_function 선택
        self.shape_transformation_function = None
        shape_transform = flow_cfg.get("shape_transform", "linear")
        self.shape_transform_type = shape_transform
        if shape_transform == "identity":
            self.shape_transformation_function = nn.Identity()
        elif shape_transform == "linear":
            self.shape_transformation_function = nn.Linear(self.feature_dim, self.teacher_dim)
        elif shape_transform == "conv1d":
            self.shape_transformation_function = nn.Conv1d(self.feature_dim, self.teacher_dim, 1)
        else:
            raise ValueError(f"Unknown shape_transform type: {shape_transform}")

        # loss 선택
        loss_type = flow_cfg.get("loss", "mse") # default: mse
        if loss_type == "mse":
            self.metric_based_loss_function = nn.MSELoss()
        elif loss_type == "cosine":
            self.metric_based_loss_function = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # noise schedule
        self.noise_schedule = None
        noise_schedule = flow_cfg.get("noise_schedule", "rectified")
        if noise_schedule == "rectified":
            self.noise_schedule = rectified_flow_schedule
            self.noise_schedule_deriv = rectified_flow_schedule_deriv
        elif noise_schedule == "vp_ode":
            self.noise_schedule = vp_ode_schedule
            self.noise_schedule_deriv = vp_ode_schedule_deriv
        elif noise_schedule == "ve_ode":
            self.noise_schedule = ve_ode_schedule
            self.noise_schedule_deriv = ve_ode_schedule_deriv
        else:
            raise NotImplementedError

    def forward(self, s_f, t_f=None, target=None, layer_sampling_step: int = None, layer_id: int = None):    
        x = s_f
        for i in range(layer_sampling_step, 0, -1):
            # s_f.shape: (32, 179, 88)
            t = torch.full((s_f.size(0), s_f.size(1), 1), i / layer_sampling_step, device=s_f.device)
            # t.shape: (32, 179, 1)
            embed_t = self.time_embed(t)
            # embed_t.shape: (32, 179, 32)
            embed_t = embed_t.permute(0, 2, 1)
            # embed_t.shape: (32, 32, 179)
            if self.meta_encoder_type == "mlp" or self.meta_encoder_type == "conformer":
                x_perm = x.permute(0, 2, 1)
                # x_perm.shape: (batch, 88, 179)
                # embed_t.shape: (batch, 32, 179)
                embed_x = torch.cat([x_perm, embed_t], dim=1)
                # embed_x.shape: (batch, 120, 179)
                embed_x = embed_x.permute(0, 2, 1)
                # embed_x.shape: (32, 179, 120)
                velocity = self.meta_encoder(embed_x)
                # velocity.shape: (32, 179, 88)
            else:
                x_perm = x.permute(0, 2, 1)
                embed_x = torch.cat([x_perm, embed_t], dim=1)
                # embed_x.shape: (batch, 120, 179)
                velocity = self.meta_encoder(embed_x)
                # velocity.shape: (32, 88, 179)
                velocity = velocity.permute(0, 2, 1)
                # velocity.shape: (32, 179, 88)
            
            if self.meta_encoder_type == "unet":
                # time-dim 이 x 와 다를 때, crop 또는 pad 로 맞춰주기
                T_x = x.size(2)
                T_v = velocity.size(2)
                if T_v != T_x:
                    if T_v > T_x:
                        # 너무 길면 앞쪽 T_x 만큼만
                        velocity = velocity[:, :, :T_x]
                    else:
                        # 너무 짧으면 뒤쪽에 0 패딩
                        pad_amt = T_x - T_v
                        velocity = F.pad(velocity, (0, pad_amt))
            # x: (batch, seq_len, Hs), velocity: (batch, seq_len, Hs)
            x = x - velocity / layer_sampling_step
            
        # 기존 loss
        loss = 0.0
        if self.training and t_f is not None:
            # t = t.permute(0, 2, 1)
            dalpha_dt, dsigma_dt = self.noise_schedule_deriv(t) # TODO: t를 이렇게?
            noise_scheduled_x = (dalpha_dt * s_f - velocity) / (-dsigma_dt)
            if self.shape_transform_type == "linear":
                transformed_s_f = self.shape_transformation_function(noise_scheduled_x)
            else:
                transformed_s_f = self.shape_transformation_function(noise_scheduled_x)
            kd_loss = self.metric_based_loss_function(transformed_s_f, t_f)
            loss = kd_loss

        # 라우터 loss를 합산(가중치 조절)

        return loss, x

def main():
    parser = argparse.ArgumentParser(
        description="Train halved-dimension Conformer CTC student on LibriSpeech 100h"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="데이터 루트 디렉토리")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/conformer_ctc_bpe.yaml",
        help="원본 모델 config YAML 경로",
    )
    parser.add_argument("--epochs", type=int, default=50, help="최대 epoch 수")
    parser.add_argument("--gpus", type=int, default=1, help="사용할 GPU 개수")
    parser.add_argument("--batch_size", type=int, default=8, help="배치 크기")
    parser.add_argument("--data_sample_rate", type=int, default=16000, help="샘플링 주파수")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="로그·체크포인트·wandb 저장 디렉토리",
    )
    parser.add_argument(
        "--data_script_path",
        type=str,
        default="./librispeech_asr.py",
        help="HuggingFace LibriSpeech 데이터 스크립트 경로",
    )
    parser.add_argument(
        "--data_config_name",
        type=str,
        default="train_100",
        help="_DL_URLS 키값 설정(train_100 등)",
    )
    parser.add_argument(
        "--data_train_split",
        type=str,
        default="train.clean.100",
        help="훈련 split 이름",
    )
    parser.add_argument(
        "--data_val_split",
        type=str,
        default="dev.clean",
        help="평가 split 이름",
    )
    parser.add_argument(
        "--data_test_split",
        type=str,
        default="test.clean",
        help="평가 split 이름",
    )
    parser.add_argument(
        "--train_student_model",
        type=str2bool,
        default=False,
        help="True: student 모델 학습",
    )
    parser.add_argument(
        "--train_teacher_model",
        type=str2bool,
        default=False,
        help="True: teacher 모델 학습",
    )
    parser.add_argument(
        "--use_ctc",
        type=str2bool,
        default=True,
        help="CTC loss 사용 여부 (True: CTC, False: CrossEntropy)"
    )
    parser.add_argument(
        "--use_logit_distillation",
        type=str2bool,
        default=False,
        help="CTC loss 외에 teacher logits 와의 KL-divergence loss 를 추가"
    )
    parser.add_argument(
        "--kd_alpha",
        type=float,
        default=1.0,
        help="logit distillation loss 의 가중치"
    )
    parser.add_argument(
        "--kd_temperature",
        type=float,
        default=1.0,
        help="softmax 온도 파라미터"
    )
    parser.add_argument(
        "--use_layerwise_distillation", 
        type=str2bool, 
        default=False,
        help="레이어 단위 KD 실행 여부"
    )
    parser.add_argument(
        "--layer_kd_alpha", 
        type=float, 
        default=1.0,
        help="레이어 KD loss 가중치"
    )
    parser.add_argument(
        "--use_flow_matching",
        type=str2bool,
        default=False,
        help="Flow Matching 기법 사용 여부"
    )
    parser.add_argument(
        "--flow_steps",
        type=int,
        default=8,
        help="Flow Matching 시 사용되는 시간 단계 수"
    )
    parser.add_argument(
        "--dirac_ratio",
        type=float,
        default=0.1,
        help="Flow Matching 시 Dirac delta 비율 (0.0 ~ 1.0)"
    )
    parser.add_argument(
        "--flow_weight",
        type=float,
        default=1.0,
        help="Flow Matching loss 의 가중치"
    )
    parser.add_argument(
        "--flow_schedule",
        type=str,
        default="rectified",
        choices=["rectified", "vp_ode", "ve_ode"],
        help="Flow Matching 시 사용되는 noise schedule"
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="테스트 모드일 때 True로 설정하면 데이터셋을 매우 적게 사용"
    )
    parser.add_argument(
        "--meta_encoder_type",
        type=str,
        default="mlp",
        choices=["mlp", "cnn", "swin", "conformer", "unet"],
        help="Flow Matching 시 사용되는 메타 인코더 architecture"
    )
    parser.add_argument(
        "--shape_transform_type",
        type=str,
        default="linear",
        choices=["identity", "linear", "conv1d"],
        help="Flow Matching 시 student feature → teacher feature 변환 방식"
    )
    def parse_sampling_steps_per_layer(s):
        if s == "random":
            # 1,2,4,8 중에서 16개를 랜덤하게 선택
            choices = [1, 2, 4, 8]
            return [random.choice(choices) for _ in range(16)]
        else:
            return json.loads(s)

    parser.add_argument(
        "--sampling_steps_per_layer",
        type=parse_sampling_steps_per_layer,
        default=None,
        help="각 레이어별로 Flow Matching 시 사용하는 샘플링 단계 수 (e.g. \"[1,1,2,2]\" 또는 \"random\")"
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="학습 재개할 체크포인트(.ckpt) 파일 경로"
    )    
    parser.add_argument(
        "--use_diffkd",
        type=str2bool,
        default=False,
        help="DiffKD knowledge distillation 기법 사용 여부"
    )
    parser.add_argument(
        "--diffkd_steps",
        type=int,
        default=9,
        help="DiffKD denoising 단계 수"
    )
    parser.add_argument(
        "--use_dynamic_steps",
        type=str2bool,
        default=False,
        help="DiffKD knowledge distillation 기법 사용 여부"
    )
    parser.add_argument(
        "--router_weight",
        type=float,
        default=1.0,
        help="DepthRouter loss 의 가중치 (0.0: 라우터 loss 비활성화, 1.0: 활성화)"
    )
    parser.add_argument(
        "--router_temperature",
        type=float,
        default=1.0,
        help="DepthRouter의 온도 파라미터 (1.0이 기본값)"
    )
    parser.add_argument(
        "--router_max_sampling_steps",
        type=int,
        default=8,
        help="DepthRouter가 선택할 수 있는 최대 샘플링 단계 수"
    )
    parser.add_argument(
        "--router_strategy",
        type=str,
        default="batch_mode",
        choices=["batch_mode", "batch_avg", "batch_median", "group"],
        help="라우팅 전략 기본값: 'batch_mode' | 'batch_avg' | 'batch_median' | 'group'"
        )
    args = parser.parse_args()
    # manifest 경로 설정
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(__file__, os.path.join(args.output_dir, os.path.basename(__file__)))
    manifest_dir = os.path.join(args.data_dir, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    # train_manifest = os.path.join(args.data_dir, "manifests", "train-clean-100.json")
    # val_manifest = os.path.join(args.data_dir, "manifests", "validation.json")
    train_manifest = os.path.join(manifest_dir, "train.json")
    val_manifest = os.path.join(manifest_dir, "val.json")
    test_manifest = os.path.join(manifest_dir, "test.json")

    # 1) HuggingFace LibriSpeech 로드
    print("Datasets cache dir:", config.HF_DATASETS_CACHE)
    
    cache_dir = os.path.join(args.data_dir, "cache")
    config.HF_DATASETS_CACHE = cache_dir
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
        args.data_script_path,
        args.data_config_name,
        split=args.data_train_split,
        trust_remote_code=True,
        download_config=dl_cfg,
        cache_dir=cache_dir,
    )
    val_ds = load_dataset(
        args.data_script_path,
        args.data_config_name,
        split=args.data_val_split,
        trust_remote_code=True,
        download_config=dl_cfg,
        cache_dir=cache_dir,
    )
    test_ds = load_dataset(
        args.data_script_path,
        args.data_config_name,
        split=args.data_test_split,
        trust_remote_code=True,
        download_config=dl_cfg,
        cache_dir=cache_dir,
    )
    print(f'train_dataset: {train_ds}')
    print(f'val_dataset: {val_ds}')
    print(f'test_dataset: {test_ds}')
    print(f'train_ds.cache_files: {train_ds.cache_files}')  # [{'filename': '/home/you/.cache/huggingface/datasets/.../train.arrow', ...}, ...]
    # 2) NeMo manifest 생성

    print("building manifest files...")
    if not os.path.isfile(train_manifest):
        build_manifest_from_hf_gigaspeech(train_ds, train_manifest, cache_dir)
        print(f"train_manifest DONE: {train_manifest}")
    if not os.path.isfile(val_manifest):
        build_manifest_from_hf_gigaspeech(val_ds, val_manifest, cache_dir)
        print(f"val_manifest DONE: {val_manifest}")
    if not os.path.isfile(test_manifest):
        build_manifest_from_hf_gigaspeech(test_ds, test_manifest, cache_dir)
        print(f"test_manifest DONE: {test_manifest}")
    print("manifest files built.")
    
    def _assert_nonempty_manifest(path, name):
        n = sum(1 for _ in open(path, "r", encoding="utf-8") if _.strip())
        if n == 0:
            raise RuntimeError(f"{name} manifest has 0 lines: {path}")

    _assert_nonempty_manifest(train_manifest, "train")
    _assert_nonempty_manifest(val_manifest, "val")
    _assert_nonempty_manifest(test_manifest, "test")
    
    # test_mode 데이터셋 축소
    if args.test_mode:
        print("Running in test mode, reducing dataset size...")
        train_ds = train_ds.select(range(100))
        val_ds = val_ds.select(range(100))
        test_ds = test_ds.select(range(100))
        # test_mode용 manifest 파일 생성
        test_train_manifest = os.path.join(manifest_dir, "train_testmode.json")
        test_val_manifest = os.path.join(manifest_dir, "val_testmode.json")
        test_test_manifest = os.path.join(manifest_dir, "test_testmode.json")
        build_manifest_from_hf_gigaspeech(train_ds, test_train_manifest, cache_dir)
        build_manifest_from_hf_gigaspeech(val_ds, test_val_manifest, cache_dir)
        build_manifest_from_hf_gigaspeech(test_ds, test_test_manifest, cache_dir)
        train_manifest = test_train_manifest
        val_manifest = test_val_manifest
        test_manifest = test_test_manifest
        
        # epochs 수 축소
        args.epochs = 5
    print(f"train_manifest: {train_manifest}")
    print(f"val_manifest: {val_manifest}")
    print(f"test_manifest: {test_manifest}")
    

    # 3) W&B logger 생성
    prj_name = os.getenv("PRJ_NAME")
    exp_name = os.getenv("EXP_NAME")
    wandb_logger = WandbLogger(project=prj_name, name=exp_name, save_dir=args.output_dir)
    last_ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    last_ckpt_path = os.path.join(last_ckpt_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        pattern = os.path.join(last_ckpt_dir, "last-v*.ckpt")
        existing_files = glob.glob(pattern)
        sub_ckpt_path = os.path.join(last_ckpt_dir, f"last-v{len(existing_files)+1}.ckpt")
        os.rename(last_ckpt_path, sub_ckpt_path)
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=last_ckpt_dir,
        filename="last",
        save_top_k=0,
        verbose=True,
        save_last=True,
    )

    # 4) PyTorch Lightning Trainer
    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu",
        max_epochs=args.epochs,
        default_root_dir=args.output_dir,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    # 5) Teacher 모델 로드 (pretrained) -> config만 사용할 것
    teacher_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name="stt_en_conformer_ctc_small",
        map_location="cuda:0",
        trainer=trainer,
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    
    # 파이썬에서 Nemo API로 풀어두는 함수 실행
    release_nemoAPI(teacher_model)
    
    # 올바른 속성 이름으로 변경
    teacher_model._save_restore_connector.model_extracted_dir = "/workspace/outputs/nemo_archive"
    AppState().nemo_file_folder = "/workspace/outputs/nemo_archive"

    if args.train_teacher_model:
        model_cfg = make_teacher_config(teacher_model, args, train_manifest, val_manifest, test_manifest)
        is_student = False
    elif args.train_student_model:
        model_cfg = make_student_config(teacher_model, args, train_manifest, val_manifest, test_manifest)
        is_student = True
    
    print(f'model_cfg: {model_cfg}')
    
    # 7) 모델 생성 (가중치는 랜덤 초기화)
    if args.train_teacher_model or args.train_student_model:
        print(f'단순 EncDecCTCModelBPE 모델을 불러옵니다.')
        model = nemo_asr.models.EncDecCTCModelBPE(cfg=model_cfg, trainer=trainer)
    elif args.use_logit_distillation or args.use_layerwise_distillation:
        print(f'LogitKD, LayerwiseKD 를 위한 DistilEncDecCTCModelBPE 모델을 불러옵니다.')
        model = DistilEncDecCTCModelBPE(
            cfg=model_cfg,
            trainer=trainer,
            teacher_model=teacher_model,
            use_logit_distillation=args.use_logit_distillation,
            kd_alpha=args.kd_alpha,
            kd_temperature=args.kd_temperature,
            use_layerwise_distillation=args.use_layerwise_distillation,
            layer_kd_alpha=args.layer_kd_alpha,
        )
    else:
        print(f'Generative KD 를 위한 DistilGenerativeCTCModelBPE 모델을 불러옵니다.')
        flow_cfg = {
                "meta_encoder_type": args.meta_encoder_type,   # ["mlp", "cnn", "swin", "unet", "conformer"]
                "feature_dim": model_cfg.encoder.d_model,
                "time_embed_dim": 32,
                "hidden_dim": 128,
                "training_sampling": args.flow_steps,
                "inference_sampling": args.flow_steps,
                "weight": args.flow_weight,
                "noise_schedule": args.flow_schedule,  # "rectified", "vp_ode", "ve_ode"
                "loss": "mse",  # or "cosine"
                "shape_transform": args.shape_transform_type,  # or "linear", "conv1d" 등
                "student_dim": model_cfg.encoder.d_model,  # student 모델의 feature dim
                "teacher_dim": teacher_model.cfg.encoder.d_model,  # teacher 모델의 feature dim
                "student_head_num": model_cfg.encoder.n_heads,  # student 모델의 head 수
                "teacher_head_num": teacher_model.cfg.encoder.n_heads,  # teacher 모델의 head
                "sampling_steps_per_layer": args.sampling_steps_per_layer,
                # 필요하다면 cnn일 경우 in_ch, out_ch 등 추가
                # --- Router ---
                "use_dynamic_steps": args.use_dynamic_steps,
                "router_strategy": args.router_strategy, # 라우팅 전략 기본값: 'batch_mode' | 'batch_avg' | 'batch_middle' | 'group'
                "router_weight": args.router_weight,               # total loss에 라우터 loss 기여
                "router_hidden": 128,
                "router_temperature": args.router_temperature,          # 1.5 → 1.0 → 0.7 스케줄링 권장
                # "router_target_avg_steps": 8,       # 고정 baseline과 공정 비교: 평균을 맞춤
                "router_max_sampling_steps": args.router_max_sampling_steps if hasattr(args, 'router_max_sampling_steps') else 16,
            }
        diffkd_cfg = {
            "steps": args.diffkd_steps,
            "student_dim": model_cfg.encoder.d_model,
            "teacher_dim": teacher_model.cfg.encoder.d_model,
            "latent_dim": model_cfg.encoder.d_model, # student 모델의 latent dim과 같게
            
            # 필요에 따라 추가 하이퍼파라미터
        }
        model = DistilGenerativeCTCModelBPE(
            cfg=model_cfg,
            trainer=trainer,
            teacher_model=teacher_model,
            use_ctc=args.use_ctc,
            use_logit_distillation=args.use_logit_distillation,
            kd_alpha=args.kd_alpha,
            kd_temperature=args.kd_temperature,
            use_layerwise_distillation=args.use_layerwise_distillation,
            layer_kd_alpha=args.layer_kd_alpha,
            use_flow_matching=args.use_flow_matching,
            flow_cfg=flow_cfg,
            use_diffkd=args.use_diffkd,
            diffkd_cfg=diffkd_cfg,
        )


    # 8) 학습 시작
    trainer.fit(model, ckpt_path=args.resume_ckpt)
        
    # 9) Best checkpoint 로드 후 .nemo로 저장
    # last_ckpt_path = os.path.join(args.output_dir, "checkpoint", "last_ckpt.ckpt")
    # trainer.save_checkpoint(last_ckpt_path)
    # print(f"✅ Final checkpoint saved to {last_ckpt_path}")
    
    # best_ckpt = checkpoint_callback.best_model_path
    # os.makedirs(f"{args.output_dir}/{exp_name}", exist_ok=True)
    # model.save_to(f"{args.output_dir}/{exp_name}/result_weight_{exp_name}.nemo")
    # print(f"Saved .nemo to {args.output_dir}/{exp_name}")
    
    # 10) 평가 시작
    if "tedlium" in args.data_script_path:
        split_names = ["validation", "test"]
    elif "librispeech" in args.data_script_path:
        split_names = ["dev.clean", "dev.other", "test.clean", "test.other"]
    elif "commonvoice" in args.data_script_path:
        split_names = ["validation", "test"]
    else:
        split_names = [args.data_val_split, args.data_test_split]
    metrics = {}
    for i, split_name in enumerate(split_names):
        print(f"\n===== Evaluating on split: {split_name} =====")
        model.eval()

        test_i_ds = load_dataset(
            args.data_script_path,
            args.data_config_name,
            split=split_name,
            trust_remote_code=True,
            download_config=dl_cfg,
            cache_dir=cache_dir,
        )
        json_name = split_name.replace(".", "_") + ".json"
        manifest_i = os.path.join(manifest_dir, json_name)
        build_manifest_from_hf_gigaspeech(test_i_ds, manifest_i, cache_dir)

        test_data_config = deepcopy(model.cfg.test_ds)
        test_data_config.manifest_filepath = manifest_i
        # shuffle 옵션이 없으면 False 로 자동 설정되지만, 명시적으로 꺼줄 수도 있습니다.
        test_data_config.shuffle = False

        # NeMo API 호출: 내부에서 _test_dl 이 세팅되고,
        # 이후 test_dataloader() 호출 시 이 _test_dl 이 반환됩니다.
        model.setup_test_data(test_data_config)
        dl = model.test_dataloader()
        
        results = trainer.test(
            model=model,
            dataloaders=[dl],
            ckpt_path=args.resume_ckpt if args.resume_ckpt else last_ckpt_path,
            verbose=True,
        )
        
        # trainer.test 는 리스트(dict) 반환, 첫 번째 원소에서 메트릭 추출
        res   = results[0]
        wer   = res.get("test_wer", res.get("wer", None))
        loss  = res.get("test_loss", res.get("loss", None))
        print(f"  → split={split_name} | loss={loss:.4f} | wer={wer:.2%}")
        
        # ① 메트릭 키에 split 이름을 붙여서 Wandb에 기록
        # #    dev.clean  → dev_clean/wer, dev_clean/loss
        key_prefix = split_name.replace(".", "_")
        metric = {
            f"{key_prefix}/wer":  wer,
            f"{key_prefix}/loss": loss,
        }
        metrics[f"{key_prefix}/wer"] = wer
        metrics[f"{key_prefix}/loss"] = loss
        # ② step을 epoch 기반으로 찍거나 global_step 을 사용
        wandb_logger.log_metrics(metric, step=trainer.current_epoch)
    print(f"metrics: {metrics}")
    wandb_logger.log_metrics(metrics, step=trainer.current_epoch)
    
if __name__ == "__main__":
    main()
