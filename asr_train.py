#!/usr/bin/env python3
"""
train_conformer_small.py

Transformers의 load_dataset으로 LibriSpeech 100h 불러와
halved-dimension Conformer CTC 모델 구조(student)를 NeMo로 생성,
Weights & Biases 로깅 포함
"""

import os
import json
import argparse
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
import torch.nn.functional as F

def build_manifest_from_hf(ds, manifest_path: str, cache_dir: str):
    """
    HuggingFace Dataset 객체(ds)를 순회하며
    NeMo 형식의 JSON manifest를 생성
    """
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    # 기본 HF_DATASETS_CACHE (원본 오디오가 풀리던 위치)
    default_root = config.HF_DATASETS_CACHE
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
                "text":            sample["text"].lower().strip(),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")


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

class DistilEncDecCTCModelBPE(nemo_asr.models.EncDecCTCModelBPE):
    def __init__(self, cfg, trainer, teacher_model, use_logit_distillation=True, kd_alpha=0.1, kd_temperature=1.0, use_layerwise_distillation=False, layer_kd_alpha=1.0):
        super().__init__(cfg=cfg, trainer=trainer)
        self.teacher = teacher_model
        self.use_logit_distillation = use_logit_distillation
        self.kd_alpha = kd_alpha
        self.temperature = kd_temperature
        self.use_layerwise_distillation = use_layerwise_distillation
        self.layer_kd_alpha = layer_kd_alpha
        
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
            kd_loss  = F.kl_div(stu_logp, tch_p, reduction="batchmean") * (T*T)
            self.log("train_kd_loss", kd_loss, prog_bar=False, on_step=True, on_epoch=True)
        else:
            kd_loss = torch.tensor(0.0, device=log_probs.device)

        # layerwise distillation loss
        layer_loss = torch.tensor(0.0, device=log_probs.device)
        if self.use_layerwise_distillation:
            # 1) raw waveform → feature
            proc_signal, proc_length = self.preprocessor(
                input_signal=signal,
                length=signal_length,
            )
            # 2) feature → encoder output (batch, feat_dim, time) 형태
            stu_feat, _ = self.encoder(
                audio_signal=proc_signal,
                length=proc_length,
            ) # torch.Size([32, 88, 405])
            with torch.no_grad():
                # teacher도 동일하게 preprocessor → encoder
                proc_signal_t, proc_length_t = self.teacher.preprocessor(
                    input_signal=signal,
                    length=signal_length,
                )
                tch_feat, _ = self.teacher.encoder(
                    audio_signal=proc_signal_t,
                    length=proc_length_t,
                ) # torch.Size([32, 176, 405])
            
            # teacher feature 차원 맞추기 (batch, hidden_t, time_t) → (batch, hidden_s, time_s)
            if self.layer_proj is None:
                self._init_layer_proj(stu_feat, tch_feat)
            B, H_s, T_s = stu_feat.size()               # (B, H_s, T_s) torch.Size([32, 88, 405])
            stu_feat = stu_feat.transpose(1, 2)         # (B, T_s, H_s) torch.Size([32, 405, 88])
            stu_flat = stu_feat.reshape(-1, H_s)        # (B*T_s, H_s)  torch.Size([12960, 88])
            proj_flat = self.layer_proj(stu_flat)       # (B*T_s, H_t)  torch.Size([12960, 176])
            stu_proj = proj_flat.reshape(B, T_s, -1)    # (B, T_s, H_t) torch.Size([32, 405, 176])
            stu_proj = stu_proj.transpose(1, 2)         # (B, H_t, T_s) torch.Size([32, 176, 405])

            # layer_loss = F.mse_loss(stu_aligned, tch_feat)
            layer_loss = F.mse_loss(stu_proj, tch_feat)
            self.log("train_layer_kd_loss", layer_loss, prog_bar=False, on_step=True, on_epoch=True)
        else:
            layer_loss = torch.tensor(0.0, device=log_probs.device)

        # 종합 loss
        loss = ctc_loss \
             + (self.kd_alpha * kd_loss if self.use_logit_distillation else 0.0) \
             + (self.layer_kd_alpha * layer_loss if self.use_layerwise_distillation else 0.0)


        # logging
        self.log("train_loss",     loss,     prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ctc_loss", ctc_loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

class DistilFlowMatchingCTCModelBPE(nemo_asr.models.EncDecCTCModelBPE):
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
        # Lazy init for layer projection
        self.layer_proj = None
        # FlowMatchingModule
        if use_flow_matching:
            assert flow_cfg is not None
            self.flow_matching = FlowMatchingModule(flow_cfg)

    def _init_layer_proj(self, stu_feat, tch_feat):
        H_s = stu_feat.size(1)
        H_t = tch_feat.size(1)
        self.layer_proj = nn.Linear(H_s, H_t).to(stu_feat.device)

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
        if self.use_flow_matching and self.training:
            with torch.no_grad():
                proc_t, len_t = self.teacher.preprocessor(input_signal=input_signal, length=input_signal_length)
                tch_feat, _ = self.teacher.encoder(audio_signal=proc_t, length=len_t)
        # flow matching (training & inference)
        if self.use_flow_matching:
            flow_loss, encoder_out = self.flow_matching(encoder_out, tch_feat)
            # encoder_out.shape : torch.Size([32, 176, 179])
        else:
            flow_loss = torch.tensor(0.0, device=encoder_out.device)
        # decode: positional → kwargs 수정
        log_probs = self.decoder(encoder_output=encoder_out)
        greedy_preds = log_probs.argmax(dim=-1, keepdim=False)
        if self.training:
            return log_probs, encoded_len, greedy_preds, flow_loss, encoder_out
        else:
            return log_probs, encoded_len, greedy_preds
    
    def training_step(self, batch, batch_idx):
        signal, sig_len, transcript, transcript_len = batch

        # 1) Student: preprocess + encode 한 번
        proc_s, len_s = self.preprocessor(input_signal=signal, length=sig_len)
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
            stu_logp = F.log_softmax(log_probs / self.temperature, dim=-1)
            tch_p    = F.softmax(tch_logp   / self.temperature, dim=-1)
            kd_loss  = F.kl_div(stu_logp, tch_p, reduction="batchmean") \
                       * (self.temperature ** 2)
        else:
            kd_loss = torch.tensor(0.0, device=log_probs.device)

        # 7) Layerwise distillation (student feature vs. teacher feature)
        if self.use_layerwise_distillation:
            # stu_feat: 원래 encoder 출력 (before flow)
            B, H_s, T_s = stu_feat.size()
            if getattr(self, 'layer_proj', None) is None:
                self._init_layer_proj(stu_feat, tch_feat)
            stu_flat  = stu_feat.transpose(1,2).reshape(-1, H_s)
            proj_flat = self.layer_proj(stu_flat)
            stu_proj  = proj_flat.reshape(B, T_s, -1).transpose(1,2)
            layer_loss = F.mse_loss(stu_proj, tch_feat)
        else:
            layer_loss = torch.tensor(0.0, device=signal.device)

        # 8) Total loss & logging
        total_loss = (
            ctc_loss
            + (self.kd_alpha * kd_loss if self.use_logit_distillation else 0.0)
            + (self.layer_kd_alpha * layer_loss if self.use_layerwise_distillation else 0.0)
            + flow_loss
        )
        self.log("train_ctc_loss", ctc_loss, on_step=True, on_epoch=True)
        if self.use_logit_distillation:
            self.log("train_kd_loss", kd_loss, on_step=True, on_epoch=True)
        if self.use_layerwise_distillation:
            self.log("train_layer_kd_loss", layer_loss, on_step=True, on_epoch=True)
        if self.use_flow_matching:
            self.log("train_flow_matching_loss", flow_loss, on_step=True, on_epoch=True)
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

class FlowMatchingModule(nn.Module):
    def __init__(self, flow_cfg):
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

        # time embedding
        self.time_embed = nn.Linear(1, time_embed_dim)

        # meta_encoder 자동 생성
        if self.meta_encoder_type == "mlp":
            self.meta_encoder = nn.Sequential(
                nn.Linear(self.feature_dim + time_embed_dim, hidden_dim),
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
            ConformerBlock(
                d_model=self.feature_dim+time_embed_dim,
                num_attention_heads=4,
                ffn_expansion_factor=4,
                conv_expansion_factor=2,
                dropout=0.1,
            )
            
        else:
            raise ValueError(f"Unknown meta_encoder type: {self.meta_encoder_type}")

        # shape_transformation_function 선택
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

    def forward(self, s_f, t_f=None, target=None, inference_sampling: int = None):
        # s_f: student feature/logit, shape: (B, F, T) ex) torch.Size([32, 88, 422])
        # t_f: teacher feature/logit, shape: (B, F', T') or None - ex) torch.Size([32, 176, 422])
        
        sampling_steps = self.training_sampling if self.training else (inference_sampling or self.inference_sampling)
        x = s_f # ex) torch.Size([32, 88, 422])
        velocities = []
        for i in reversed(range(1, sampling_steps + 1)):
            t = torch.full((s_f.size(0), 1, s_f.size(2)), i / sampling_steps, device=s_f.device) # [32, 1, 422]
            t = t.permute(0, 2, 1) # [32, 422, 1]
            embed_t = self.time_embed(t) # Size: [32, 422, 1] -> [32, 422, 32]
            # embed_t = embed_t.permute(0, 2, 1) # [32, 32, 422]
            
            if self.meta_encoder_type == "mlp":
                # 1) (B, feature_dim, T) → (B, T, feature_dim)
                x_perm = x.permute(0, 2, 1) # torch.Size([32, 422, 88])
                # 2) (B, T, feature_dim) → (B, T, time_embed_dim)
                # t = embed_t.unsqueeze(1).expand(-1, x_perm.size(1), -1)
                # 3) concat → (B, T, feature_dim + time_embed_dim)
                embed_x = torch.cat([x_perm, embed_t], dim=-1) # torch.Size([32, 422, 120])
                # ex) embed_x.shape = torch.Size([32, 411, 120])
                # 4) MLP 적용 → (B, T, feature_dim)
                velocity = self.meta_encoder(embed_x) # torch.Size([32, 422, 88])
                # 5) 다시 (B, feature_dim, T)
                velocity = velocity.permute(0, 2, 1)
                # ex) velocity.shape = torch.Size([32, 88, 422])
            else:
                # cnn, swin 의 경우 기존 코드 유지
                # cnn 분기: 시간 임베딩을 (B, T, E) → (B, E, T) 로 permute
                embed_t_perm = embed_t.permute(0, 2, 1)       # [B, E, T]
                # student feature x: (B, F, T) 과 채널 차원으로 concat
                embed_x = torch.cat([x, embed_t_perm], dim=1) # [B, F+E, T]
                velocity = self.meta_encoder(embed_x)
            velocities.append(velocity) # velocity: (B, F, T)
            # x = x - velocity / sampling_steps
        # Compute loss only in training with shape transformation
        loss = 0.0
        # noise schedule 적용
        # student, teacher feature noise schedule
        # alpha_t, sigma_t = self.noise_schedule(t)
        # t.shape : torch.Size([32, 422, 1])
        t = t.permute(0, 2, 1)  # [32, 1, 422]
        dalpha_dt, dsigma_dt = self.noise_schedule_deriv(t)
        # \frac{\nabla_t \alpha_t Z_1 - g_{v_\theta}\left(Z_{1-i/N}, 1-i/N\right)}{-\nabla_t \sigma_t} 적용
        # velocity.shape : torch.Size([32, 88, 422])
        # s_f.shape : torch.Size([32, 88, 422])
        # noise_scheduled_t_f = alpha_t * transformed_s_f + sigma_t * t_f
        # loss = self.metric_based_loss_function(transformed_s_f, noise_scheduled_t_f)
        noise_scheduled_x = (dalpha_dt * s_f - torch.stack(velocities, dim=0).mean(0)) / (-dsigma_dt)
        if self.training and t_f is not None:
            # shape transform student->teacher dim for loss
            if self.shape_transform_type == "linear":
                transformed_s_f = self.shape_transformation_function(noise_scheduled_x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                transformed_s_f = self.shape_transformation_function(noise_scheduled_x)
            loss = self.metric_based_loss_function(transformed_s_f, t_f)
        # In inference or no teacher, no loss
        return loss, noise_scheduled_x
        # x 반환은 shape_transformation 전의 student feature
        # 즉, shape_transformation은 loss 계산에만 사용됨

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
        "--train_teacher_model",
        type=bool,
        default=False,
        help="True: teacher 모델 학습, False: student 모델 학습",
    )
    parser.add_argument(
        "--use_ctc",
        type=bool,
        default=True,
        help="CTC loss 사용 여부 (True: CTC, False: CrossEntropy)"
    )
    parser.add_argument(
        "--use_logit_distillation",
        type=bool,
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
        type=bool, 
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
        type=bool,
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
        choices=["mlp", "cnn", "swin"],
        help="Flow Matching 시 사용되는 메타 인코더 architecture"
    )
    args = parser.parse_args()

    # manifest 경로 설정
    os.makedirs(args.output_dir, exist_ok=True)
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
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
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
    print(f'train_ds.cache_files: {train_ds.cache_files}')  # [{'filename': '/home/you/.cache/huggingface/datasets/.../train.arrow', ...}, ...]
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
        build_manifest_from_hf(train_ds, test_train_manifest, cache_dir)
        build_manifest_from_hf(val_ds, test_val_manifest, cache_dir)
        build_manifest_from_hf(test_ds, test_test_manifest, cache_dir)
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
    
    # 파이썬에서 Nemo API로 풀어두는 함수 실행
    release_nemoAPI(teacher_model)
    
    # 올바른 속성 이름으로 변경
    teacher_model._save_restore_connector.model_extracted_dir = "/workspace/outputs/nemo_archive"
    AppState().nemo_file_folder = "/workspace/outputs/nemo_archive"

    if args.train_teacher_model:
        model_cfg = make_teacher_config(teacher_model, args, train_manifest, val_manifest, test_manifest)
        is_student = False
    else:
        model_cfg = make_student_config(teacher_model, args, train_manifest, val_manifest, test_manifest)
        is_student = True
    
    print(f'model_cfg: {model_cfg}')
    
    # 7) 모델 생성 (가중치는 랜덤 초기화)
    if not is_student:
        print(f'단순 teacher 모델을 불러옵니다.')
        model = nemo_asr.models.EncDecCTCModelBPE(cfg=model_cfg, trainer=trainer)
    
    else:
        if args.use_flow_matching:
            flow_cfg = {
                    "meta_encoder_type": args.meta_encoder_type,   # ["mlp", "cnn", "swin"]
                    "feature_dim": model_cfg.encoder.d_model,
                    "time_embed_dim": 32,
                    "hidden_dim": 128,
                    "training_sampling": args.flow_steps,
                    "inference_sampling": args.flow_steps,
                    "weight": args.flow_weight,
                    "noise_schedule": args.flow_schedule,  # "rectified", "vp_ode", "ve_ode"
                    "loss": "mse",  # or "cosine"
                    "shape_transform": "linear",  # or "linear", "conv1d" 등
                    "student_dim": model_cfg.encoder.d_model,  # student 모델의 feature dim
                    "teacher_dim": teacher_model.cfg.encoder.d_model,  # teacher 모델의 feature dim
                    "student_head_num": model_cfg.encoder.n_heads,  # student 모델의 head 수
                    "teacher_head_num": teacher_model.cfg.encoder.n_heads,  # teacher 모델의 head
                    # 필요하다면 cnn일 경우 in_ch, out_ch 등 추가
                }
            model = DistilFlowMatchingCTCModelBPE(
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
            )
        elif args.use_logit_distillation or args.use_layerwise_distillation:
            print(f'distillation 모델을 불러옵니다.')
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
            print(f'단순 student 모델을 불러옵니다.')
            model = nemo_asr.models.EncDecCTCModelBPE(cfg=model_cfg, trainer=trainer)


    # 8) 학습 시작
    trainer.fit(model)
        
    # 9) Best checkpoint 로드 후 .nemo로 저장
    # last_ckpt_path = os.path.join(args.output_dir, "checkpoint", "last_ckpt.ckpt")
    # trainer.save_checkpoint(last_ckpt_path)
    # print(f"✅ Final checkpoint saved to {last_ckpt_path}")
    
    # best_ckpt = checkpoint_callback.best_model_path
    # os.makedirs(f"{args.output_dir}/{exp_name}", exist_ok=True)
    # model.save_to(f"{args.output_dir}/{exp_name}/result_weight_{exp_name}.nemo")
    # print(f"Saved .nemo to {args.output_dir}/{exp_name}")
    
    # 10) 평가 시작
    split_names = ["dev.clean", "dev.other", "test.clean", "test.other"]
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
        build_manifest_from_hf(test_i_ds, manifest_i, cache_dir)

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
            ckpt_path=last_ckpt_path or None,
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
