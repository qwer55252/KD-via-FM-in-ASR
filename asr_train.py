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
from ruamel.yaml import YAML
import nemo.collections.asr as nemo_asr
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
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
    def __init__(self, cfg, trainer, teacher_model, kd_alpha, kd_temperature):
        super().__init__(cfg=cfg, trainer=trainer)
        self.teacher = teacher_model
        self.kd_alpha = kd_alpha
        self.temperature = kd_temperature

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
        # 4) teacher forward (no grad) → softmax logits
        with torch.no_grad():
            tch_log_probs, tch_encoded_len, _ = self.teacher.forward(
                input_signal=signal, input_signal_length=signal_length
            )
        T = self.temperature
        stu_logp = F.log_softmax(log_probs / T, dim=-1)
        tch_p    = F.softmax(tch_log_probs  / T, dim=-1)
        kd_loss = F.kl_div(stu_logp, tch_p, reduction="batchmean") * (T * T)

        # 5) 합성 loss
        loss = ctc_loss + self.kd_alpha * kd_loss

        # 6) logging (PyTorch Lightning 방식)
        self.log("train_loss",     loss,     prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ctc_loss", ctc_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_kd_loss",  kd_loss,  prog_bar=False, on_step=True, on_epoch=True)

        return loss



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
        "--logit_distillation",
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
    args = parser.parse_args()

    # manifest 경로 설정
    # train_manifest = os.path.join(args.data_dir, "manifests", "train-clean-100.json")
    # val_manifest = os.path.join(args.data_dir, "manifests", "validation.json")
    train_manifest = os.path.join(args.data_dir, "manifests", "train.json")
    val_manifest = os.path.join(args.data_dir, "manifests", "val.json")
    test_manifest = os.path.join(args.data_dir, "manifests", "test.json")

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

    # 3) W&B logger 생성
    exp_name = os.getenv("EXP_NAME")
    wandb_logger = WandbLogger(project=exp_name, save_dir=args.output_dir)

    # 4) PyTorch Lightning Trainer
    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu",
        max_epochs=args.epochs,
        default_root_dir=args.output_dir,
        logger=wandb_logger,
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
        student_cfg = make_teacher_config(teacher_model, args, train_manifest, val_manifest, test_manifest)
    else:
        student_cfg = make_student_config(teacher_model, args, train_manifest, val_manifest, test_manifest)
    
    print(f'student_cfg: {student_cfg}')
    
    # 7) student 모델 생성 (가중치는 랜덤 초기화)
    if args.logit_distillation:
        student_model = DistilEncDecCTCModelBPE(
            cfg=student_cfg,
            trainer=trainer,
            teacher_model=teacher_model,
            kd_alpha=args.kd_alpha,
            kd_temperature=args.kd_temperature,
        )
    else:
        student_model = nemo_asr.models.EncDecCTCModelBPE(
            cfg=student_cfg,
            trainer=trainer,
        )


    # 9) 학습 시작
    trainer.fit(student_model)
    student_model.save_to(f"outputs/{exp_name}/result_weight_{exp_name}.nemo")


    # 10) 평가 시작
    split_names = ["dev.clean", "dev.other", "test.clean", "test.other"]
    for split_name in split_names:
        print(f"\n===== Evaluating on split: {split_name} =====")
        student_model.eval()
        
        test_i_ds = load_dataset(
            args.data_script_path,
            args.data_config_name,
            split=split_name,
            trust_remote_code=True,
            download_config=dl_cfg,
            cache_dir=cache_dir,
        )
        
        json_name = split_name.replace(".", "_") + ".json"   # ex: dev_clean.json
        manifest_i = os.path.join(args.data_dir, "manifests", json_name)
        build_manifest_from_hf(test_i_ds, manifest_i, cache_dir)

        student_model.cfg.test_ds.manifest_filepath = manifest_i
        
        dl = student_model.test_dataloader()

        results = trainer.test(
            model=student_model,
            dataloaders=[dl],
            ckpt_path="best",
            verbose=True,
        )
        for res in results:
            wer = res.get("test_wer", res.get("wer", None))
            loss = res.get("test_loss", None)
            print(f"  → split={split_name} | loss={loss:.4f} | wer={wer:.2%}")

    
    
    

if __name__ == "__main__":
    main()
