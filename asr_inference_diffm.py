#!/usr/bin/env python3
# inference_splits.py

import os
import argparse
import torch
import lightning as pl
import nemo.collections.asr as nemo_asr
from datasets import load_dataset, DownloadConfig, config as hf_config
from copy import deepcopy
from asr_train_diffm import (
    release_nemoAPI,
    build_manifest_from_hf,
    make_student_config,
    DistilFlowMatchingCTCModelBPE,
)
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
        "--model_ver", type=int, default="5",
        help="모델 버전 (1~8)"
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
    val_m   = os.path.join(manifest_dir, "val.json")
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
        "hidden_dim":       96,
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
        "latent_dim": 96, # student 모델의 latent dim과 같게
        
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
        version=args.model_ver,
        diffkd_cfg=diffkd_cfg,
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
    split_names = ["dev.clean", "dev.other", "test.clean", "test.other"]
    dl_cfg = DownloadConfig(
        cache_dir=cache_dir,
        force_download=False,
        resume_download=True,
        max_retries=10,
        disable_tqdm=True,
        extract_compressed_file=True,
        delete_extracted=False,
    )
    all_metrics = {}
    for split in split_names:
        print(f"\n===== Evaluating split: {split} =====")
        ds = load_dataset(
            args.data_script_path,
            args.data_config_name,
            split=split,
            trust_remote_code=True,
            download_config=dl_cfg,
            cache_dir=cache_dir,
        )
        json_name = split.replace(".", "_") + ".json"
        manifest_i = os.path.join(manifest_dir, json_name)
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
