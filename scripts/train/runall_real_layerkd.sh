#!/usr/bin/env bash
set -euo pipefail

# 실험할 layer_kd_alpha 값들 (파일명 suffix)
alphas=(05 01 005 001 5 10)

for alpha in "${alphas[@]}"; do
  script="scripts/train/real_layerkd_layeralpha${alpha}.sh"
  if [[ -f "$script" ]]; then
    echo "==== 실행: $script ===="
    bash "$script"
    echo "==== 완료: $script ===="
    echo
  else
    echo "!! 스크립트가 없습니다: $script"
  fi
done
