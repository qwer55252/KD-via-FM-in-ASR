#!/usr/bin/env bash
set -euo pipefail

# 순서대로 처리하고 싶은 temp 값과 alpha 값 정의
temps=(1 2 4)
alphas=(1 01 05)

for temp in "${temps[@]}"; do
  for alpha in "${alphas[@]}"; do
    script="scripts/train/logitkd_temp${temp}_alpha${alpha}.sh"
    if [[ -f "$script" ]]; then
      echo "==== 실행: $script ===="
      bash "$script"
      echo "==== 완료: $script ===="
      echo
    else
      echo "!! 스크립트가 없습니다: $script"
    fi
  done
done
