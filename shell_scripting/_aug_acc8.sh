#!/usr/bin/env bash
# 최종 run acc8 전문가의 증강 설정. train_final_acc8.sh 가 _final_base.sh 다음에 source 한다.
# --mask-aug offset 은 지금 배선에서 no-op 이지만 resume 일치 필드라 지우면 안 된다(final_run_todo.md).
AUG_ARGS=( --aug-on --aug-schedule constant --aug-max-prob 0.5 --flip-max-prob 0.5 --affine-max-prob 0.5 --affine-delay 5 --affine-peak-epoch 10 --aug-delay 0 --remask 8 --mask-aug offset )
