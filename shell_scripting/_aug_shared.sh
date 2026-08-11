#!/usr/bin/env bash
# 최종 run shared 단계의 증강 설정. train_final_shared.sh 가 _final_base.sh 다음에 source 한다.
AUG_ARGS=( --aug-on --aug-schedule constant --aug-max-prob 0.5 --aug-delay 0 --mask-aug full )
