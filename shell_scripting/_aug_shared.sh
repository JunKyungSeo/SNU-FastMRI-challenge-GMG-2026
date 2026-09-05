#!/usr/bin/env bash
# 최종 run shared 단계의 증강 설정. train_final_shared.sh 가 _final_base.sh 다음에 source 한다.
# --no-aug-affine: 본 런은 affine 이 배선되기 전 코드로 시작해 affine 이 0 이었다(체크포인트 args 에
# aug_affine 키가 없다). 지금 코드는 기본 on 이고 peak 미지정이면 constant 0.5 라, 빼면 재현 run 만
# affine 을 쓰게 된다. False 면 rng 도 안 뽑아(transforms.py __call__ 단락) flip 스트림까지 같다.
AUG_ARGS=( --aug-on --aug-schedule constant --aug-max-prob 0.5 --aug-delay 0 --mask-aug full --no-aug-affine )
