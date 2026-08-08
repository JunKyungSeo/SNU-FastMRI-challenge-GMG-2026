"""최종 run 3단계 스크립트가 train.py 에 넘기는 증강 인자를 동결한다.

`python` 스텁을 PATH 앞에 두고 train_final_*.sh 를 실제로 돌려 argv 를 잡는다.
GPU·데이터·torch 불필요. argv 전체가 아니라 증강 도메인만 본다 —
LR·epoch 까지 묶으면 정상 수정을 테스트가 막는다.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = "promptmrp_GMG_final_0807"

# 증강 도메인 = 이 접두사로 시작하는 플래그. 나머지는 이 테스트의 관심 밖.
# --flip/--affine 은 변환별 확률(--flip-max-prob·--affine-max-prob) 몫이다. 접두사에서 빠뜨리면
# 그 값이 바뀌거나 사라져도 이 테스트가 초록불이라, 동결이 이름만 남는다.
# --no-aug-* 부정형도 잡는다 — --no-aug-affine 은 affine 을 통째로 끄는데 "--aug" 로는 안 걸린다.
AUG_PREFIXES = ("--aug", "--no-aug", "--mask-aug", "--remask", "--flip", "--affine")

EXPECTED = {
    "shared": {"--aug-on": True, "--aug-schedule": "constant", "--aug-max-prob": "0.5",
               "--aug-delay": "0", "--mask-aug": "full"},
    "acc4": {"--aug-on": True, "--aug-schedule": "constant", "--aug-max-prob": "0.5",
             "--flip-max-prob": "0.5", "--affine-max-prob": "0.5",
             "--affine-delay": "3", "--affine-peak-epoch": "5",
             "--aug-delay": "0", "--mask-aug": "offset", "--remask": "4"},
    "acc8": {"--aug-on": True, "--aug-schedule": "constant", "--aug-max-prob": "0.5",
             "--flip-max-prob": "0.5", "--affine-max-prob": "0.5",
             "--affine-delay": "5", "--affine-peak-epoch": "10",
             "--aug-delay": "0", "--mask-aug": "offset", "--remask": "8"},
}


def capture_argv(stage):
    """train_final_<stage>.sh 를 스텁 python 으로 돌려 argv 리스트를 돌려준다."""
    tmp = tempfile.mkdtemp()
    try:
        repo = os.path.join(tmp, "repo")
        os.makedirs(repo)
        shutil.copytree(os.path.join(ROOT, "shell_scripting"),
                        os.path.join(repo, "shell_scripting"),
                        ignore=shutil.ignore_patterns("logs", "results", "legacy"))
        # acc4/acc8 의 shared best 존재 가드 통과용 (경로가 repo 밖 ../result 다).
        shared = os.path.join(tmp, "result", RUN + "_shared", "checkpoints")
        os.makedirs(shared)
        open(os.path.join(shared, "best_model.pt"), "w").close()

        out = os.path.join(tmp, "argv.txt")
        binp = os.path.join(tmp, "bin")
        os.makedirs(binp)
        stub = os.path.join(binp, "python")
        with open(stub, "w") as f:
            f.write('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGV_OUT"\n')
        os.chmod(stub, 0o755)

        env = dict(os.environ, PATH=binp + os.pathsep + os.environ["PATH"], ARGV_OUT=out)
        r = subprocess.run(["bash", os.path.join(repo, "shell_scripting", "train_final_%s.sh" % stage)],
                           env=env, capture_output=True, text=True)
        assert os.path.exists(out), "python 이 안 불렸다 (rc=%d)\n%s\n%s" % (r.returncode, r.stdout, r.stderr)
        with open(out) as f:
            return f.read().splitlines()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def aug_domain(argv):
    """argv 에서 증강 도메인만 {플래그: 값} 으로. store_true 는 True."""
    got = {}
    for i, tok in enumerate(argv):
        if not tok.startswith(AUG_PREFIXES):
            continue
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        got[tok] = True if (nxt is None or nxt.startswith("-")) else nxt
    return got


def _check(stage):
    argv = capture_argv(stage)
    assert argv[0] == "train.py", argv[:1]
    # 중복 검사도 증강 도메인에만 건다 — argv 전체에 걸면 단계별 -l 오버라이드 같은
    # 정상 수정(shell31.sh:66이 실제로 쓰던 방식)을 이 테스트가 막는다.
    dup = [f for f in set(t for t in argv if t.startswith(AUG_PREFIXES)) if argv.count(f) > 1]
    assert not dup, "%s: 증강 플래그 중복 %s" % (stage, dup)
    assert aug_domain(argv) == EXPECTED[stage], "%s:\n  got %s\n  exp %s" % (
        stage, aug_domain(argv), EXPECTED[stage])


def test_shared():
    _check("shared")


def test_acc4():
    _check("acc4")


def test_acc8():
    _check("acc8")


if __name__ == "__main__":
    if "--dump" in sys.argv:
        for s in ("shared", "acc4", "acc8"):
            print(s, capture_argv(s))
    else:
        for s in ("shared", "acc4", "acc8"):
            _check(s)
        print("ok")
