import os
import sys

# scripts/tests/conftest.py → repo root는 세 단계 위(tests → scripts → root).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_model = os.path.join(ROOT, "utils", "model")
if _model not in sys.path:
    sys.path.append(_model)   # vendored fastmri(varnet.py:11)용. append = utils 네임스페이스 패키지 shadow 방지.
