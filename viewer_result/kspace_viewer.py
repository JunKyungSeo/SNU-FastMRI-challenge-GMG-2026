"""fastMRI k-space 뷰어 — 복소수 k-space 자체를 들여다보는 도구.

보는 대상은 재구성 이미지가 아니라 h5 에 들어 있는 복소수 배열 그 자체다:

  성분   |k| (dB) · 실수부 Re · 허수부 Im · 위상 ∠k     (코일별 또는 15코일 grid)
  확대   중앙 ACS 부근만 ×2·×4·×8 로 — 에너지가 중앙 몇 줄에 몰려 있어 전체 그림은 거의 검다
  값     화면을 클릭하면 그 지점의 (kx, ky), 복소수 값, |k| dB, 위상, 샘플 여부를 숫자로
  라인   위상-인코딩 라인별 에너지 막대 + 마스크 색 → 마스크가 어떤 라인을 버렸는지 정량으로

파일 구조 (train/val/leaderboard 공통):
  <split>/kspace/*.h5   kspace (S, C, 640, 368) complex64  — 마스크를 곱하기 **전**의 full k-space
                        mask   (368,)           float32    — phase-encoding 방향 언더샘플 패턴
  <split>/image/*.h5    image_label / image_input / image_grappa (S, 384, 384) + bbox annotations

k-space 는 full · masked · mask 세 칸이 항상 나란히 뜬다. ifft 재구성과 짝 image h5 는
기본으로 꺼져 있고 필요할 때만 켠다 (이미지 위주로 보려면 h5_viewer.py 를 쓰면 된다).

acc4 파일로 확인한 관계 (slice 13, 최대 오차 2e-11):
  RSS(ifftc(kspace)) -> crop = image_label,  RSS(ifftc(kspace*mask)) -> crop = image_input
  crop = readout 640 중앙 384 + phase 368 을 좌우 8px zero-pad 해서 384

사용법:
  python kspace_viewer.py                          # 현재 폴더의 k-space h5
  python kspace_viewer.py /root/Data --host 0.0.0.0 --port 8000   # train·val·leaderboard 전부
  python kspace_viewer.py /root/Data/val --export sheet.png --slice 13

의존성: pip install h5py numpy pillow  (torch 불필요, GPU 불필요)
"""

import argparse
import io
import json
import os
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import h5py
import numpy as np
from PIL import Image, ImageDraw

from h5_viewer import draw_boxes   # bbox 그리기는 image 뷰어 것을 그대로 쓴다

CROP_SIZE = 384       # 대회 출력 해상도
PANEL_MAX = 250       # 코일 grid 한 칸의 최대 변 길이(px)
CAPTION_H = 16        # 몽타주 캡션 높이(px)
CACHE_SLOTS = 4       # 메모리에 들고 있을 슬라이스 수 (1개 = 약 28MB)
DISPLAY_PCT = 99.5    # 밝기/색 상한(백분위). max 로 맞추면 이상치 한 점에 전체가 눌린다
PROFILE_H = 190       # 라인 에너지 그림 높이(px)
PAIR_KEYS = {"label": "image_label", "input": "image_input", "grappa": "image_grappa"}


# ======================================================================
# k-space 연산 (utils/common/utils.py 와 같은 규약: centered, norm="ortho")
# ======================================================================

def ifftc(data, axes=(-2, -1)):
    """Centered inverse FFT."""
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(data, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def rss(data, axis=0):
    """Root-sum-of-squares 코일 합성."""
    return np.sqrt(np.sum(np.abs(data) ** 2, axis=axis))


def crop_to_384(img):
    """(H, W) -> (384, 384). readout 은 중앙 crop, 폭이 모자라면 zero-pad."""
    h, w = img.shape
    if h >= CROP_SIZE:
        r0 = (h - CROP_SIZE) // 2
        img = img[r0:r0 + CROP_SIZE]
    if w >= CROP_SIZE:
        c0 = (w - CROP_SIZE) // 2
        img = img[:, c0:c0 + CROP_SIZE]
    out = np.zeros((CROP_SIZE, CROP_SIZE), np.float32)
    y0 = (CROP_SIZE - img.shape[0]) // 2
    x0 = (CROP_SIZE - img.shape[1]) // 2
    out[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return out


def zoom_box(shape, factor):
    """중앙 확대 창 (y0, x0, h, w). factor=1 이면 전체."""
    h, w = shape
    if factor <= 1:
        return 0, 0, h, w
    hh, ww = max(8, h // factor), max(8, w // factor)
    return (h - hh) // 2, (w - ww) // 2, hh, ww


def zoom_center(arr, factor):
    """마지막 두 축을 중앙 기준으로 잘라낸다."""
    y0, x0, hh, ww = zoom_box(arr.shape[-2:], factor)
    return arr[..., y0:y0 + hh, x0:x0 + ww]


# ======================================================================
# 파일 찾기 — train / val / leaderboard/acc{4,8} 를 한 번에
# ======================================================================

_CACHE = {}
_LOCK = threading.Lock()


def acc_key(name):
    """'kspace_knee_acc4_1.h5' -> 'acc4_1'. utils/data 처럼 이름이 어긋난 짝을 잇는 용도."""
    m = re.search(r"acc(\d+)_(\d+)", name)
    return f"acc{m.group(1)}_{m.group(2)}" if m else None


def find_pair(kspace_path):
    """k-space h5 에 대응하는 image h5 경로. 없으면 None.

    1) 같은 split 의 image/ 폴더에 같은 이름 (train/val/leaderboard 실제 배치)
    2) 같은 폴더 안에서 acc 토큰이 같은 img_*.h5 (utils/data 샘플 배치)
    """
    d, name = os.path.split(os.path.abspath(kspace_path))
    sibling = os.path.join(os.path.dirname(d), "image", name)
    if os.path.basename(d) == "kspace" and os.path.isfile(sibling):
        return sibling
    key = acc_key(name)
    if key:
        for cand in sorted(os.listdir(d)):
            if cand != name and cand.endswith(".h5") and acc_key(cand) == key and not cand.startswith("kspace"):
                return os.path.join(d, cand)
    return None


def find_files(root):
    """root(파일 또는 폴더) 아래의 k-space h5 -> {표시이름: {"kspace":…, "image":…|None}}."""
    if os.path.isfile(root):
        cands, base = [root], os.path.dirname(os.path.abspath(root)) or "."
    else:
        base = root
        cands = [os.path.join(d, fn) for d, _, fs in os.walk(root) for fn in sorted(fs) if fn.endswith(".h5")]
    found = {}
    for path in sorted(cands):
        try:
            with h5py.File(path, "r") as f:
                if "kspace" not in f:
                    continue
        except OSError:
            continue
        # 표시이름에서 'kspace' 폴더는 뺀다: train/kspace/x.h5 -> train/x.h5
        name = os.sep.join(p for p in os.path.relpath(path, base).split(os.sep) if p != "kspace")
        found[name] = {"kspace": path, "image": find_pair(path)}
    return found


def read_meta(entry):
    """뷰어 헤더에 띄울 파일 정보."""
    with h5py.File(entry["kspace"], "r") as f:
        n_slice, n_coil, height, width = f["kspace"].shape
        dtype = str(f["kspace"].dtype)
        has_mask = "mask" in f
        mask = f["mask"][:].astype(np.float32) if has_mask else np.ones(width, np.float32)
    sampled = int(mask.sum())
    pair_keys, annotated = [], []
    if entry["image"]:
        with h5py.File(entry["image"], "r") as f:
            pair_keys = [p for p, ds in PAIR_KEYS.items() if ds in f]
        annotated = sorted(load_annotations(entry["image"]))
    return {
        "slices": int(n_slice), "coils": int(n_coil),
        "height": int(height), "width": int(width), "dtype": dtype,
        "has_mask": has_mask, "sampled": sampled,
        "acc": round(width / sampled, 2) if sampled else 0.0,
        "acs": acs_lines(mask),
        "pair": os.path.basename(entry["image"]) if entry["image"] else None,
        "pair_keys": pair_keys, "annotated": annotated,
    }


def acs_lines(mask):
    """중앙 저주파(ACS) 블록 길이 = 1이 연속으로 가장 길게 이어지는 구간."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return int(best)


def load_annotations(image_path):
    """image h5 의 attrs['annotations'] -> {slice: [box, ...]}. 없으면 빈 dict."""
    with h5py.File(image_path, "r") as f:
        raw = f.attrs.get("annotations")
    if raw is None:
        return {}
    try:
        return {int(k): v for k, v in json.loads(raw).items()}
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return {}


def load_slice(path, idx):
    """(kspace (C,H,W) complex64, mask (W,)) — 최근 몇 장은 캐시."""
    key = (path, idx)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        with h5py.File(path, "r") as f:
            k = f["kspace"][idx]
            mask = f["mask"][:].astype(np.float32) if "mask" in f else np.ones(k.shape[-1], np.float32)
        if len(_CACHE) >= CACHE_SLOTS:
            _CACHE.clear()
        _CACHE[key] = (k, mask)
        return k, mask


# ======================================================================
# k-space 성분 그리기
# ======================================================================

def to_pil(arr01):
    return Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8))


def db_scale(mag, dr):
    """크기 -> dB. 최댓값이 0dB, 바닥이 -dr dB 이 되도록 [0,1] 로 눌러 담는다."""
    peak = float(mag.max())
    if peak <= 0:
        return np.zeros(mag.shape, np.float32)
    db = 20.0 * np.log10(np.maximum(mag, peak * 1e-12) / peak)
    return np.clip((db + dr) / dr, 0.0, 1.0).astype(np.float32)


def diverging(arr, dr):
    """부호 있는 값(Re·Im) -> 빨강(+) · 검정(0) · 파랑(-).

    밝기는 |값| 을 dB 로 눌러 담는다. k-space 는 중앙 몇 픽셀이 나머지보다 수천 배 커서
    선형으로 그리면 중앙만 타고 나머지는 새까맣다.
    """
    level = db_scale(np.abs(arr), dr)
    pos = (level * (arr > 0) * 255).astype(np.uint8)
    neg = (level * (arr < 0) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([pos, np.zeros_like(pos), neg], -1))


def cyclic_phase(arr, mag, dr):
    """위상 -> 색상환. cos 셋을 120°씩 어긋나게 겹치면 -π 와 +π 가 같은 색이라 이음매가 없다.

    밝기는 |k| (dB). 크기가 없는 곳의 위상은 난수라 어둡게 깔아야 구조가 보인다.
    """
    level = db_scale(mag, dr)[..., None]
    ch = np.stack([np.cos(arr - t * 2 * np.pi / 3) * 0.5 + 0.5 for t in (0, 1, 2)], -1)
    return Image.fromarray((ch * level * 255).astype(np.uint8))


def norm01(arr, gamma=1.0):
    """이미지(재구성·label 등) 밝기 정규화. 상위 백분위를 흰색으로 두고 감마."""
    lo, hi = float(arr.min()), float(np.percentile(arr, DISPLAY_PCT))
    if hi <= lo:
        return np.zeros(arr.shape, np.float32)
    v = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return v ** gamma if gamma != 1.0 else v


def kspace_cell(k2d, comp, dr):
    """복소수 2D k-space 한 장 -> PIL. comp: mag | re | im | phase."""
    if comp == "re":
        return diverging(k2d.real, dr)
    if comp == "im":
        return diverging(k2d.imag, dr)
    if comp == "phase":
        return cyclic_phase(np.angle(k2d), np.abs(k2d), dr)
    return to_pil(db_scale(np.abs(k2d), dr))


def coil_pick(k, coil, comp):
    """표시할 2D 배열 하나. mag 만 코일 합성이 의미가 있고, 나머지는 coil 0 으로."""
    if coil == "rss":
        if comp == "mag":
            return rss(k, 0).astype(np.complex64)   # 크기만 쓰므로 실수를 그대로 담는다
        return k[0]
    return k[coil]


def recon_cell(k, coil, crop, gamma):
    """ifft 재구성 한 장 (옵션 뷰)."""
    img = rss(ifftc(k), 0) if coil == "rss" else np.abs(ifftc(k[coil]))
    return to_pil(norm01(crop_to_384(img) if crop else img, gamma))


def montage(items, cols=None, max_side=PANEL_MAX):
    """items: [(캡션, PIL 또는 [0,1] 2D)] -> 캡션 붙인 격자 이미지 하나."""
    cells = []
    for caption, arr in items:
        im = (arr if isinstance(arr, Image.Image) else to_pil(arr)).convert("RGB")
        if max_side and max(im.size) != max_side:
            r = max_side / max(im.size)   # 중앙 확대처럼 작은 패널은 키운다(픽셀 그대로 NEAREST)
            im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                           Image.NEAREST if r > 1 else Image.BILINEAR)
        cells.append((caption, im))
    cols = cols or int(np.ceil(np.sqrt(len(cells))))
    rows = int(np.ceil(len(cells) / cols))
    cw = max(im.width for _, im in cells)
    ch = max(im.height for _, im in cells) + CAPTION_H
    out = Image.new("RGB", (cols * cw, rows * ch), "black")
    draw = ImageDraw.Draw(out)
    for i, (caption, im) in enumerate(cells):
        x, y = (i % cols) * cw, (i // cols) * ch
        out.paste(im, (x, y + CAPTION_H))
        draw.text((x + 3, y + 3), caption, fill="yellow")
    return out


def profile_png(k, mask, dr, scale=2):
    """위상-인코딩 라인별 에너지 막대. 초록=샘플된 라인, 빨강=마스크가 버린 라인.

    에너지는 **마스크 전** k-space 로 잰다 — 버려진 라인이 원래 얼마나 들고 있었는지가 요점.
    """
    energy = np.sqrt((np.abs(k) ** 2).sum(axis=(0, 1)))
    peak = float(energy.max()) or 1.0
    db = np.clip(20 * np.log10(np.maximum(energy, peak * 1e-12) / peak), -dr, 0.0)
    height = (db + dr) / dr * (PROFILE_H - 26)
    img = Image.new("RGB", (len(energy) * scale, PROFILE_H), "black")
    draw = ImageDraw.Draw(img)
    for j, h in enumerate(height):
        color = (70, 220, 130) if mask[j] else (215, 70, 70)
        draw.rectangle([j * scale, PROFILE_H - h, j * scale + scale - 1, PROFILE_H], fill=color)
    kept = float((energy * mask).sum() / energy.sum() * 100)
    # PIL 기본 비트맵 폰트에 한글 글리프가 없어 그림 안 글씨는 영문으로 둔다(웹 캡션은 한글).
    draw.text((5, 4), f"line energy  top 0dB / bottom -{int(dr)}dB   "
                      f"green {int(mask.sum())} sampled / red {int((mask == 0).sum())} dropped   "
                      f"energy kept {kept:.1f}%", fill="white")
    return img


# ======================================================================
# 요청 하나 = 패널 하나
# ======================================================================

def boxes_for(entry, idx):
    return load_annotations(entry["image"]).get(idx, []) if entry["image"] else []


def render(entry, idx, comp, part, coil, dr, gamma, kzoom, crop, bbox):
    """웹 /img 요청 하나 -> PIL 이미지. part: full | masked | mask | diff | label | input | grappa."""
    if part in PAIR_KEYS:
        with h5py.File(entry["image"], "r") as f:
            img = to_pil(norm01(f[PAIR_KEYS[part]][idx], gamma))
        boxes = boxes_for(entry, idx) if bbox else []
        return draw_boxes(img, boxes, scale=1.0) if boxes else img

    k, mask = load_slice(entry["kspace"], idx)
    if part == "mask":
        band = np.tile(mask, (k.shape[1], 1))          # 옆 k-space 패널과 높이를 맞춘다
        return to_pil(zoom_center(band, kzoom))
    km = k * mask[None, None, :] if part == "masked" else k

    if comp == "recon":
        if part == "diff":
            a = np.asarray(recon_cell(k, coil, crop, gamma), np.float32)
            b = np.asarray(recon_cell(k * mask[None, None, :], coil, crop, gamma), np.float32)
            img = to_pil(norm01(np.abs(a - b)))
        elif coil == "grid":
            return montage([(f"coil {c}", recon_cell(km[c:c + 1], "rss", crop, gamma))
                            for c in range(km.shape[0])])
        else:
            img = recon_cell(km, coil, crop, gamma)
        boxes = boxes_for(entry, idx) if bbox and img.size == (CROP_SIZE, CROP_SIZE) else []
        return draw_boxes(img, boxes, scale=1.0) if boxes else img

    kz = zoom_center(km, kzoom)
    if coil == "grid":
        return montage([(f"coil {c}", kspace_cell(kz[c], comp, dr)) for c in range(kz.shape[0])])
    return kspace_cell(coil_pick(kz, coil, comp), comp, dr)


def value_at(entry, idx, part, coil, kzoom, x, y):
    """클릭한 화면 좌표 -> 그 지점의 복소수 k-space 값 (숫자 그대로)."""
    k, mask = load_slice(entry["kspace"], idx)
    c = 0 if coil in ("rss", "grid") else coil
    y0, x0, hh, ww = zoom_box(k.shape[-2:], kzoom)
    yy, xx = y0 + int(np.clip(y, 0, hh - 1)), x0 + int(np.clip(x, 0, ww - 1))
    sampled = bool(mask[xx])
    v = complex(k[c, yy, xx]) * (float(mask[xx]) if part == "masked" else 1.0)
    peak = float(np.abs(k[c]).max()) or 1.0
    mag = abs(v)
    return {
        "ky": yy - k.shape[1] // 2, "kx": xx - k.shape[2] // 2, "row": yy, "col": xx,
        "coil": c, "re": v.real, "im": v.imag, "mag": mag,
        "db": 20 * np.log10(mag / peak) if mag > 0 else None,
        "deg": float(np.degrees(np.angle(v))) if mag > 0 else None,
        "rss": float(rss(k[:, yy, xx], 0)), "sampled": sampled,
    }


def export_sheet(entry, idx, out_png, dr=60.0, kzoom=1):
    """한 슬라이스 k-space 요약 시트: 성분 4종 + 마스크 전/후 + 라인 에너지."""
    k, mask = load_slice(entry["kspace"], idx)
    km = zoom_center(k * mask[None, None, :], kzoom)
    kz = zoom_center(k, kzoom)
    items = [
        ("|k| full (coil RSS, dB)", kspace_cell(rss(kz, 0).astype(np.complex64), "mag", dr)),
        ("|k| masked", kspace_cell(rss(km, 0).astype(np.complex64), "mag", dr)),
        ("mask", to_pil(zoom_center(np.tile(mask, (k.shape[1], 1)), kzoom))),
        ("Re k (coil 0)", kspace_cell(kz[0], "re", dr)),
        ("Im k (coil 0)", kspace_cell(kz[0], "im", dr)),
        ("phase of k (coil 0)", kspace_cell(kz[0], "phase", dr)),
    ]
    grid = montage(items, cols=3, max_side=420)
    prof = profile_png(k, mask, dr)
    sheet = Image.new("RGB", (max(grid.width, prof.width), grid.height + prof.height), "black")
    sheet.paste(grid, (0, 0))
    sheet.paste(prof, (0, grid.height))
    if os.path.dirname(out_png):
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
    sheet.save(out_png, "PNG")
    return sheet.size


# ======================================================================
# 웹 서버
# ======================================================================

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fastMRI k-space viewer</title>
<style>
 :root { color-scheme: dark; }
 body { margin:0; background:#111; color:#ddd; font:13px/1.5 system-ui, sans-serif; }
 header { display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center;
          padding:10px 14px; background:#1b1b1b; border-bottom:1px solid #333; position:sticky; top:0; z-index:1; }
 select, input[type=range] { background:#262626; color:#ddd; border:1px solid #3a3a3a; border-radius:4px; }
 select { padding:3px 6px; max-width:320px; }
 label { display:flex; gap:5px; align-items:center; white-space:nowrap; }
 #imgopts { display:flex; gap:10px 14px; align-items:center; }   /* 이미지 쪽에만 쓰는 옵션 */
 #imgopts[hidden] { display:none; }
 #info { padding:6px 14px; color:#8bc; background:#161616; border-bottom:1px solid #282828; }
 #readout { padding:6px 14px; background:#141414; border-bottom:1px solid #282828;
            font-family:ui-monospace, Menlo, monospace; color:#ffd479; }
 .row { display:flex; flex-wrap:wrap; gap:16px; padding:14px 16px 0; align-items:flex-start; }
 .row + .row { border-top:1px solid #262626; margin-top:12px; }
 figure { margin:0; }
 figcaption { margin-top:5px; color:#999; }
 img { display:block; background:#000; image-rendering:pixelated; zoom:var(--z,1); }
 .k img { cursor:crosshair; }
 .hint { color:#666; padding:14px; }
</style></head><body>
<header>
  <select id="file"></select>
  <select id="comp">
    <option value="mag">|k| 크기 (dB)</option>
    <option value="re">실수부 Re k</option>
    <option value="im">허수부 Im k</option>
    <option value="phase">위상 ∠k</option>
    <option value="recon">— ifft 재구성 —</option>
  </select>
  <select id="coil"></select>
  <select id="kzoom">
    <option value="1">전체</option>
    <option value="2">중앙 ×2</option>
    <option value="4">중앙 ×4</option>
    <option value="8">중앙 ×8 (ACS)</option>
  </select>
  <label>dR <input type="range" id="dr" min="20" max="140" step="5" value="60"><span id="drv"></span>dB</label>
  <label>배율 <input type="range" id="zoom" min="0.5" max="3" step="0.25" value="1"><span id="zv"></span>x</label>
  <label>slice <input type="range" id="slice" min="0" value="0"><span id="sv"></span></label>
  <label><input type="checkbox" id="profile" checked> 라인 에너지</label>
  <label><input type="checkbox" id="pair"> 짝 image h5</label>
  <span id="imgopts" hidden>
    <label><input type="checkbox" id="crop" checked> 384 crop</label>
    <label>&gamma; <input type="range" id="gamma" min="0.3" max="2" step="0.05" value="0.8"><span id="gv"></span></label>
  </span>
</header>
<div id="info">읽는 중...</div>
<div id="readout">k-space 그림을 클릭하면 그 지점의 복소수 값을 여기에 찍는다.</div>
<div id="stage"></div>
<p class="hint">&larr; &rarr; 슬라이스 &middot; [ ] 코일 &middot; 1 2 4 8 중앙 확대 &middot; b bbox 슬라이스로 점프</p>
<script>
const $ = id => document.getElementById(id);
const PARTS = [['full','full — 마스크 없음'], ['masked','masked — 마스크 적용'], ['mask','mask 패턴']];
const RECON = [['full','full RSS (= label)'], ['masked','masked RSS (= input)'], ['diff','|full - masked|']];
let meta = null;

async function init() {
  const files = await (await fetch('/api/files')).json();
  $('file').innerHTML = files.map(f => `<option>${f}</option>`).join('');
  await loadMeta();
  $('file').onchange = loadMeta;
  ['comp','coil','kzoom','dr','zoom','slice','profile','pair','crop','gamma']
    .forEach(id => $(id).oninput = draw);
  document.onkeydown = onKey;
}

async function loadMeta() {
  meta = await (await fetch('/api/meta?f=' + encodeURIComponent($('file').value))).json();
  $('slice').max = meta.slices - 1;
  $('slice').value = Math.floor(meta.slices / 2);
  $('coil').innerHTML = '<option value="rss">RSS (코일 합성)</option>'
    + '<option value="grid">grid (전체 코일)</option>'
    + Array.from({length: meta.coils}, (_, c) => `<option value="${c}">coil ${c}</option>`).join('');
  $('pair').disabled = !meta.pair_keys.length;
  $('info').textContent = `kspace (${meta.slices}, ${meta.coils}, ${meta.height}, ${meta.width}) ${meta.dtype}`
    + ` · 가속 ${meta.acc}× (sampled ${meta.sampled}/${meta.width}, ACS ${meta.acs} lines)`
    + (meta.has_mask ? '' : ' · mask 없음(전부 sampled 취급)')
    + (meta.pair ? ` · 짝 ${meta.pair}` : '')
    + (meta.annotated.length ? ` · bbox slice ${meta.annotated.join(', ')}` : '');
  draw();
}

function params(part) {
  return new URLSearchParams({
    f: $('file').value, s: $('slice').value, comp: $('comp').value, coil: $('coil').value,
    part: part, dr: $('dr').value, gamma: $('gamma').value, kzoom: $('kzoom').value,
    crop: $('crop').checked ? 1 : 0, bbox: 1,
  });
}

function draw() {
  $('sv').textContent = $('slice').value;
  $('drv').textContent = $('dr').value;
  $('gv').textContent = (+$('gamma').value).toFixed(2);
  $('zv').textContent = $('zoom').value;
  document.documentElement.style.setProperty('--z', $('zoom').value);
  const comp = $('comp').value, coil = $('coil').value;
  $('imgopts').hidden = !(comp === 'recon' || $('pair').checked);   // 이미지 볼 때만 노출
  const one = (comp !== 'mag' && comp !== 'recon' && coil === 'rss') ? ' (coil 0)' : '';
  const fig = (part, cap) => `<figure><img src="/img?${params(part)}" alt="${cap}" data-part="${part}">`
                           + `<figcaption>${cap}</figcaption></figure>`;
  const rows = [];
  rows.push('<div class="row k">' + (comp === 'recon' ? RECON : PARTS)
    .map(([p, c]) => fig(p, c + (p === 'mask' ? '' : one))).join('') + '</div>');
  if ($('profile').checked) {
    rows.push(`<div class="row"><figure><img src="/profile?${params('full')}">`
            + '<figcaption>라인별 에너지 · 초록 = 샘플, 빨강 = 마스크가 버린 라인</figcaption></figure></div>');
  }
  if (meta.pair_keys.length && $('pair').checked) {
    rows.push('<div class="row">' + meta.pair_keys
      .map(k => fig(k, 'image_' + k)).join('') + '</div>');
  }
  $('stage').innerHTML = rows.join('');
  $('stage').querySelectorAll('.k img').forEach(im => im.onclick = readValue);
}

async function readValue(e) {
  const part = e.target.dataset.part;
  if (part === 'mask' || $('comp').value === 'recon' || $('coil').value === 'grid') return;
  const r = e.target.getBoundingClientRect(), z = +$('zoom').value;
  const p = params(part);
  p.set('x', Math.floor((e.clientX - r.left) / z));
  p.set('y', Math.floor((e.clientY - r.top) / z));
  const v = await (await fetch('/api/value?' + p)).json();
  const e6 = x => x.toExponential(3);
  $('readout').textContent =
    `(kx ${v.kx >= 0 ? '+' : ''}${v.kx}, ky ${v.ky >= 0 ? '+' : ''}${v.ky})  [row ${v.row}, col ${v.col}]  coil ${v.coil}`
    + `   k = ${e6(v.re)} ${v.im >= 0 ? '+' : '-'} ${e6(Math.abs(v.im))}i`
    + `   |k| = ${e6(v.mag)}${v.db === null ? '' : ` (${v.db.toFixed(1)} dB)`}`
    + `${v.deg === null ? '' : `   ∠ ${v.deg.toFixed(1)}°`}`
    + `   coil RSS = ${e6(v.rss)}   이 라인: ${v.sampled ? '샘플됨' : '마스크가 버림'}`;
}

function onKey(e) {
  const sl = $('slice'), co = $('coil');
  if (e.key === 'ArrowLeft')  sl.value = Math.max(+sl.min, +sl.value - 1);
  else if (e.key === 'ArrowRight') sl.value = Math.min(+sl.max, +sl.value + 1);
  else if (e.key === '[' || e.key === ']') {
    co.selectedIndex = Math.min(co.options.length - 1, Math.max(0, co.selectedIndex + (e.key === ']' ? 1 : -1)));
  }
  else if (['1','2','4','8'].includes(e.key)) $('kzoom').value = e.key;
  else if (e.key === 'b' && meta.annotated.length) {
    const next = meta.annotated.find(s => s > +sl.value);
    sl.value = next === undefined ? meta.annotated[0] : next;
  }
  else return;
  e.preventDefault();
  draw();
}

init();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    files = {}   # 표시이름 -> {"kspace":…, "image":…} (serve() 에서 채움)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            elif url.path == "/api/files":
                self._send(200, "application/json", json.dumps(sorted(self.files)).encode())
            elif url.path == "/api/meta":
                self._send(200, "application/json", json.dumps(read_meta(self._entry(q))).encode())
            elif url.path == "/api/value":
                value = value_at(self._entry(q), self._int(q, "s"), self._str(q, "part", "full"),
                                 self._coil(q), self._int(q, "kzoom", 1),
                                 self._int(q, "x"), self._int(q, "y"))
                self._send(200, "application/json", json.dumps(value).encode())
            elif url.path == "/img":
                self._send(200, "image/png", self._png(self._image(q)))
            elif url.path == "/profile":
                k, mask = load_slice(self._entry(q)["kspace"], self._int(q, "s"))
                self._send(200, "image/png", self._png(profile_png(k, mask, self._float(q, "dr", 60))))
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found")
        except Exception as e:                      # 브라우저에 원인을 그대로 보여준다
            self._send(500, "text/plain; charset=utf-8", f"{type(e).__name__}: {e}".encode())

    # ---------- 요청 파라미터 ----------

    def _entry(self, q):
        name = self._str(q, "f", "")
        if name not in self.files:
            raise KeyError(f"알 수 없는 파일: {name}")
        return self.files[name]

    def _str(self, q, key, default=""):
        return q.get(key, [default])[0]

    def _int(self, q, key, default=0):
        return int(self._str(q, key, str(default)))

    def _float(self, q, key, default=0.0):
        return float(self._str(q, key, str(default)))

    def _coil(self, q):
        coil = self._str(q, "coil", "rss")
        return coil if coil in ("rss", "grid") else int(coil)

    def _image(self, q):
        return render(
            self._entry(q), idx=self._int(q, "s"),
            comp=self._str(q, "comp", "mag"), part=self._str(q, "part", "full"),
            coil=self._coil(q), dr=self._float(q, "dr", 60), gamma=self._float(q, "gamma", 0.8),
            kzoom=self._int(q, "kzoom", 1),
            crop=self._str(q, "crop", "1") == "1", bbox=self._str(q, "bbox", "1") == "1",
        )

    # ---------- 응답 ----------

    def _png(self, img):
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass   # 요청 로그는 조용히


def serve(files, host, port, open_browser):
    Handler.files = files
    server = ThreadingHTTPServer((host, port), Handler)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    paired = sum(1 for e in files.values() if e["image"])
    print(f"k-space 뷰어: http://{shown}:{port}   "
          f"(파일 {len(files)}개, 그중 짝 image 있는 것 {paired}개, Ctrl+C 로 종료)")
    if open_browser:
        webbrowser.open(f"http://{shown}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        server.server_close()


def main():
    ap = argparse.ArgumentParser(description="fastMRI k-space h5 뷰어")
    ap.add_argument("root", nargs="?", default=".",
                    help="k-space h5 파일 또는 폴더 (/root/Data 를 주면 train·val·leaderboard 전부)")
    ap.add_argument("--export", metavar="OUT.png",
                    help="웹 대신 k-space 요약 시트 PNG 한 장을 저장 (root 가 폴더면 첫 파일)")
    ap.add_argument("--slice", type=int, default=None, help="--export 대상 슬라이스 (기본: 가운데)")
    ap.add_argument("--kzoom", type=int, default=1, help="--export 중앙 확대 배율 (1·2·4·8)")
    ap.add_argument("--host", default="127.0.0.1", help="원격 서버에서 열려면 0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="브라우저 자동 실행")
    args = ap.parse_args()

    files = find_files(args.root)
    if not files:
        raise SystemExit(f"'kspace' 데이터셋을 가진 h5 를 찾지 못했습니다: {args.root}")

    if args.export:
        name, entry = sorted(files.items())[0]
        idx = args.slice if args.slice is not None else read_meta(entry)["slices"] // 2
        size = export_sheet(entry, idx, args.export, kzoom=args.kzoom)
        print(f"{name} slice {idx} -> {args.export} ({size[0]}x{size[1]})")
    else:
        serve(files, args.host, args.port, args.open)


if __name__ == "__main__":
    main()
