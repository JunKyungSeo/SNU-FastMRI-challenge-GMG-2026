"""fastMRI k-space h5 viewer — 로컬 웹 UI + 헤드리스 PNG 내보내기.

파일 구조 (train/val/leaderboard 공통):
  <split>/kspace/*.h5   kspace (S, C, 640, 368) complex64  — 마스크를 곱하기 **전**의 full k-space
                        mask   (368,)           float32    — phase-encoding 방향 언더샘플 패턴
  <split>/image/*.h5    image_label / image_input / image_grappa (S, 384, 384)
                        attrs["annotations"] = 병변 bbox JSON

acc4 파일로 확인한 관계 (slice 13, 최대 오차 2e-11):
  RSS(ifftc(kspace))          -> crop -> image_label
  RSS(ifftc(kspace * mask))   -> crop -> image_input
  crop = readout 640 중앙 384 + phase 368 을 좌우 8px zero-pad 해서 384

화면은 두 줄이다. 윗줄은 마스크 적용 전/후/마스크 자체를 항상 나란히 놓고(언더샘플이
무엇을 버리는지 바로 보인다), 아랫줄은 같은 이름의 image h5 가 있으면 label·input·GRAPPA 를
붙인다. label 에는 bbox 를 그린다.

사용법:
  python kspace_viewer.py                          # 현재 폴더의 k-space h5 로 웹뷰어
  python kspace_viewer.py knee_acc4_1.h5           # 파일 하나만
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
DISPLAY_PCT = 99.5    # 이미지 밝기 상한(백분위). norm01 참고
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
        has_mask = "mask" in f
        mask = f["mask"][:].astype(np.float32) if has_mask else np.ones(width, np.float32)
    sampled = int(mask.sum())
    pair_keys, annotated = [], []
    if entry["image"]:
        with h5py.File(entry["image"], "r") as f:
            pair_keys = [k for k, ds in PAIR_KEYS.items() if ds in f]
        annotated = sorted(load_annotations(entry["image"]))
    return {
        "slices": int(n_slice), "coils": int(n_coil),
        "height": int(height), "width": int(width),
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
# 렌더링 — 모든 함수가 [0,1] float 2D 를 만들고, 마지막에 uint8 PNG 로
# ======================================================================

def db_scale(mag, dr):
    """크기 -> dB. 최댓값이 0dB, 바닥이 -dr dB 이 되도록 [0,1] 로 눌러 담는다."""
    peak = float(mag.max())
    if peak <= 0:
        return np.zeros(mag.shape, np.float32)
    db = 20.0 * np.log10(np.maximum(mag, peak * 1e-12) / peak)
    return np.clip((db + dr) / dr, 0.0, 1.0).astype(np.float32)


def norm01(arr, gamma=1.0):
    """상위 백분위를 흰색으로 두고 정규화한 뒤 감마. gamma < 1 이면 밝아진다.

    max 로 맞추면 밝은 점 하나에 화면 전체가 눌린다 — 예를 들어 acc8 볼륨은 99.9 백분위가
    max 의 0.3배라 min-max 로는 거의 새까맣게 보인다.
    """
    lo, hi = float(arr.min()), float(np.percentile(arr, DISPLAY_PCT))
    if hi <= lo:
        return np.zeros(arr.shape, np.float32)
    v = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return v ** gamma if gamma != 1.0 else v


def to_pil(arr01):
    return Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8))


def coil_image(k, coil, crop):
    """k: (C,H,W) -> 크기 이미지. coil='rss' 면 코일 합성."""
    img = rss(ifftc(k), 0) if coil == "rss" else np.abs(ifftc(k[coil]))
    return crop_to_384(img) if crop else img


def cell(k, mask, view, part, coil, dr, gamma, crop):
    """패널 하나를 [0,1] 2D 로. part: full | masked | diff."""
    km = k * mask[None, None, :] if part == "masked" else k
    if view == "image":
        if part == "diff":
            full = coil_image(k, coil, crop)
            under = coil_image(k * mask[None, None, :], coil, crop)
            return norm01(np.abs(full - under), gamma)
        return norm01(coil_image(km, coil, crop), gamma)
    if view == "phase":
        c = 0 if coil == "rss" else coil   # 위상은 코일 합성이 의미가 없어 coil 0
        return ((np.angle(km[c]) + np.pi) / (2 * np.pi)).astype(np.float32)
    mag = rss(km, 0) if coil == "rss" else np.abs(km[coil])
    return db_scale(mag, dr)


def montage(items, cols=None, max_side=PANEL_MAX):
    """items: [(캡션, [0,1] 2D)] -> 캡션 붙인 격자 이미지 하나."""
    cells = []
    for caption, arr in items:
        im = (arr if isinstance(arr, Image.Image) else to_pil(arr)).convert("RGB")
        if max_side and max(im.size) > max_side:
            r = max_side / max(im.size)
            im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.BILINEAR)
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


def pair_image(image_path, idx, part, gamma):
    """image h5 의 label/input/grappa 한 장 -> [0,1] 2D."""
    with h5py.File(image_path, "r") as f:
        return norm01(f[PAIR_KEYS[part]][idx], gamma)


def boxes_for(entry, idx):
    return load_annotations(entry["image"]).get(idx, []) if entry["image"] else []


def render(entry, idx, view, part, coil, dr, gamma, crop, bbox):
    """웹 /img 요청 하나 -> PIL 이미지."""
    if part in PAIR_KEYS:
        img = to_pil(pair_image(entry["image"], idx, part, gamma))
    elif part == "mask":
        k, mask = load_slice(entry["kspace"], idx)
        return to_pil(np.tile(mask, (k.shape[1], 1)))   # 옆 k-space 패널과 높이를 맞춘다
    else:
        k, mask = load_slice(entry["kspace"], idx)
        if coil == "grid":
            items = [(f"coil {c}", cell(k[c:c + 1], mask, view, part, 0, dr, gamma, crop))
                     for c in range(k.shape[0])]
            return montage(items)
        img = to_pil(cell(k, mask, view, part, coil, dr, gamma, crop))
    if bbox and img.size == (CROP_SIZE, CROP_SIZE):
        boxes = boxes_for(entry, idx)
        if boxes:
            img = draw_boxes(img, boxes, scale=1.0)
    return img


def export_sheet(entry, idx, out_png, dr=60.0, gamma=0.8):
    """한 슬라이스 요약 시트. 짝 image h5 가 있으면 label·input·GRAPPA 줄까지."""
    k, mask = load_slice(entry["kspace"], idx)
    km = k * mask[None, None, :]
    full, under = crop_to_384(rss(ifftc(k), 0)), crop_to_384(rss(ifftc(km), 0))
    items = [
        ("k-space full (coil RSS, dB)", db_scale(rss(k, 0), dr)),
        ("k-space masked", db_scale(rss(km, 0), dr)),
        ("mask", np.tile(mask, (k.shape[1], 1))),
        ("recon full = label", norm01(full, gamma)),
        ("recon masked = input", norm01(under, gamma)),
        ("diff |full - masked|", norm01(np.abs(full - under), gamma)),
    ]
    if entry["image"]:
        with h5py.File(entry["image"], "r") as f:
            available = [p for p, ds in PAIR_KEYS.items() if ds in f]
        boxes = boxes_for(entry, idx)
        for part in available:
            img = to_pil(pair_image(entry["image"], idx, part, gamma))
            caption = f"image_{part}"
            if part == "label" and boxes:
                img = draw_boxes(img, boxes, scale=1.0)
                caption += f" ({len(boxes)} bbox)"
            items.append((caption, img))
    sheet = montage(items, cols=3, max_side=420)
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
 #info { padding:6px 14px; color:#8bc; background:#161616; border-bottom:1px solid #282828; }
 .row { display:flex; flex-wrap:wrap; gap:16px; padding:14px 16px 0; align-items:flex-start; }
 .row + .row { border-top:1px solid #262626; margin-top:12px; }
 figure { margin:0; }
 figcaption { margin-top:5px; color:#999; }
 img { display:block; background:#000; image-rendering:pixelated; zoom:var(--z,1); }
 .hint { color:#666; padding:14px; }
</style></head><body>
<header>
  <select id="file"></select>
  <select id="view">
    <option value="kspace">k-space 크기 (dB)</option>
    <option value="phase">k-space 위상</option>
    <option value="image">이미지 (ifft)</option>
  </select>
  <select id="coil"></select>
  <label><input type="checkbox" id="crop" checked> 384 crop</label>
  <label><input type="checkbox" id="bbox" checked> bbox</label>
  <label><input type="checkbox" id="pair" checked> 짝 image h5</label>
  <label>dR <input type="range" id="dr" min="20" max="140" step="5" value="60"><span id="drv"></span>dB</label>
  <label>&gamma; <input type="range" id="gamma" min="0.3" max="2" step="0.05" value="0.8"><span id="gv"></span></label>
  <label>배율 <input type="range" id="zoom" min="0.5" max="3" step="0.25" value="1"><span id="zv"></span>x</label>
  <label>slice <input type="range" id="slice" min="0" value="0"><span id="sv"></span></label>
</header>
<div id="info">읽는 중...</div>
<div id="stage"></div>
<p class="hint">&larr; &rarr; 슬라이스 이동 &middot; [ ] 코일 이동 &middot; b bbox 슬라이스로 점프</p>
<script>
const $ = id => document.getElementById(id);
const ROWS = {
  kspace: [['full','full — 마스크 없음'], ['masked','masked — 마스크 적용'], ['mask','mask 패턴']],
  phase:  [['full','full 위상'],          ['masked','masked 위상'],          ['mask','mask 패턴']],
  image:  [['full','full RSS (= label)'], ['masked','masked RSS (= input)'], ['diff','|full - masked|']],
};
let meta = null;

async function init() {
  const files = await (await fetch('/api/files')).json();
  $('file').innerHTML = files.map(f => `<option>${f}</option>`).join('');
  await loadMeta();
  $('file').onchange = loadMeta;
  ['view','coil','crop','bbox','pair','dr','gamma','zoom','slice'].forEach(id => $(id).oninput = draw);
  document.onkeydown = onKey;
}

async function loadMeta() {
  meta = await (await fetch('/api/meta?f=' + encodeURIComponent($('file').value))).json();
  $('slice').max = meta.slices - 1;
  $('slice').value = meta.annotated.length ? meta.annotated[0] : Math.floor(meta.slices / 2);
  $('coil').innerHTML = '<option value="rss">RSS (코일 합성)</option>'
    + '<option value="grid">grid (전체 코일)</option>'
    + Array.from({length: meta.coils}, (_, c) => `<option value="${c}">coil ${c}</option>`).join('');
  $('pair').disabled = !meta.pair_keys.length;
  $('info').textContent = `${meta.slices} slices · ${meta.coils} coils · ${meta.height}×${meta.width}`
    + ` · 가속 ${meta.acc}× (sampled ${meta.sampled}/${meta.width}, ACS ${meta.acs} lines)`
    + (meta.has_mask ? '' : ' · mask 없음(전부 sampled 취급)')
    + (meta.pair ? ` · 짝 ${meta.pair} [${meta.pair_keys.join(', ')}]` : ' · 짝 image h5 없음')
    + (meta.annotated.length ? ` · bbox slice ${meta.annotated.join(', ')}` : '');
  draw();
}

function src(part) {
  const p = new URLSearchParams({
    f: $('file').value, s: $('slice').value, view: $('view').value, coil: $('coil').value,
    part: part, dr: $('dr').value, gamma: $('gamma').value,
    crop: $('crop').checked ? 1 : 0, bbox: $('bbox').checked ? 1 : 0,
  });
  return '/img?' + p.toString();
}

function draw() {
  $('sv').textContent = $('slice').value;
  $('drv').textContent = $('dr').value;
  $('gv').textContent = (+$('gamma').value).toFixed(2);
  $('zv').textContent = $('zoom').value;
  document.documentElement.style.setProperty('--z', $('zoom').value);
  const nbox = (meta.annotated.includes(+$('slice').value) && $('bbox').checked) ? ' · bbox' : '';
  const fig = (part, cap) => `<figure><img src="${src(part)}" alt="${cap}"><figcaption>${cap}</figcaption></figure>`;
  let html = '<div class="row">' + ROWS[$('view').value].map(([p, c]) => fig(p, c)).join('') + '</div>';
  if (meta.pair_keys.length && $('pair').checked) {
    html += '<div class="row">' + meta.pair_keys
      .map(k => fig(k, 'image_' + k + (k === 'label' ? nbox : ''))).join('') + '</div>';
  }
  $('stage').innerHTML = html;
}

function onKey(e) {
  const sl = $('slice'), co = $('coil');
  if (e.key === 'ArrowLeft')  sl.value = Math.max(+sl.min, +sl.value - 1);
  else if (e.key === 'ArrowRight') sl.value = Math.min(+sl.max, +sl.value + 1);
  else if (e.key === '[' || e.key === ']') {
    co.selectedIndex = Math.min(co.options.length - 1, Math.max(0, co.selectedIndex + (e.key === ']' ? 1 : -1)));
  }
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
            elif url.path == "/img":
                self._send(200, "image/png", self._png(q))
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found")
        except Exception as e:                      # 브라우저에 원인을 그대로 보여준다
            self._send(500, "text/plain; charset=utf-8", f"{type(e).__name__}: {e}".encode())

    def _entry(self, q):
        name = q.get("f", [""])[0]
        if name not in self.files:
            raise KeyError(f"알 수 없는 파일: {name}")
        return self.files[name]

    def _png(self, q):
        coil = q.get("coil", ["rss"])[0]
        img = render(
            self._entry(q),
            idx=int(q.get("s", ["0"])[0]),
            view=q.get("view", ["kspace"])[0],
            part=q.get("part", ["masked"])[0],
            coil=coil if coil in ("rss", "grid") else int(coil),
            dr=float(q.get("dr", ["60"])[0]),
            gamma=float(q.get("gamma", ["0.8"])[0]),
            crop=q.get("crop", ["1"])[0] == "1",
            bbox=q.get("bbox", ["1"])[0] == "1",
        )
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
                    help="웹 대신 요약 시트 PNG 한 장을 저장 (root 가 폴더면 첫 파일)")
    ap.add_argument("--slice", type=int, default=None, help="--export 대상 슬라이스 (기본: 가운데)")
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
        size = export_sheet(entry, idx, args.export)
        print(f"{name} slice {idx} -> {args.export} ({size[0]}x{size[1]})")
    else:
        serve(files, args.host, args.port, args.open)


if __name__ == "__main__":
    main()
