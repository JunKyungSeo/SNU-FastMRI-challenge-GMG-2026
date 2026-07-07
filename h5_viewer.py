"""fastMRI image h5 viewer.

Opens an image-type h5 file (keys: image_label, image_input, image_grappa,
optional attrs["annotations"]) and shows:
  - Images tab: every slice as a Label | Input | GRAPPA row (bbox drawn on Label)
  - BBox tab:   crops of image_label for every annotated bbox
  - Attrs tab:  raw file attributes as text

Usage:
  python h5_viewer.py [file.h5]        # GUI (open a single file)
  python h5_viewer.py --export         # headless: one stacked PNG per image h5
                                       #   under /root/Data/{train,val} -> ./viewer_result
  python h5_viewer.py --export DATA_ROOT OUT_DIR   # custom source / destination

Export layout: each h5 file == one subject (volume). Every slice is rendered as a
'Label | Input | GRAPPA' row (bbox on Label) and all of one subject's slices are
stacked vertically into a single PNG so the whole volume is viewable at once:
  {OUT_DIR}/{split}/{file_stem}.png
Deps :  pip install h5py numpy pillow
"""

import json
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw

DISPLAY_SIZE = 320   # on-screen size of each slice image (px)
CROP_SCALE = 3       # magnification of bbox crops
IMAGE_KEYS = ("image_label", "image_input", "image_grappa")
COL_TITLES = ("Label", "Input", "GRAPPA")

# ---------- batch export defaults ----------
DEFAULT_DATA_ROOT = "/root/Data"          # holds train/ and val/, each with image/
DEFAULT_OUT_DIR = "viewer_result"         # created in the current directory
EXPORT_SPLITS = ("train", "val")
TITLE_STRIP_H = 18                         # px reserved above the panels for the caption


def slice_to_uint8(sl):
    """Min-max normalize one slice to uint8 grayscale (matches tutorial's imshow look)."""
    sl = np.asarray(sl, dtype=np.float32)
    lo, hi = float(sl.min()), float(sl.max())
    if hi <= lo:
        return np.zeros(sl.shape, dtype=np.uint8)
    return ((sl - lo) / (hi - lo) * 255.0).astype(np.uint8)


def draw_boxes(pil_img, boxes, scale):
    """Draw bbox rectangles + labels on an RGB copy of pil_img (module-level, tk-free)."""
    pil_img = pil_img.convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    for box in boxes:
        x, y = box["x"] * scale, box["y"] * scale
        w, h = box["width"] * scale, box["height"] * scale
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        draw.text((x, max(0, y - 12)), str(box.get("label", "")).strip(), fill="red")
    return pil_img


# ======================================================================
# Headless batch export (no tkinter / no display required)
# ======================================================================

def read_image_h5(path):
    """Load an image-type h5 -> (volumes dict, annotations {slice: [boxes]}, attrs).

    Returns None if the file is not an image-type h5 (missing IMAGE_KEYS).
    """
    with h5py.File(path, "r") as f:
        if any(k not in f for k in IMAGE_KEYS):
            return None
        volumes = {k: f[k][:] for k in IMAGE_KEYS}
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}

    annotations = {}
    raw_ann = attrs.get("annotations")
    if raw_ann is not None:
        try:
            parsed = json.loads(raw_ann)
            annotations = {int(k): v for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
            pass
    return volumes, annotations, attrs


def render_slice(volumes, idx, boxes):
    """Render one slice as a Label | Input | GRAPPA composite (bbox drawn on Label).

    Full-resolution panels (no downscale) so lesion detail is preserved; a thin
    caption strip labels the three columns and the slice / bbox count.
    """
    panels = []
    for key, title in zip(IMAGE_KEYS, COL_TITLES):
        pil = Image.fromarray(slice_to_uint8(volumes[key][idx]), "L").convert("RGB")
        if key == "image_label" and boxes:
            pil = draw_boxes(pil, boxes, scale=1.0)
        panels.append((title, pil))

    total_w = sum(p.width for _, p in panels)
    max_h = max(p.height for _, p in panels)
    comp = Image.new("RGB", (total_w, max_h + TITLE_STRIP_H), "black")
    draw = ImageDraw.Draw(comp)
    x = 0
    for title, p in panels:
        comp.paste(p, (x, TITLE_STRIP_H))
        draw.text((x + 4, 4), title, fill="white")
        x += p.width
    suffix = f"  ({len(boxes)} bbox)" if boxes else ""
    draw.text((total_w - 150, 4), f"slice {idx}{suffix}", fill="yellow")
    return comp


def render_volume(volumes, annotations):
    """Stack every slice's 'Label | Input | GRAPPA' row into one tall image.

    The whole subject (volume) is viewable at once, top-to-bottom by slice index.
    """
    n = volumes["image_label"].shape[0]
    rows = [render_slice(volumes, idx, annotations.get(idx, [])) for idx in range(n)]
    width = max(r.width for r in rows)
    height = sum(r.height for r in rows)
    stack = Image.new("RGB", (width, height), "black")
    y = 0
    for r in rows:
        stack.paste(r, (0, y))
        y += r.height
    return stack


def export_file(path, dst_png):
    """Render one image h5 as a single stacked PNG at dst_png. Returns (#slices, #bytes)."""
    loaded = read_image_h5(path)
    if loaded is None:
        return 0, 0
    volumes, annotations, _ = loaded
    stack = render_volume(volumes, annotations)
    os.makedirs(os.path.dirname(dst_png), exist_ok=True)
    stack.save(dst_png, "PNG")
    return volumes["image_label"].shape[0], os.path.getsize(dst_png)


def export_all(data_root=DEFAULT_DATA_ROOT, out_root=DEFAULT_OUT_DIR):
    """Batch-render every image h5 under {data_root}/{split}/image into a single
    stacked PNG per subject: {out_root}/{split}/{file_stem}.png. GUI-free (PIL only)."""
    total_files = total_slices = total_bytes = 0
    for split in EXPORT_SPLITS:
        img_dir = os.path.join(data_root, split, "image")
        if not os.path.isdir(img_dir):
            print(f"[skip] {img_dir} (없음)")
            continue
        files = sorted(f for f in os.listdir(img_dir) if f.endswith(".h5"))
        print(f"[{split}] {len(files)} image h5 파일 처리 시작 -> {out_root}/{split}/")
        for i, fname in enumerate(files, 1):
            stem = os.path.splitext(fname)[0]
            dst_png = os.path.join(out_root, split, f"{stem}.png")
            n, b = export_file(os.path.join(img_dir, fname), dst_png)
            total_files += 1
            total_slices += n
            total_bytes += b
            print(f"  ({i}/{len(files)}) {fname}: {n} slices -> {stem}.png, {b / 1024 / 1024:.1f} MB")
    print(
        f"\n완료: {total_files} 파일(사람), {total_slices} slices, "
        f"총 {total_bytes / 1024 / 1024 / 1024:.2f} GB -> {out_root}/"
    )


# ======================================================================
# GUI (tkinter) — loaded lazily so --export works on headless servers
# ======================================================================

def run_gui(initial_path=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import ImageTk

    class ScrollFrame(ttk.Frame):
        """Vertically scrollable frame; put content in .inner."""

        def __init__(self, master):
            super().__init__(master)
            self.canvas = tk.Canvas(self, highlightthickness=0)
            vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=vsb.set)
            vsb.pack(side="right", fill="y")
            self.canvas.pack(side="left", fill="both", expand=True)

            self.inner = ttk.Frame(self.canvas)
            self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
            self.inner.bind(
                "<Configure>",
                lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
            )
            self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_wheel))
            self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

        def _on_wheel(self, event):
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

        def clear(self):
            for child in self.inner.winfo_children():
                child.destroy()
            self.canvas.yview_moveto(0)

    class Viewer(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("fastMRI h5 viewer")
            self.geometry("1200x900")
            self._photos = []  # keep PhotoImage references alive

            toolbar = ttk.Frame(self, padding=6)
            toolbar.pack(side="top", fill="x")
            ttk.Button(toolbar, text="h5 파일 열기...", command=self.open_dialog).pack(side="left")
            self.file_var = tk.StringVar(value="파일이 열려 있지 않습니다")
            ttk.Label(toolbar, textvariable=self.file_var, padding=(10, 0)).pack(side="left")

            nb = ttk.Notebook(self)
            nb.pack(fill="both", expand=True)
            self.images_tab = ScrollFrame(nb)
            self.bbox_tab = ScrollFrame(nb)
            attrs_frame = ttk.Frame(nb)
            self.attrs_text = tk.Text(attrs_frame, wrap="word", font=("Consolas", 10))
            attrs_vsb = ttk.Scrollbar(attrs_frame, orient="vertical", command=self.attrs_text.yview)
            self.attrs_text.configure(yscrollcommand=attrs_vsb.set, state="disabled")
            attrs_vsb.pack(side="right", fill="y")
            self.attrs_text.pack(side="left", fill="both", expand=True)
            nb.add(self.images_tab, text="Images")
            nb.add(self.bbox_tab, text="BBox")
            nb.add(attrs_frame, text="Attrs")

        # ---------- file loading ----------

        def open_dialog(self):
            path = filedialog.askopenfilename(
                title="image h5 파일 선택",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
            )
            if path:
                self.load_file(path)

        def load_file(self, path):
            try:
                loaded = read_image_h5(path)
            except OSError as e:
                messagebox.showerror("열기 실패", f"h5 파일을 열 수 없습니다:\n{e}")
                return
            if loaded is None:
                with h5py.File(path, "r") as f:
                    missing = [k for k in IMAGE_KEYS if k not in f]
                    keys = ", ".join(f.keys())
                messagebox.showerror(
                    "형식 오류",
                    f"image 형식 h5 파일이 아닙니다.\n"
                    f"없는 key: {', '.join(missing)}\n"
                    f"파일 key: {keys}",
                )
                return
            volumes, annotations, attrs = loaded

            self.file_var.set(f"{path}   |   slices: {volumes['image_label'].shape[0]}")
            self._photos.clear()
            self._build_images_tab(volumes, annotations)
            self._build_bbox_tab(volumes["image_label"], annotations)
            self._build_attrs_tab(attrs)

        # ---------- tabs ----------

        def _photo(self, pil_img):
            photo = ImageTk.PhotoImage(pil_img)
            self._photos.append(photo)
            return photo

        def _build_images_tab(self, volumes, annotations):
            self.images_tab.clear()
            n_slices = volumes["image_label"].shape[0]
            for idx in range(n_slices):
                row = ttk.Frame(self.images_tab.inner, padding=(8, 6))
                row.pack(anchor="w")
                boxes = annotations.get(idx, [])
                suffix = f"  ({len(boxes)} bbox)" if boxes else ""
                ttk.Label(row, text=f"slice {idx}{suffix}", font=("", 10, "bold")).pack(anchor="w")
                imgs = ttk.Frame(row)
                imgs.pack(anchor="w")
                for key, title in zip(IMAGE_KEYS, COL_TITLES):
                    pil = Image.fromarray(slice_to_uint8(volumes[key][idx]), "L")
                    scale = DISPLAY_SIZE / pil.width
                    pil = pil.resize((DISPLAY_SIZE, int(pil.height * scale)), Image.BILINEAR)
                    if key == "image_label" and boxes:
                        pil = draw_boxes(pil, boxes, scale)
                    cell = ttk.Frame(imgs, padding=(0, 0, 8, 0))
                    cell.pack(side="left")
                    ttk.Label(cell, text=title).pack()
                    ttk.Label(cell, image=self._photo(pil)).pack()

        def _build_bbox_tab(self, label_vol, annotations):
            self.bbox_tab.clear()
            if not annotations:
                ttk.Label(self.bbox_tab.inner, text="annotation 없음", padding=12).pack(anchor="w")
                return
            for idx in sorted(annotations):
                for box in annotations[idx]:
                    x, y = int(box["x"]), int(box["y"])
                    w, h = int(box["width"]), int(box["height"])
                    sl = slice_to_uint8(label_vol[idx])
                    y0, y1 = max(0, y), min(sl.shape[0], y + h)
                    x0, x1 = max(0, x), min(sl.shape[1], x + w)
                    if y1 <= y0 or x1 <= x0:
                        continue
                    crop = Image.fromarray(sl[y0:y1, x0:x1], "L")
                    crop = crop.resize((crop.width * CROP_SCALE, crop.height * CROP_SCALE), Image.BILINEAR)
                    cell = ttk.Frame(self.bbox_tab.inner, padding=(8, 6))
                    cell.pack(anchor="w")
                    label = str(box.get("label", "")).strip()
                    ttk.Label(
                        cell,
                        text=f"slice {idx}   {label}   (x={x}, y={y}, w={w}, h={h})",
                        font=("", 10, "bold"),
                    ).pack(anchor="w")
                    ttk.Label(cell, image=self._photo(crop)).pack(anchor="w")

        def _build_attrs_tab(self, attrs):
            self.attrs_text.configure(state="normal")
            self.attrs_text.delete("1.0", "end")
            if not attrs:
                self.attrs_text.insert("end", "(attrs 없음)\n")
            for key, value in attrs.items():
                self.attrs_text.insert("end", f"=== {key} ===\n")
                text = value
                if key == "annotations":
                    try:
                        text = json.dumps(json.loads(value), indent=2, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        pass
                self.attrs_text.insert("end", f"{text}\n\n")
            self.attrs_text.configure(state="disabled")

    app = Viewer()
    if initial_path:
        app.after(50, lambda: app.load_file(initial_path))
    app.mainloop()


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--export":
        data_root = argv[1] if len(argv) > 1 else DEFAULT_DATA_ROOT
        out_root = argv[2] if len(argv) > 2 else DEFAULT_OUT_DIR
        export_all(data_root, out_root)
    else:
        run_gui(argv[0] if argv else None)


if __name__ == "__main__":
    main()
