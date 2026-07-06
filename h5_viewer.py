"""fastMRI image h5 viewer.

Opens an image-type h5 file (keys: image_label, image_input, image_grappa,
optional attrs["annotations"]) and shows:
  - Images tab: every slice as a Label | Input | GRAPPA row (bbox drawn on Label)
  - BBox tab:   crops of image_label for every annotated bbox
  - Attrs tab:  raw file attributes as text

Usage:  python h5_viewer.py [file.h5]
Deps :  pip install h5py numpy pillow
"""

import json
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageTk

DISPLAY_SIZE = 320   # on-screen size of each slice image (px)
CROP_SCALE = 3       # magnification of bbox crops
IMAGE_KEYS = ("image_label", "image_input", "image_grappa")
COL_TITLES = ("Label", "Input", "GRAPPA")


def slice_to_uint8(sl):
    """Min-max normalize one slice to uint8 grayscale (matches tutorial's imshow look)."""
    sl = np.asarray(sl, dtype=np.float32)
    lo, hi = float(sl.min()), float(sl.max())
    if hi <= lo:
        return np.zeros(sl.shape, dtype=np.uint8)
    return ((sl - lo) / (hi - lo) * 255.0).astype(np.uint8)


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
            with h5py.File(path, "r") as f:
                missing = [k for k in IMAGE_KEYS if k not in f]
                if missing:
                    messagebox.showerror(
                        "형식 오류",
                        f"image 형식 h5 파일이 아닙니다.\n"
                        f"없는 key: {', '.join(missing)}\n"
                        f"파일 key: {', '.join(f.keys())}",
                    )
                    return
                volumes = {k: f[k][:] for k in IMAGE_KEYS}
                attrs = {k: f.attrs[k] for k in f.attrs.keys()}
        except OSError as e:
            messagebox.showerror("열기 실패", f"h5 파일을 열 수 없습니다:\n{e}")
            return

        annotations = {}
        raw_ann = attrs.get("annotations")
        if raw_ann is not None:
            try:
                parsed = json.loads(raw_ann)
                annotations = {int(k): v for k, v in parsed.items()}
            except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
                pass  # Attrs tab still shows the raw string

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
                    pil = self._draw_boxes(pil, boxes, scale)
                cell = ttk.Frame(imgs, padding=(0, 0, 8, 0))
                cell.pack(side="left")
                ttk.Label(cell, text=title).pack()
                ttk.Label(cell, image=self._photo(pil)).pack()

    @staticmethod
    def _draw_boxes(pil_img, boxes, scale):
        pil_img = pil_img.convert("RGB")
        draw = ImageDraw.Draw(pil_img)
        for box in boxes:
            x, y = box["x"] * scale, box["y"] * scale
            w, h = box["width"] * scale, box["height"] * scale
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
            draw.text((x, max(0, y - 12)), str(box.get("label", "")).strip(), fill="red")
        return pil_img

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


def main():
    app = Viewer()
    if len(sys.argv) > 1:
        app.after(50, lambda: app.load_file(sys.argv[1]))
    app.mainloop()


if __name__ == "__main__":
    main()
