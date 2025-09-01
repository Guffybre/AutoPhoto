import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from io import BytesIO
import base64
import threading
from SegmentModification import SegmentModification
import numpy as np

class EnhancedSegmentDisplay:
    def __init__(self, root, processor):
        self.root = root
        self.processor = processor
        self.modifier = SegmentModification(processor)
        self.segment_controls = {}
        self.original_segments = {}
        self.current_segment_id = None

        # Main unified window
        self.window = tk.Toplevel(self.root)
        self.window.title("Segments - Unified View")
        self.window.geometry("1000x700")

        # Layout: top thumbnails, bottom controls
        self.thumbnails_frame = tk.Canvas(self.window, bg="white", height=200)
        self.thumbnails_frame.pack(fill="x", padx=10, pady=10)
        self.thumbnails_frame.bind("<Button-1>", self._on_segment_click)

        self.preview_canvas = tk.Canvas(self.window, bg="white", height=300)
        self.preview_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self.controls_frame = ttk.LabelFrame(self.window, text="Adjustments", padding=10)
        self.controls_frame.pack(fill="x", padx=10, pady=10)

    def show_all_segments(self):
        result = self.modifier.load_segments()
        if not result["success"]:
            messagebox.showerror("Error", result["error"])
            return

        segs = self.processor.get_all_segments()
        if not segs["success"]:
            messagebox.showerror("Error", segs["error"])
            return

        self.original_segments.clear()
        self.segment_controls.clear()
        self.thumbnails_frame.delete("all")

        x_offset = 10
        for segment in segs["segments"]:
            seg_id = segment["id"]
            self.original_segments[seg_id] = segment["image"]

            # Decode segment image
            img_data = base64.b64decode(segment["image"])
            pil_img = Image.open(BytesIO(img_data))
            pil_img.thumbnail((120, 120))
            photo = ImageTk.PhotoImage(pil_img)

            # Draw with red border
            self.thumbnails_frame.create_rectangle(
                x_offset-2, 5, x_offset+124, 129, outline="red", width=2
            )
            self.thumbnails_frame.create_image(x_offset+60, 67, image=photo)

            # Keep reference
            self.thumbnails_frame.image = getattr(self.thumbnails_frame, "image", []) + [photo]
            # Tag for click
            self.thumbnails_frame.addtag_withtag(f"seg{seg_id}", self.thumbnails_frame.create_rectangle(x_offset, 5, x_offset+120, 125))

            x_offset += 130

        # Auto‑select first segment
        if segs["segments"]:
            self._select_segment(segs["segments"][0]["id"])

    def _on_segment_click(self, event):
        item = self.thumbnails_frame.find_closest(event.x, event.y)
        tags = self.thumbnails_frame.gettags(item)
        for t in tags:
            if t.startswith("seg"):
                seg_id = int(t[3:])
                self._select_segment(seg_id)

    def _select_segment(self, seg_id):
        self.current_segment_id = seg_id
        self._show_segment_preview(seg_id)
        self._create_controls(seg_id)

    def _show_segment_preview(self, seg_id):
        self.preview_canvas.delete("all")
        img_data = base64.b64decode(self.original_segments[seg_id])
        pil_img = Image.open(BytesIO(img_data))
        pil_img.thumbnail((400, 300))
        photo = ImageTk.PhotoImage(pil_img)
        self.preview_canvas.create_image(200, 150, image=photo)
        self.preview_canvas.image = photo

    def _create_controls(self, seg_id):
        for child in self.controls_frame.winfo_children():
            child.destroy()

        # Example: brightness slider (same pattern for others)
        ttk.Label(self.controls_frame, text="Brightness").pack(anchor="w")
        var = tk.DoubleVar(value=0)
        ttk.Scale(self.controls_frame, from_=-100, to=100, variable=var,
                  orient="horizontal").pack(fill="x")
        self.segment_controls[seg_id] = {"brightness": var}

        # (repeat: contrast, gamma, etc. like your old code)

        # Apply button
        ttk.Button(self.controls_frame, text="Apply",
                   command=lambda: self._apply_changes(seg_id)).pack(pady=5)

    def _apply_changes(self, seg_id):
        messagebox.showinfo("Applied", f"Applied changes to Segment {seg_id+1}")
