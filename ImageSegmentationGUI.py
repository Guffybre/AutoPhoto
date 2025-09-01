
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
from PIL import Image, ImageTk
import numpy as np
from threading import Thread
import base64
from io import BytesIO

from EnhancedSegmentDisplay import EnhancedSegmentDisplay
from ImageSegmentationProcessor import ImageSegmentationProcessor
class ImageSegmentationGUI:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Image Segmentation Tool")
        self.root.geometry("1200x800")

        self.processor = ImageSegmentationProcessor()
        self.current_display_image = None

        # Setup UI and enhanced display with a right-side panel
        self.setup_ui()
        self.setup_drag_drop()
        # Pass the right-side frame to the enhanced display so it docks inside main window
        self.enhanced_display = EnhancedSegmentDisplay(self.right_panel_frame, self.processor)
        self.root.app = self

    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="8")
        main_frame.grid(row=0, column=0, sticky=("nsew"))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Configure two columns: left=image area (expand), right=segments panel (fixed)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)
        main_frame.rowconfigure(1, weight=1)

        # Control panel (top, across both columns)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=("we"), pady=(0, 8))

        # File operations
        ttk.Button(control_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(control_frame, text="Watershed Segment", command=self.segment_watershed).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(control_frame, text="NN segment", command=self.segment_nn).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(control_frame, text="Show All Segments", command=self.show_all_segments).grid(row=0, column=3, padx=(20, 0))

        # Image display area (left)
        self.image_frame = ttk.LabelFrame(main_frame, text="Image Display", padding="6")
        self.image_frame.grid(row=1, column=0, sticky=("nsew"), padx=(0, 8))
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)

        # Canvas for image display
        self.canvas = tk.Canvas(self.image_frame, bg="white")
        self.canvas.grid(row=0, column=0, sticky=("nsew"))

        # Scrollbars for canvas
        v_scrollbar = ttk.Scrollbar(self.image_frame, orient="vertical", command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=("ns"))
        self.canvas.configure(yscrollcommand=v_scrollbar.set)

        h_scrollbar = ttk.Scrollbar(self.image_frame, orient="horizontal", command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=("we"))
        self.canvas.configure(xscrollcommand=h_scrollbar.set)

        # Drag and drop label (centered placeholder)
        self.drop_label = ttk.Label(self.canvas, text="Drag and drop an image here\nor click 'Load Image'",
                                    font=("Arial", 14), foreground="gray")
        self.canvas.create_window(400, 300, window=self.drop_label)

        # Right-side segments panel
        self.right_panel_frame = ttk.Frame(main_frame, width=340)
        self.right_panel_frame.grid(row=1, column=1, sticky=("ns"))
        # Make sure it doesn't expand when window widened
        self.right_panel_frame.grid_propagate(False)

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Drag and drop an image or click Load Image")
        ttk.Label(main_frame, textvariable=self.status_var, relief="sunken").grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(8, 0))

    def setup_drag_drop(self):
        """Setup drag and drop functionality"""
        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind('<<Drop>>', self.on_drop)

    def on_drop(self, event):
        """Handle drag and drop event"""
        files = self.root.tk.splitlist(event.data)
        if files:
            self.load_image_file(files[0])

    def load_image(self):
        """Load image through file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if file_path:
            self.load_image_file(file_path)

    def load_image_file(self, file_path: str):
        """Load and display image"""
        self.status_var.set("Loading image...")
        result = self.processor.load_image(file_path)
        if result["success"]:
            self.display_current_image()
            self.status_var.set(f"Image loaded: {result['width']}x{result['height']} pixels")
            try:
                self.drop_label.destroy()
            except:
                pass
        else:
            messagebox.showerror("Error", f"Failed to load image: {result['error']}")
            self.status_var.set("Ready")

    def display_current_image(self):
        """Display the current image on canvas"""
        if self.processor.current_image is not None:
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(self.processor.current_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Resize image if too large to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                pil_image.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)

            self.current_display_image = ImageTk.PhotoImage(pil_image)

            # Clear canvas and display image
            self.canvas.delete("all")
            self.canvas.create_image(10, 10, anchor="nw", image=self.current_display_image)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def segment_watershed(self):
        """Perform watershed segmentation (wrapped in thread)"""
        self.status_var.set("Performing watershed segmentation...")
        thread = Thread(target=self._run_watershed_segmentation)
        thread.daemon = True
        thread.start()

    def _run_watershed_segmentation(self):
        result = self.processor.segment_watershed()
        self.root.after(0, self._on_segmentation_complete, result, "Watershed")

    def segment_nn(self):
        """Perform NN segmentation"""
        self.status_var.set("Performing NN segmentation...")
        thread = Thread(target=self._run_segment_nn)
        thread.daemon = True
        thread.start()

    def _run_segment_nn(self):
        result = self.processor.segment_deeplab()
        self.root.after(0, self._on_segmentation_complete, result, "NN")

    def _on_segmentation_complete(self, result, method):
        """Handle segmentation completion"""
        if result["success"]:
            # Display segmented image
            self._display_base64_image(result["segmented_image"])
            self.status_var.set(f"{method} segmentation complete - {result['num_segments']} segments found")
        else:
            messagebox.showerror("Error", f"Segmentation failed: {result['error']}")
            self.status_var.set("Ready")

    def _display_base64_image(self, base64_string):
        """Display base64 encoded image on the main canvas"""
        image_data = base64.b64decode(base64_string)
        pil_image = Image.open(BytesIO(image_data))

        # Resize if needed to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            pil_image.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)

        self.current_display_image = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(10, 10, anchor="nw", image=self.current_display_image)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def show_all_segments(self):
        """Ask enhanced display to populate the docked panel (no popups)"""
        self.enhanced_display.show_all_segments()

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


# --- ENHANCED SEGMENT DISPLAY (DOCKED PANEL) ---
import threading
from SegmentModification import SegmentModification

class EnhancedSegmentDisplay:
    """
    Docked segments panel: shows selectable thumbnails of segments with red separators,
    single preview and inline controls (no popups).
    """

    def __init__(self, parent_frame, processor):
        self.parent = parent_frame  # frame supplied by main GUI (docked)
        self.processor = processor
        self.modifier = SegmentModification(processor)
        self.segment_controls = {}
        self.original_segments = {}
        self.current_selected_id = None

        # Build the docked UI skeleton inside parent (only once)
        self._build_docked_ui()

    def _build_docked_ui(self):
        # Clear parent
        for child in self.parent.winfo_children():
            child.destroy()

        # Title
        title = ttk.Label(self.parent, text="Segments", font=("Arial", 12, "bold"))
        title.pack(anchor="nw", pady=(6, 2), padx=6)

        # Scrollable thumbnails area
        thumbs_container = ttk.Frame(self.parent)
        thumbs_container.pack(fill="both", expand=False, padx=6, pady=(0, 6))

        # Create canvas + internal frame for thumb items
        self.thumbs_canvas = tk.Canvas(thumbs_container, height=220, bg="white", highlightthickness=0)
        self.thumbs_canvas.pack(side="left", fill="both", expand=True)
        self.thumbs_scroll = ttk.Scrollbar(thumbs_container, orient="vertical", command=self.thumbs_canvas.yview)
        self.thumbs_scroll.pack(side="right", fill="y")
        self.thumbs_canvas.configure(yscrollcommand=self.thumbs_scroll.set)

        self._thumbs_inner = ttk.Frame(self.thumbs_canvas)
        self.thumbs_canvas.create_window((0, 0), window=self._thumbs_inner, anchor='nw')
        self._thumbs_inner.bind("<Configure>", lambda e: self.thumbs_canvas.configure(scrollregion=self.thumbs_canvas.bbox("all")))

        # Preview frame (single preview for selected segment)
        preview_frame = ttk.LabelFrame(self.parent, text="Preview", padding=6)
        preview_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.preview_canvas = tk.Canvas(preview_frame, bg="white", height=260)
        self.preview_canvas.pack(fill="both", expand=True)

        # Controls frame
        self.controls_frame = ttk.LabelFrame(self.parent, text="Adjustments", padding=6)
        self.controls_frame.pack(fill="x", padx=6, pady=(0, 6))

        # Placeholder while no segments
        self.no_segments_label = ttk.Label(self._thumbs_inner, text="No segments yet. Click 'Show All Segments' after segmenting.", foreground="gray")
        self.no_segments_label.pack(padx=8, pady=8)

        # Buttons: Reset / Apply / Save / Auto-update checkbox
        self.buttons_frame = ttk.Frame(self.parent)
        self.buttons_frame.pack(fill="x", padx=6, pady=(0, 8))
        ttk.Button(self.buttons_frame, text="Reset", command=self._reset_current).pack(side="left")
        ttk.Button(self.buttons_frame, text="Apply", command=self._apply_current).pack(side="left", padx=(6,0))
        ttk.Button(self.buttons_frame, text="Save", command=self._save_current).pack(side="left", padx=(6,0))
        # Auto-update
        self.auto_update_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.buttons_frame, text="Auto Update", variable=self.auto_update_var).pack(side="right")

    def show_all_segments(self):
        """Populate thumbnail list and create controls for items (docked inline)"""
        # Load segments into modifier
        result = self.modifier.load_segments()
        if not result["success"]:
            messagebox.showerror("Error", result["error"])
            return

        segments_result = self.processor.get_all_segments()
        if not segments_result["success"]:
            messagebox.showerror("Error", segments_result["error"])
            return

        # Clear any previous thumbnails and controls
        for child in self._thumbs_inner.winfo_children():
            child.destroy()
        for child in self.controls_frame.winfo_children():
            child.destroy()
        self.no_segments_label.pack_forget()
        self.segment_controls.clear()
        self.original_segments.clear()
        self.current_selected_id = None

        # Create thumbnail entries with red separators
        for seg in segments_result["segments"]:
            seg_id = seg["id"]
            seg_image_b64 = seg["image"]
            self.original_segments[seg_id] = seg_image_b64

            # Container for each thumbnail + separator
            item_frame = ttk.Frame(self._thumbs_inner)
            item_frame.pack(fill="x", expand=True, pady=(0, 6))

            # Convert base64 to small thumbnail image
            try:
                data = base64.b64decode(seg_image_b64)
                pil = Image.open(BytesIO(data))
                pil.thumbnail((120, 80), Image.Resampling.LANCZOS)
                thumb_photo = ImageTk.PhotoImage(pil)
            except Exception as e:
                thumb_photo = None

            # Thumbnail button
            btn = ttk.Button(item_frame, text=f"Segment {seg_id+1}", compound="left",
                             command=lambda sid=seg_id: self._on_segment_selected(sid))
            if thumb_photo:
                btn.image = thumb_photo
                btn.config(image=thumb_photo)
            btn.pack(side="left", fill="x", expand=True)

            # Red separator line under each item: use a small canvas
            sep_canvas = tk.Canvas(self._thumbs_inner, height=2, bg="white", highlightthickness=0)
            sep_canvas.pack(fill="x", padx=2, pady=(2,6))
            sep_canvas.create_line(0,1,320,1, fill="red", width=2)

            # Pre-create control state for each segment
            self.segment_controls[seg_id] = {
                'brightness': tk.DoubleVar(value=0.0),
                'contrast': tk.DoubleVar(value=1.0),
                'gamma': tk.DoubleVar(value=1.0),
                'saturation': tk.DoubleVar(value=1.0),
                'blur': tk.IntVar(value=1),
                'sharpen': tk.DoubleVar(value=0.0)
            }

        # If at least one segment, select the first by default
        if segments_result["segments"]:
            first_id = segments_result["segments"][0]["id"]
            self._on_segment_selected(first_id)

    def _on_segment_selected(self, segment_id):
        """Handle user selecting a segment from the thumbnail list"""
        self.current_selected_id = segment_id
        # Build controls for the selected segment
        self._build_controls_for_segment(segment_id)
        # Immediately update the preview (original image shown first, then controls apply)
        original_b64 = self.original_segments.get(segment_id)
        if original_b64:
            self._update_preview_from_b64(original_b64)

    def _build_controls_for_segment(self, segment_id):
        """Generate controls UI inside self.controls_frame for the chosen segment"""
        # Clear previous controls
        for child in self.controls_frame.winfo_children():
            child.destroy()

        controls = self.segment_controls[segment_id]

        # Utility: create labeled scale with value label
        def make_scale(parent, label_text, var, frm, to, resolution=0.1, integer=False):
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label_text, width=12).pack(side="left")
            scale = ttk.Scale(row, from_=frm, to=to, variable=var, orient="horizontal")
            scale.pack(side="left", fill="x", expand=True, padx=(6,8))
            value_label = ttk.Label(row, text="", width=8)
            value_label.pack(side="right")
            # set label and update handler
            def _update_label(*a):
                val = var.get()
                if integer:
                    value_label.config(text=f"{int(val)}")
                else:
                    if isinstance(val, float):
                        value_label.config(text=f"{val:.2f}")
                    else:
                        value_label.config(text=str(val))
                if self.auto_update_var.get():
                    self._update_live_preview(segment_id)
            var.trace('w', _update_label)
            # Initialize label
            _update_label()
            return scale

        # Brightness
        make_scale(self.controls_frame, "Brightness:", controls['brightness'], -100, 100, resolution=1, integer=False)
        # Contrast
        make_scale(self.controls_frame, "Contrast:", controls['contrast'], 0.1, 3.0, resolution=0.01)
        # Gamma
        make_scale(self.controls_frame, "Gamma:", controls['gamma'], 0.1, 3.0, resolution=0.01)
        # Saturation
        make_scale(self.controls_frame, "Saturation:", controls['saturation'], 0.0, 3.0, resolution=0.01)
        # Blur (odd kernel)
        def _on_blur_change(*a):
            if self.auto_update_var.get():
                self._update_live_preview(segment_id)
        blur_row = ttk.Frame(self.controls_frame)
        blur_row.pack(fill="x", pady=3)
        ttk.Label(blur_row, text="Blur:", width=12).pack(side="left")
        blur_scale = ttk.Scale(blur_row, from_=1, to=31, variable=controls['blur'], orient="horizontal", command=lambda v: _on_blur_change())
        blur_scale.pack(side="left", fill="x", expand=True, padx=(6,8))
        blur_label = ttk.Label(blur_row, text=str(controls['blur'].get()), width=8)
        blur_label.pack(side="right")
        controls['blur'].trace('w', lambda *a: blur_label.config(text=str(controls['blur'].get())))

        # Sharpen
        make_scale(self.controls_frame, "Sharpen:", controls['sharpen'], 0.0, 3.0, resolution=0.01)

        # Buttons are in the bottom area (Reset / Apply / Save / Auto Update already created in parent)
        # Immediately show the modified preview if auto-update is set
        if self.auto_update_var.get():
            self._update_live_preview(segment_id)

    def _update_preview_from_b64(self, base64_image):
        """Directly draw a base64 image into the preview canvas"""
        try:
            image_data = base64.b64decode(base64_image)
            pil_image = Image.open(BytesIO(image_data))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Resize to preview canvas
            self.preview_canvas.update_idletasks()
            cw = self.preview_canvas.winfo_width()
            ch = self.preview_canvas.winfo_height()
            if cw <= 1 or ch <= 1:
                cw, ch = 300, 220
            pil_image.thumbnail((cw-20, ch-20), Image.Resampling.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(pil_image)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(cw//2, ch//2, image=self._preview_photo)
        except Exception as e:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(160, 110, text=f"Preview Error", fill="red")

    def _update_live_preview(self, segment_id):
        """Compute the adjusted image in a background thread and update the preview canvas"""
        def worker():
            try:
                controls = self.segment_controls[segment_id]
                original_img = self.modifier.get_segment_image(segment_id)  # expected RGB uint8 numpy array
                if original_img is None:
                    return

                current_img = original_img.copy().astype(np.float32)

                # brightness & contrast
                brightness = controls['brightness'].get()
                contrast = controls['contrast'].get()
                if brightness != 0.0 or contrast != 1.0:
                    # convertScaleAbs style: new = original * contrast + brightness
                    current_img = current_img * contrast + brightness
                    current_img = np.clip(current_img, 0, 255)

                # gamma
                gamma = controls['gamma'].get()
                if abs(gamma - 1.0) > 1e-6:
                    inv_gamma = 1.0 / gamma
                    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
                    # apply LUT per channel
                    current_img = cv2.LUT(current_img.astype(np.uint8), table)

                # saturation
                saturation = controls['saturation'].get()
                if abs(saturation - 1.0) > 1e-6:
                    hsv = cv2.cvtColor(current_img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[:, :, 1] = hsv[:, :, 1] * saturation
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                    current_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

                # blur
                blur_val = int(controls['blur'].get())
                if blur_val > 1:
                    if blur_val % 2 == 0:
                        blur_val += 1
                    current_img = cv2.GaussianBlur(current_img.astype(np.uint8), (blur_val, blur_val), 0)

                # sharpen
                sharpen_val = controls['sharpen'].get()
                if sharpen_val > 0:
                    kernel = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]], dtype=np.float32) * sharpen_val
                    kernel[1, 1] = 8 * sharpen_val + 1.0
                    current_img = cv2.filter2D(current_img.astype(np.float32), -1, kernel)
                    current_img = np.clip(current_img, 0, 255).astype(np.uint8)

                # ensure uint8
                if current_img.dtype != np.uint8:
                    current_img = np.clip(current_img, 0, 255).astype(np.uint8)

                # encode to PNG bytes -> base64 for update
                _, buf = cv2.imencode('.png', cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR))
                b64 = base64.b64encode(buf).decode('utf-8')
                self.parent.after(0, lambda: self._update_preview_from_b64(b64))
            except Exception as e:
                print("Preview update error:", e)

        threading.Thread(target=worker, daemon=True).start()

    def _reset_current(self):
        """Reset the controls for current selected and update preview to original (no popup)"""
        sid = self.current_selected_id
        if sid is None:
            return
        controls = self.segment_controls[sid]
        controls['brightness'].set(0.0)
        controls['contrast'].set(1.0)
        controls['gamma'].set(1.0)
        controls['saturation'].set(1.0)
        controls['blur'].set(1)
        controls['sharpen'].set(0.0)
        # instruct modifier to reset its internal state for the segment
        try:
            self.modifier.reset_segment(sid)
        except Exception:
            pass
        # Update using original image
        orig = self.original_segments.get(sid)
        if orig:
            self._update_preview_from_b64(orig)

    def _apply_current(self):
        """Apply current changes to the modifier and attempt to modify the main image"""
        sid = self.current_selected_id
        if sid is None:
            return

        def worker_apply():
            try:
                controls = self.segment_controls[sid]
                original_img = self.modifier.get_segment_image(sid)
                if original_img is None:
                    return

                # Build modified image (same logic as live preview)
                current_img = original_img.copy().astype(np.float32)

                # brightness & contrast
                brightness = controls['brightness'].get()
                contrast = controls['contrast'].get()
                if brightness != 0.0 or contrast != 1.0:
                    current_img = current_img * contrast + brightness
                    current_img = np.clip(current_img, 0, 255)

                # gamma
                gamma = controls['gamma'].get()
                if abs(gamma - 1.0) > 1e-6:
                    inv_gamma = 1.0 / gamma
                    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
                    current_img = cv2.LUT(current_img.astype(np.uint8), table)

                # saturation
                saturation = controls['saturation'].get()
                if abs(saturation - 1.0) > 1e-6:
                    hsv = cv2.cvtColor(current_img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[:, :, 1] = hsv[:, :, 1] * saturation
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                    current_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

                # blur
                blur_val = int(controls['blur'].get())
                if blur_val > 1:
                    if blur_val % 2 == 0:
                        blur_val += 1
                    current_img = cv2.GaussianBlur(current_img.astype(np.uint8), (blur_val, blur_val), 0)

                # sharpen
                sharpen_val = controls['sharpen'].get()
                if sharpen_val > 0:
                    kernel = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]], dtype=np.float32) * sharpen_val
                    kernel[1, 1] = 8 * sharpen_val + 1.0
                    current_img = cv2.filter2D(current_img.astype(np.float32), -1, kernel)
                    current_img = np.clip(current_img, 0, 255).astype(np.uint8)

                if current_img.dtype != np.uint8:
                    current_img = np.clip(current_img, 0, 255).astype(np.uint8)

                # 1) Try to give modified segment to modifier (existing behavior)
                try:
                    self.modifier.set_modified_segment(sid, current_img)
                except Exception:
                    # not fatal; continue to attempt integration into the main image
                    pass

                integrated = False

                # 2) If processor supports a dedicated API to integrate a modified segment, try it
                try:
                    if hasattr(self.processor, 'integrate_modified_segment'):
                        # expected signature: integrate_modified_segment(segment_id, rgb_uint8_numpy)
                        ok = self.processor.integrate_modified_segment(sid, current_img)
                        integrated = bool(ok)
                except Exception:
                    integrated = False

                # 3) If processor exposes a helper to replace the segment in-place, try that
                if not integrated:
                    try:
                        if hasattr(self.processor, 'replace_segment_in_image'):
                            ok = self.processor.replace_segment_in_image(sid, current_img)
                            integrated = bool(ok)
                    except Exception:
                        integrated = False

                # 4) Fallback: try to obtain a mask from processor/modifier and overlay into processor.current_image
                if not integrated:
                    try:
                        mask = None
                        # try modifier first
                        if hasattr(self.modifier, 'get_segment_mask'):
                            mask = self.modifier.get_segment_mask(sid)
                        # then processor
                        if mask is None and hasattr(self.processor, 'get_segment_mask'):
                            mask = self.processor.get_segment_mask(sid)

                        if mask is not None and isinstance(mask, np.ndarray) and self.processor.current_image is not None:
                            # Normalize mask to boolean and ensure shapes align
                            # mask expected shape: (H, W) or (H, W, 1)
                            m = mask.squeeze()
                            h_img, w_img = self.processor.current_image.shape[:2]
                            if m.shape == (h_img, w_img):
                                # If modified segment is full-size, use directly; else attempt to tile/resize cautiously
                                mod = current_img
                                # If sizes mismatch, try to resize modified image to main image
                                if mod.shape[:2] != (h_img, w_img):
                                    try:
                                        mod_rgb = cv2.resize(mod, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                                    except Exception:
                                        mod_rgb = None
                                else:
                                    mod_rgb = mod
                                if mod_rgb is not None:
                                    # Overlay: processor.current_image is BGR; current_img is RGB -> convert
                                    try:
                                        mod_bgr = cv2.cvtColor(mod_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                                    except Exception:
                                        mod_bgr = mod_rgb.astype(np.uint8)
                                    # Apply mask
                                    m_bool = (m != 0)
                                    # ensure shapes
                                    if mod_bgr.shape[:2] == self.processor.current_image.shape[:2]:
                                        base = self.processor.current_image.copy()
                                        base[m_bool] = mod_bgr[m_bool]
                                        self.processor.current_image = base
                                        integrated = True
                            # if mask not same size, skip overlay
                    except Exception:
                        integrated = False

                # 5) If integrated into processor.current_image, try to refresh main display
                if integrated:
                    try:
                        top = self.parent.winfo_toplevel()
                        # We attached the main GUI as 'app' on root in ImageSegmentationGUI.__init__
                        if hasattr(top, 'app') and hasattr(top.app, 'display_current_image'):
                            # call display_current_image on GUI instance to refresh left canvas
                            top.app.display_current_image()
                        else:
                            # As a fallback, try generating a <<Refresh>> event for other code to catch (non-fatal)
                            try:
                                top.event_generate('<<SegmentApplied>>')
                            except Exception:
                                pass
                        # Inform user
                        self.parent.after(0, lambda: messagebox.showinfo("Applied", f"Changes applied to Segment {sid+1} and merged into the main image"))
                        return
                    except Exception:
                        pass

                # If not integrated into main image, still notify the user it was saved into modifier
                self.parent.after(0, lambda: messagebox.showinfo("Applied", f"Changes applied to Segment {sid+1} (modifier updated, main image not changed)"))

            except Exception as e:
                print("Apply error:", e)
                self.parent.after(0, lambda: messagebox.showerror("Error", f"Failed to apply changes: {e}"))

        threading.Thread(target=worker_apply, daemon=True).start()

    def _save_current(self):
        """Save the currently modified (or original) segment to disk (file dialog)"""
        sid = self.current_selected_id
        if sid is None:
            messagebox.showinfo("No selection", "No segment is selected.")
            return
        try:
            modified = None
            try:
                modified = self.modifier.get_modified_segment(sid)
            except Exception:
                modified = None
            if modified is not None:
                pil_image = Image.fromarray(modified)
                filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
                                                        title=f"Save Segment {sid+1}")
                if filename:
                    pil_image.save(filename)
                    messagebox.showinfo("Saved", f"Segment saved as {filename}")
            else:
                # Save original base64
                orig_b64 = self.original_segments.get(sid)
                if orig_b64:
                    filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
                                                            title=f"Save Segment {sid+1}")
                    if filename:
                        with open(filename, 'wb') as f:
                            f.write(base64.b64decode(orig_b64))
                        messagebox.showinfo("Saved", f"Original segment saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save segment: {e}")