import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import requests
import base64
from PIL import Image, ImageTk
import numpy as np
from threading import Thread
from io import BytesIO
import json
import cv2
import tempfile
import os

class APIClient:
    """Client for communicating with the Image Segmentation REST API"""
    
    def __init__(self, base_url="http://localhost:5000/api"):
        self.base_url = base_url
        self.session_id = None
        self.timeout = 30
    
    def create_session(self):
        """Create a new processing session"""
        try:
            response = requests.post(f"{self.base_url}/session", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")
                return {"success": True, "session_id": self.session_id}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_session(self):
        """Delete the current session"""
        if not self.session_id:
            return {"success": False, "error": "No active session"}
        
        try:
            response = requests.delete(f"{self.base_url}/session/{self.session_id}", timeout=self.timeout)
            if response.status_code == 200:
                self.session_id = None
                return {"success": True}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def upload_image(self, file_path):
        """Upload an image file"""
        if not self.session_id:
            return {"success": False, "error": "No active session"}
        
        try:
            with open(file_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(f"{self.base_url}/session/{self.session_id}/image", 
                                       files=files, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_current_image(self):
        """Get the current processed image"""
        if not self.session_id:
            return {"success": False, "error": "No active session"}
        
        try:
            response = requests.get(f"{self.base_url}/session/{self.session_id}/image", timeout=self.timeout)
            if response.status_code == 200:
                return {"success": True, **response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def segment_watershed(self):
        """Perform watershed segmentation"""
        if not self.session_id:
            return {"success": False, "error": "No active session"}
        
        try:
            response = requests.post(f"{self.base_url}/session/{self.session_id}/segment/watershed", timeout=60)
            if response.status_code == 200:
                data = response.json()
                return {"success": True, **data}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def segment_deeplab(self):
        """Perform DeepLab segmentation"""
        if not self.session_id:
            return {"success": False, "error": "No active session"}
        
        try:
            response = requests.post(f"{self.base_url}/session/{self.session_id}/segment/deeplab", timeout=60)
            if response.status_code == 200:
                data = response.json()
                return {"success": True, **data}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_all_segments(self):
        """Get all segments"""
        if not self.session_id:
            return {"success": False, "error": "No active session"}
        
        try:
            response = requests.get(f"{self.base_url}/session/{self.session_id}/segments", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return {"success": True, "segments": data["segments"], "count": data["count"]}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_segment(self, segment_id):
        """Get a specific segment"""
        if not self.session_id:
            return {"success": False, "error": "No active session"}
        
        try:
            response = requests.get(f"{self.base_url}/session/{self.session_id}/segment/{segment_id}", timeout=self.timeout)
            if response.status_code == 200:
                return {"success": True, **response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def modify_segment(self, segment_id, modifications):
        """Apply modifications to a segment"""
        if not self.session_id:
            return {"success": False, "error": "No active session"}
        
        try:
            response = requests.post(f"{self.base_url}/session/{self.session_id}/segment/{segment_id}/modify",
                                   json=modifications, timeout=self.timeout)
            if response.status_code == 200:
                return {"success": True, **response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def reset_segment(self, segment_id):
        """Reset a segment to original state"""
        if not self.session_id:
            return {"success": False, "error": "No active session"}
        
        try:
            response = requests.post(f"{self.base_url}/session/{self.session_id}/segment/{segment_id}/reset", timeout=self.timeout)
            if response.status_code == 200:
                return {"success": True, **response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def apply_segment(self, segment_id):
        """Apply modified segment to main image"""
        if not self.session_id:
            return {"success": False, "error": "No active session"}
        
        try:
            response = requests.post(f"{self.base_url}/session/{self.session_id}/segment/{segment_id}/apply", timeout=self.timeout)
            if response.status_code == 200:
                return {"success": True, **response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def download_segment(self, segment_id, save_path):
        """Download a segment as image file"""
        if not self.session_id:
            return {"success": False, "error": "No active session"}
        
        try:
            response = requests.get(f"{self.base_url}/session/{self.session_id}/segment/{segment_id}/download", timeout=self.timeout)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return {"success": True}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

class ImageSegmentationGUIClient:
    def __init__(self, api_base_url="http://localhost:5000/api"):
        self.root = TkinterDnD.Tk()
        self.root.title("Image Segmentation Tool (API Client)")
        self.root.geometry("1200x800")

        self.api_client = APIClient(api_base_url)
        self.current_display_image = None

        # Setup UI and enhanced display with a right-side panel
        self.setup_ui()
        self.setup_drag_drop()
        # Pass the right-side frame to the enhanced display
        self.enhanced_display = EnhancedSegmentDisplayClient(self.right_panel_frame, self.api_client)
        
        # Initialize API session
        self.initialize_session()

    def initialize_session(self):
        """Create API session on startup"""
        result = self.api_client.create_session()
        if result["success"]:
            self.status_var.set(f"Connected to API - Session: {result['session_id'][:8]}...")
        else:
            messagebox.showerror("API Error", f"Failed to connect to API: {result['error']}")
            self.status_var.set("API connection failed")

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

        # Connection status
        self.connection_var = tk.StringVar(value="Disconnected")
        ttk.Label(control_frame, textvariable=self.connection_var, foreground="red").grid(row=0, column=4, padx=(20, 0))

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
        self.status_var = tk.StringVar(value="Connecting to API...")
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
        """Load and display image via API"""
        self.status_var.set("Uploading image to API...")
        
        def upload_worker():
            result = self.api_client.upload_image(file_path)
            self.root.after(0, self._on_image_upload_complete, result)
        
        thread = Thread(target=upload_worker, daemon=True)
        thread.start()

    def _on_image_upload_complete(self, result):
        """Handle image upload completion"""
        if result.get("success"):
            self.display_current_image()
            self.status_var.set(f"Image loaded: {result.get('width')}x{result.get('height')} pixels")
            self.connection_var.set("Connected")
            self.connection_var.config(foreground="green")
            try:
                self.drop_label.destroy()
            except:
                pass
        else:
            messagebox.showerror("Error", f"Failed to load image: {result.get('error', 'Unknown error')}")
            self.status_var.set("Ready")

    def display_current_image(self):
        """Display the current image from API"""
        def fetch_worker():
            result = self.api_client.get_current_image()
            self.root.after(0, self._on_image_fetch_complete, result)
        
        thread = Thread(target=fetch_worker, daemon=True)
        thread.start()

    def _on_image_fetch_complete(self, result):
        """Handle image fetch completion"""
        if result.get("success") and result.get("image"):
            self._display_base64_image(result["image"])

    def _display_base64_image(self, base64_string):
        """Display base64 encoded image on the main canvas"""
        try:
            image_data = base64.b64decode(base64_string)
            pil_image = Image.open(BytesIO(image_data))

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
        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to display image: {e}")

    def segment_watershed(self):
        """Perform watershed segmentation via API"""
        self.status_var.set("Performing watershed segmentation...")
        
        def segment_worker():
            result = self.api_client.segment_watershed()
            self.root.after(0, self._on_segmentation_complete, result, "Watershed")
        
        thread = Thread(target=segment_worker, daemon=True)
        thread.start()

    def segment_nn(self):
        """Perform NN segmentation via API"""
        self.status_var.set("Performing NN segmentation...")
        
        def segment_worker():
            result = self.api_client.segment_deeplab()
            self.root.after(0, self._on_segmentation_complete, result, "NN")
        
        thread = Thread(target=segment_worker, daemon=True)
        thread.start()

    def _on_segmentation_complete(self, result, method):
        """Handle segmentation completion"""
        if result.get("success"):
            # Display segmented image if provided
            if result.get("segmented_image"):
                self._display_base64_image(result["segmented_image"])
            num_segments = result.get("num_segments", "unknown")
            self.status_var.set(f"{method} segmentation complete - {num_segments} segments found")
        else:
            messagebox.showerror("Error", f"Segmentation failed: {result.get('error', 'Unknown error')}")
            self.status_var.set("Ready")

    def show_all_segments(self):
        """Ask enhanced display to populate the docked panel"""
        self.enhanced_display.show_all_segments()

    def on_closing(self):
        """Clean up when closing application"""
        if self.api_client.session_id:
            self.api_client.delete_session()
        self.root.destroy()

    def run(self):
        """Start the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


class EnhancedSegmentDisplayClient:
    """
    Client version of the enhanced segment display that communicates with the REST API
    """

    def __init__(self, parent_frame, api_client):
        self.parent = parent_frame
        self.api_client = api_client
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
        """Populate thumbnail list via API"""
        def fetch_worker():
            result = self.api_client.get_all_segments()
            self.parent.after(0, self._on_segments_fetched, result)
        
        thread = Thread(target=fetch_worker, daemon=True)
        thread.start()

    def _on_segments_fetched(self, result):
        """Handle segments fetch completion"""
        if not result.get("success"):
            messagebox.showerror("Error", result.get("error", "Failed to fetch segments"))
            return

        segments = result.get("segments", [])
        if not segments:
            messagebox.showinfo("No Segments", "No segments found. Please run segmentation first.")
            return

        # Clear any previous thumbnails and controls
        for child in self._thumbs_inner.winfo_children():
            child.destroy()
        for child in self.controls_frame.winfo_children():
            child.destroy()
        try:
            self.no_segments_label.pack_forget()
        except:
            pass
        self.segment_controls.clear()
        self.original_segments.clear()
        self.current_selected_id = None

        # Create thumbnail entries with red separators
        for seg in segments:
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
        if segments:
            first_id = segments[0]["id"]
            self._on_segment_selected(first_id)

    def _on_segment_selected(self, segment_id):
        """Handle user selecting a segment from the thumbnail list"""
        self.current_selected_id = segment_id
        # Build controls for the selected segment
        self._build_controls_for_segment(segment_id)
        # Show original image in preview
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
        """Update preview with modifications via API call"""
        if not self.auto_update_var.get():
            return
        
        def modify_worker():
            try:
                controls = self.segment_controls[segment_id]
                
                # Build modifications dictionary
                modifications = {
                    'brightness': controls['brightness'].get(),
                    'contrast': controls['contrast'].get(),
                    'gamma': controls['gamma'].get(),
                    'saturation': controls['saturation'].get(),
                    'blur': int(controls['blur'].get()),
                    'sharpen': controls['sharpen'].get()
                }
                
                # Call API to get modified segment
                result = self.api_client.modify_segment(segment_id, modifications)
                if result.get("success") and result.get("modified_image"):
                    self.parent.after(0, lambda: self._update_preview_from_b64(result["modified_image"]))
                
            except Exception as e:
                print("Preview update error:", e)
        
        Thread(target=modify_worker, daemon=True).start()

    def _reset_current(self):
        """Reset the controls for current selected segment"""
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
        
        # Reset on server side
        def reset_worker():
            result = self.api_client.reset_segment(sid)
            if not result.get("success"):
                self.parent.after(0, lambda: messagebox.showerror("Error", f"Failed to reset: {result.get('error')}"))
        
        Thread(target=reset_worker, daemon=True).start()
        
        # Update using original image
        orig = self.original_segments.get(sid)
        if orig:
            self._update_preview_from_b64(orig)

    def _apply_current(self):
        """Apply current changes to the main image via API"""
        sid = self.current_selected_id
        if sid is None:
            return

        def apply_worker():
            try:
                controls = self.segment_controls[sid]
                
                # Build modifications dictionary
                modifications = {
                    'brightness': controls['brightness'].get(),
                    'contrast': controls['contrast'].get(),
                    'gamma': controls['gamma'].get(),
                    'saturation': controls['saturation'].get(),
                    'blur': int(controls['blur'].get()),
                    'sharpen': controls['sharpen'].get()
                }
                
                # First modify the segment
                modify_result = self.api_client.modify_segment(sid, modifications)
                if not modify_result.get("success"):
                    self.parent.after(0, lambda: messagebox.showerror("Error", f"Failed to modify segment: {modify_result.get('error')}"))
                    return
                
                # Then apply it to the main image
                apply_result = self.api_client.apply_segment(sid)
                if apply_result.get("success"):
                    # Refresh main display
                    top = self.parent.winfo_toplevel()
                    if hasattr(top, 'master') and hasattr(top.master, 'display_current_image'):
                        top.master.display_current_image()
                    
                    self.parent.after(0, lambda: messagebox.showinfo("Applied", f"Changes applied to Segment {sid+1} and merged into the main image"))
                else:
                    self.parent.after(0, lambda: messagebox.showerror("Error", f"Failed to apply to main image: {apply_result.get('error')}"))

            except Exception as e:
                print("Apply error:", e)
                self.parent.after(0, lambda: messagebox.showerror("Error", f"Failed to apply changes: {e}"))

        Thread(target=apply_worker, daemon=True).start()

    def _save_current(self):
        """Save the currently modified segment to disk via API download"""
        sid = self.current_selected_id
        if sid is None:
            messagebox.showinfo("No selection", "No segment is selected.")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
                title=f"Save Segment {sid+1}"
            )
            
            if filename:
                def download_worker():
                    # First apply current modifications
                    controls = self.segment_controls[sid]
                    modifications = {
                        'brightness': controls['brightness'].get(),
                        'contrast': controls['contrast'].get(),
                        'gamma': controls['gamma'].get(),
                        'saturation': controls['saturation'].get(),
                        'blur': int(controls['blur'].get()),
                        'sharpen': controls['sharpen'].get()
                    }
                    
                    # Modify segment on server (this updates the server's copy)
                    modify_result = self.api_client.modify_segment(sid, modifications)
                    
                    # Download the segment
                    download_result = self.api_client.download_segment(sid, filename)
                    
                    if download_result.get("success"):
                        self.parent.after(0, lambda: messagebox.showinfo("Saved", f"Segment saved as {filename}"))
                    else:
                        self.parent.after(0, lambda: messagebox.showerror("Error", f"Failed to save: {download_result.get('error')}"))
                
                Thread(target=download_worker, daemon=True).start()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save segment: {e}")


# Example usage and startup script
if __name__ == '__main__':
    import sys
    import subprocess
    import time
    
    def check_api_server(url="http://localhost:5000/api/health"):
        """Check if API server is running"""
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_api_server():
        """Start the API server in background"""
        try:
            # This assumes the API server code is in a file called 'segmentation_api.py'
            # You may need to adjust this path
            subprocess.Popen([sys.executable, 'segmentation_api.py'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            return True
        except:
            return False
    
    # Check if API server is running
    if not check_api_server():
        print("API server not detected. Attempting to start...")
        if start_api_server():
            print("Waiting for API server to start...")
            time.sleep(3)  # Wait for server to start
            if check_api_server():
                print("API server started successfully!")
            else:
                print("Failed to start API server. Please start it manually.")
                sys.exit(1)
        else:
            print("Could not start API server. Please ensure 'segmentation_api.py' is available and start it manually.")
            sys.exit(1)
    else:
        print("API server detected. Starting GUI client...")
    
    # Start the GUI client
    app = ImageSegmentationGUIClient()
    app.run()