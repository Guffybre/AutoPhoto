
from typing import Any, Dict
import cv2
import numpy as np
from ImageSegmenter import ImageSegmenter
import base64

class ImageSegmentationProcessor:
    def __init__(self):
        self.current_image = None
        self.segments = []
        self.segment_masks = []
        self.segmenter = ImageSegmenter()

    def load_image(self, image_path: str) -> Dict[str, Any]:
        """Load image and return basic info"""
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                return {"success": False, "error": "Could not load image"}
            
            height, width = self.current_image.shape[:2]
            return {
                "success": True,
                "width": width,
                "height": height,
                "channels": self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def load_image_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Load image from bytes (for future API integration)"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            self.current_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if self.current_image is None:
                return {"success": False, "error": "Could not decode image"}
            
            height, width = self.current_image.shape[:2]
            return {
                "success": True,
                "width": width,
                "height": height,
                "channels": self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def segment_deeplab(self) -> Dict[str, Any]:
        if self.current_image is None:
            return {"success": False, "error": "No image loaded"}
        self.last_result = self.segmenter.segment_oneformer(self.current_image)
        self.segments = self.last_result["segments"]
        return self.last_result
    
    def segment_watershed(self) -> Dict[str, Any]:
        """Perform watershed segmentation"""
        if self.current_image is None:
            return {"success": False, "error": "No image loaded"}
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            
            # Unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Apply watershed
            markers = cv2.watershed(self.current_image, markers)
            
            # Create segmented image
            segmented_image = self.current_image.copy()
            segmented_image[markers == -1] = [255, 0, 0]  # Mark boundaries in red
            
            # Create individual segments
            unique_markers = np.unique(markers)
            self.segments = []
            self.segment_masks = []
            
            for marker in unique_markers:
                if marker > 1:  # Skip background (0) and boundary (-1)
                    mask = (markers == marker)
                    self.segment_masks.append(mask)
                    
                    segment = np.zeros_like(self.current_image)
                    segment[mask] = self.current_image[mask]
                    self.segments.append(segment)
            
            return {
                "success": True,
                "num_segments": len(self.segments),
                "segmented_image": self._image_to_base64(segmented_image)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_segment(self, segment_id: int) -> Dict[str, Any]:
        """Get a specific segment"""
        if segment_id >= len(self.segments):
            return {"success": False, "error": "Invalid segment ID"}
        
        return {
            "success": True,
            "segment_image": self._image_to_base64(self.segments[segment_id]),
            "segment_id": segment_id
        }
    
    def get_all_segments(self) -> Dict[str, Any]:
        """
        Return all segments as base64 images.
        """
        results = []
        for i, segment in enumerate(self.segmenter.segments):
            if isinstance(segment, np.ndarray) and segment.size > 0:
                try:
                    encoded = self._image_to_base64(segment)
                    results.append({"id": i, "image": encoded})
                except Exception as e:
                    print(f"Encoding failed for segment {i}: {e}")
            else:
                print(f"Invalid segment at index {i}, skipping.")

        return {
            "success": True,
            "segments": results
        }
        
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Encode an RGB image as base64 string of PNG."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Invalid image shape for base64 conversion: {image.shape}")
        
        _, buf = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode('utf-8')