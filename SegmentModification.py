from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import base64
from ImageSegmentationProcessor import ImageSegmentationProcessor


class SegmentModification:
    def __init__(self, processor: ImageSegmentationProcessor):
        """
        Initialize with an ImageSegmentationProcessor instance.
        
        Args:
            processor: ImageSegmentationProcessor instance with segmented image
        """
        self.processor = processor
        self.segments_data = []
        self.modified_segments = {}
        
    def load_segments(self) -> Dict[str, Any]:
        """
        Load all segments from the processor.
        
        Returns:
            Dictionary with success status and segment count
        """
        try:
            result = self.processor.get_all_segments()
            if result["success"]:
                self.segments_data = result["segments"]
                return {
                    "success": True,
                    "num_segments": len(self.segments_data),
                    "segment_ids": [seg["id"] for seg in self.segments_data]
                }
            else:
                return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_segment_image(self, segment_id: int) -> Optional[np.ndarray]:
        """
        Get segment image as numpy array.
        
        Args:
            segment_id: ID of the segment to retrieve
            
        Returns:
            Numpy array of the segment image or None if not found
        """
        try:
            for segment in self.segments_data:
                if segment["id"] == segment_id:
                    # Decode base64 to image
                    img_data = base64.b64decode(segment["image"])
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        print(f"Failed to decode image for segment {segment_id}")
                        return None
                    
                    # Convert BGR to RGB and ensure proper data type
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    print(f"Loaded segment {segment_id}: shape={img_rgb.shape}, dtype={img_rgb.dtype}, range={img_rgb.min()}-{img_rgb.max()}")
                    
                    return img_rgb
            
            print(f"Segment {segment_id} not found in segments_data")
            return None
            
        except Exception as e:
            print(f"Error getting segment image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def adjust_brightness(self, segment_id: int, brightness_factor: float = 1.0) -> Dict[str, Any]:
        """
        Adjust brightness of a segment.
        
        Args:
            segment_id: ID of the segment to modify
            brightness_factor: Brightness multiplier (1.0 = no change, >1.0 brighter, <1.0 darker)
            
        Returns:
            Dictionary with success status and modified image
        """
        img = self.get_segment_image(segment_id)
        if img is None:
            return {"success": False, "error": "Segment not found"}
        
        try:
            # Apply brightness adjustment
            bright_img = cv2.convertScaleAbs(img, alpha=1.0, beta=brightness_factor * 50)
            
            # Store modified segment
            self.modified_segments[segment_id] = bright_img
            
            return {
                "success": True,
                "modified_image": self._image_to_base64(bright_img),
                "segment_id": segment_id,
                "modification": f"brightness_{brightness_factor}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def adjust_contrast(self, segment_id: int, contrast_factor: float = 1.0) -> Dict[str, Any]:
        """
        Adjust contrast of a segment.
        
        Args:
            segment_id: ID of the segment to modify
            contrast_factor: Contrast multiplier (1.0 = no change, >1.0 higher contrast, <1.0 lower contrast)
            
        Returns:
            Dictionary with success status and modified image
        """
        img = self.get_segment_image(segment_id)
        if img is None:
            return {"success": False, "error": "Segment not found"}
        
        try:
            # Apply contrast adjustment
            contrast_img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)
            
            # Store modified segment
            self.modified_segments[segment_id] = contrast_img
            
            return {
                "success": True,
                "modified_image": self._image_to_base64(contrast_img),
                "segment_id": segment_id,
                "modification": f"contrast_{contrast_factor}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def adjust_brightness_contrast(self, segment_id: int, brightness: float = 0.0, 
                                 contrast: float = 1.0) -> Dict[str, Any]:
        """
        Adjust both brightness and contrast of a segment.
        
        Args:
            segment_id: ID of the segment to modify
            brightness: Brightness offset (-100 to 100)
            contrast: Contrast multiplier (0.0 to 3.0, 1.0 = no change)
            
        Returns:
            Dictionary with success status and modified image
        """
        img = self.get_segment_image(segment_id)
        if img is None:
            return {"success": False, "error": "Segment not found"}
        
        try:
            # Apply brightness and contrast adjustment
            adjusted_img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
            
            # Store modified segment
            self.modified_segments[segment_id] = adjusted_img
            
            return {
                "success": True,
                "modified_image": self._image_to_base64(adjusted_img),
                "segment_id": segment_id,
                "modification": f"brightness_{brightness}_contrast_{contrast}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def adjust_gamma(self, segment_id: int, gamma: float = 1.0) -> Dict[str, Any]:
        """
        Apply gamma correction to a segment.
        
        Args:
            segment_id: ID of the segment to modify
            gamma: Gamma value (1.0 = no change, <1.0 brighter, >1.0 darker)
            
        Returns:
            Dictionary with success status and modified image
        """
        img = self.get_segment_image(segment_id)
        if img is None:
            return {"success": False, "error": "Segment not found"}
        
        try:
            # Build lookup table for gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            
            # Apply gamma correction
            gamma_img = cv2.LUT(img, table)
            
            # Store modified segment
            self.modified_segments[segment_id] = gamma_img
            
            return {
                "success": True,
                "modified_image": self._image_to_base64(gamma_img),
                "segment_id": segment_id,
                "modification": f"gamma_{gamma}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def adjust_saturation(self, segment_id: int, saturation_factor: float = 1.0) -> Dict[str, Any]:
        """
        Adjust color saturation of a segment.
        
        Args:
            segment_id: ID of the segment to modify
            saturation_factor: Saturation multiplier (1.0 = no change, >1.0 more saturated, <1.0 less saturated)
            
        Returns:
            Dictionary with success status and modified image
        """
        img = self.get_segment_image(segment_id)
        if img is None:
            return {"success": False, "error": "Segment not found"}
        
        try:
            # Convert to HSV for saturation adjustment
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Adjust saturation
            hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            # Convert back to RGB
            saturated_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # Store modified segment
            self.modified_segments[segment_id] = saturated_img
            
            return {
                "success": True,
                "modified_image": self._image_to_base64(saturated_img),
                "segment_id": segment_id,
                "modification": f"saturation_{saturation_factor}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def apply_blur(self, segment_id: int, blur_radius: int = 5) -> Dict[str, Any]:
        """
        Apply Gaussian blur to a segment.
        
        Args:
            segment_id: ID of the segment to modify
            blur_radius: Radius of the blur kernel (must be odd)
            
        Returns:
            Dictionary with success status and modified image
        """
        img = self.get_segment_image(segment_id)
        if img is None:
            return {"success": False, "error": "Segment not found"}
        
        try:
            # Ensure blur_radius is odd
            if blur_radius % 2 == 0:
                blur_radius += 1
            
            # Apply Gaussian blur
            blurred_img = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)
            
            # Store modified segment
            self.modified_segments[segment_id] = blurred_img
            
            return {
                "success": True,
                "modified_image": self._image_to_base64(blurred_img),
                "segment_id": segment_id,
                "modification": f"blur_{blur_radius}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def sharpen(self, segment_id: int, strength: float = 1.0) -> Dict[str, Any]:
        """
        Apply sharpening filter to a segment.
        
        Args:
            segment_id: ID of the segment to modify
            strength: Sharpening strength (1.0 = default, >1.0 stronger)
            
        Returns:
            Dictionary with success status and modified image
        """
        img = self.get_segment_image(segment_id)
        if img is None:
            return {"success": False, "error": "Segment not found"}
        
        try:
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) * strength
            kernel[1, 1] = 8 * strength + 1  # Adjust center value
            
            # Apply sharpening
            sharpened_img = cv2.filter2D(img, -1, kernel)
            sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)
            
            # Store modified segment
            self.modified_segments[segment_id] = sharpened_img
            
            return {
                "success": True,
                "modified_image": self._image_to_base64(sharpened_img),
                "segment_id": segment_id,
                "modification": f"sharpen_{strength}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_modified_segment(self, segment_id: int) -> Optional[np.ndarray]:
        """
        Get a modified segment if it exists.
        
        Args:
            segment_id: ID of the segment to retrieve
            
        Returns:
            Modified segment image or None if not found
        """
        return self.modified_segments.get(segment_id)
    
    def list_available_segments(self) -> List[int]:
        """
        List all available segment IDs.
        
        Returns:
            List of segment IDs
        """
        return [seg["id"] for seg in self.segments_data]
    
    def list_modified_segments(self) -> List[int]:
        """
        List all modified segment IDs.
        
        Returns:
            List of modified segment IDs
        """
        return list(self.modified_segments.keys())
    
    def reset_segment(self, segment_id: int) -> Dict[str, Any]:
        """
        Reset a segment to its original state.
        
        Args:
            segment_id: ID of the segment to reset
            
        Returns:
            Dictionary with success status
        """
        try:
            if segment_id in self.modified_segments:
                del self.modified_segments[segment_id]
                return {"success": True, "message": f"Segment {segment_id} reset to original"}
            else:
                return {"success": False, "error": "Segment not in modified list"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """
        Encode an RGB image as base64 string of PNG.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Base64 encoded PNG image
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Invalid image shape for base64 conversion: {image.shape}")
        
        _, buf = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode('utf-8')