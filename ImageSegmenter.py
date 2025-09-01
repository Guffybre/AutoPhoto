import base64
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from typing import Dict, Any
from PIL import Image
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

class ImageSegmenter:
    def __init__(self, device: str = None):
        """
        Initialize the segmentation model.

        Args:
            device (str, optional): Device to run inference on ('cpu' or 'cuda').
                                     Defaults to automatically selecting CUDA if available.
        """
        try:
            print(torch.cuda.get_device_name(0))
        except Exception:
            print("CUDA device not available.")

        preferred = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            self.device = torch.device(preferred)
            # Sanity check
            if self.device.type == 'cuda':
                torch.zeros(1).to(self.device)
        except Exception as e:
            print(f"Warning: could not use CUDA ({e}), falling back to CPU.")
            self.device = torch.device('cpu')

        # Load pretrained DeepLabV3 model
        
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.model = deeplabv3_resnet101(weights=weights)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.segments: list = []
        self.segment_masks: list = []

        # --- OneFormer setup ---
        self.oneformer_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        self.oneformer = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_tiny"
        ).to(self.device).eval()
        

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Encode an RGB image as base64 string of PNG."""
        _, buf = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode('utf-8')

    def segment_kmeans(self, current_image: np.ndarray, k: int = 5) -> Dict[str, Any]:
        """Perform K-means segmentation."""
        if current_image is None:
            return {"success": False, "error": "No image loaded"}

        try:
            h, w, _ = current_image.shape
            data = current_image.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented = centers[labels.flatten()].reshape((h, w, 3))

            self.segments, self.segment_masks = [], []
            for i in range(k):
                mask = (labels.flatten() == i).reshape((h, w))
                self.segment_masks.append(mask)
                seg = np.zeros_like(current_image)
                seg[mask] = centers[i]
                self.segments.append(seg)

            return {
                "success": True,
                "num_segments": k,
                "segmented_image": self._image_to_base64(segmented)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def segment_deeplab(self, current_image: np.ndarray) -> Dict[str, Any]:
        """
        Perform semantic segmentation using DeepLabV3, drawing red boundaries and extracting segments.
        """
        if current_image is None:
            return {"success": False, "error": "No image loaded"}

        try:
            orig_h, orig_w, _ = current_image.shape

            # Preprocess and infer
            input_t = self.transform(current_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(input_t)['out'][0]
            mask = torch.argmax(out, dim=0).cpu().numpy().astype(np.uint8)

            # Resize mask to match original image size
            mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # Generate segmented image with red boundaries
            segmented_image = current_image.copy()
            boundary_mask = np.zeros_like(mask_resized, dtype=np.uint8)

            # Detect boundaries between labels
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(mask_resized, kernel, iterations=1)
            eroded = cv2.erode(mask_resized, kernel, iterations=1)
            boundary_mask = (dilated != eroded)

            # Draw red on boundaries
            segmented_image[boundary_mask] = [255, 0, 0]  # Red in RGB

            # Extract individual segments
            unique_labels = np.unique(mask_resized)
            self.segments = []
            self.segment_masks = []
            base64_segments = []

            for label in unique_labels:
                segment_mask = (mask_resized == label)
                self.segment_masks.append(segment_mask)

                segment = np.zeros_like(current_image)
                segment[segment_mask] = current_image[segment_mask]
                self.segments.append(segment)

                base64_image = self._image_to_base64(segment)
                base64_segments.append({"id": int(label), "image": base64_image})

            return {
                "success": True,
                "num_segments": int(len(self.segments)),
                "segmented_image": self._image_to_base64(segmented_image),
                "segments": base64_segments
            }

        except Exception as e:
            return {"success": False, "error": str(e)}



    def segment_oneformer(
        self,
        current_image: np.ndarray,
        task: str = "semantic"  # one of: "semantic", "instance", "panoptic"
    ) -> Dict[str, Any]:
        """Universal segmentation via OneFormer, red boundaries + per-mask segments."""
        if current_image is None:
            return {"success": False, "error": "No image loaded"}
        try:
            # to PIL RGB
            if not isinstance(current_image, Image.Image):
                pil = Image.fromarray(current_image[:, :, ::-1])
            else:
                pil = current_image
            H, W = pil.height, pil.width

            # predict
            inputs = self.oneformer_processor(pil, [task], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.oneformer(**inputs)

            # post‑process
            if task == "semantic":
                mask = self.oneformer_processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(H, W)]
                )[0].cpu().numpy().astype(np.uint8)
            elif task == "instance":
                out = self.oneformer_processor.post_process_instance_segmentation(
                    outputs, target_sizes=[(H, W)]
                )[0]
                mask = out["segmentation"].cpu().numpy().astype(np.uint8)
            elif task == "panoptic":
                out = self.oneformer_processor.post_process_panoptic_segmentation(
                    outputs, target_sizes=[(H, W)]
                )[0]
                mask = out["segmentation"].cpu().numpy().astype(np.uint8)
            else:
                return {"success": False, "error": f"Unknown task '{task}'"}

            # draw boundaries on original (RGB)
            rgb = np.array(pil)  # RGB
            seg_img = rgb.copy()
            kern = np.ones((3,3), np.uint8)
            dil = cv2.dilate(mask, kern, iterations=1)
            ero = cv2.erode(mask, kern, iterations=1)
            bmask = (dil != ero)
            seg_img[bmask] = [255,0,0]

            # extract segments
            labels = np.unique(mask)
            self.segments, self.segment_masks = [], []
            base64_segments = []
            for lbl in labels:
                m = (mask == lbl)
                self.segment_masks.append(m)
                seg = np.zeros_like(rgb)
                seg[m] = rgb[m]
                self.segments.append(seg)
                base64_segments.append({"id": int(lbl), "image": self._image_to_base64(seg)})

            return {
                "success": True,
                "num_segments": len(self.segments),
                "segmented_image": self._image_to_base64(seg_img),
                "segments": base64_segments
            }
        except Exception as e:
            return {"success": False, "error": str(e)}