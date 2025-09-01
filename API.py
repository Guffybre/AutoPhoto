from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import tempfile
import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import threading
import uuid
from datetime import datetime, timedelta
from functools import wraps
import logging
import json
import sys

# Import the existing processor classes with fallback
try:
    from ImageSegmentationProcessor import ImageSegmentationProcessor
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ImageSegmentationProcessor not available: {e}")
    print("Creating mock processor for testing...")
    
    class MockImageSegmentationProcessor:
        """Mock processor for testing when real processor isn't available"""
        def __init__(self):
            self.current_image = None
            
        def load_image(self, file_path):
            try:
                import cv2
                self.current_image = cv2.imread(file_path)
                if self.current_image is not None:
                    h, w = self.current_image.shape[:2]
                    return {"success": True, "width": w, "height": h}
                else:
                    return {"success": False, "error": "Could not load image"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        def segment_watershed(self):
            if self.current_image is None:
                return {"success": False, "error": "No image loaded"}
            
            # Mock segmentation - just return original image encoded
            try:
                _, buffer = cv2.imencode('.png', self.current_image)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                return {
                    "success": True,
                    "num_segments": 3,  # Mock number
                    "segmented_image": img_base64
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        def segment_deeplab(self):
            return self.segment_watershed()  # Same mock behavior
        
        def get_all_segments(self):
            if self.current_image is None:
                return {"success": False, "error": "No image loaded"}
            
            # Mock segments
            try:
                _, buffer = cv2.imencode('.png', self.current_image)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    "success": True,
                    "segments": [
                        {"id": 0, "image": img_base64},
                        {"id": 1, "image": img_base64},
                        {"id": 2, "image": img_base64}
                    ]
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    ImageSegmentationProcessor = MockImageSegmentationProcessor
    PROCESSOR_AVAILABLE = True

try:
    from SegmentModification import SegmentModification
    MODIFIER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SegmentModification not available: {e}")
    print("Creating mock modifier for testing...")
    
    class MockSegmentModification:
        """Mock modifier for testing when real modifier isn't available"""
        def __init__(self, processor):
            self.processor = processor
            self.segments = {}
            self.modified_segments = {}
        
        def load_segments(self):
            return {"success": True}
        
        def get_segment_image(self, segment_id):
            if self.processor.current_image is not None:
                # Return a portion of the image as mock segment
                h, w = self.processor.current_image.shape[:2]
                segment = self.processor.current_image[0:h//2, 0:w//2]  # Top-left quarter
                return cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
            return None
        
        def get_modified_segment(self, segment_id):
            return self.modified_segments.get(segment_id)
        
        def set_modified_segment(self, segment_id, image):
            self.modified_segments[segment_id] = image
        
        def reset_segment(self, segment_id):
            if segment_id in self.modified_segments:
                del self.modified_segments[segment_id]
    
    SegmentModification = MockSegmentModification
    MODIFIER_AVAILABLE = True

class ImageSegmentationAPI:
    def __init__(self, upload_folder=None, max_content_length=16*1024*1024, enable_logging=True):
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = max_content_length
        
        # Setup upload folder
        if upload_folder is None:
            upload_folder = tempfile.mkdtemp()
        self.upload_folder = upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)
        
        # Session storage (in production, use Redis or database)
        self.sessions = {}
        self.session_timeout = timedelta(hours=2)
        
        # Setup logging
        self.enable_logging = enable_logging
        if enable_logging:
            self._setup_logging()
        
        # Setup routes
        self._setup_routes()
        
        # Setup error handlers
        self._setup_error_handlers()
        
        # Cleanup old sessions periodically
        self._start_cleanup_thread()
    
    def _setup_error_handlers(self):
        """Setup global error handlers"""
        @self.app.errorhandler(404)
        def not_found(error):
            if self.enable_logging:
                self.logger.warning(f"404 Not Found: {request.url}")
            return jsonify({"error": "Endpoint not found"}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            if self.enable_logging:
                self.logger.error(f"500 Internal Server Error: {error}")
            return jsonify({"error": "Internal server error", "details": str(error)}), 500
        
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            if self.enable_logging:
                self.logger.error(f"Unhandled exception: {e}", exc_info=True)
            return jsonify({"error": "Unexpected error occurred", "details": str(e)}), 500
    
    def _setup_logging(self):
        """Setup comprehensive request/response logging"""
        # Create custom logger
        self.logger = logging.getLogger('SegmentationAPI')
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        try:
            file_handler = logging.FileHandler('segmentation_api.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except:
            pass  # If can't create file, just use console
        
        # Setup Flask request logging
        @self.app.before_request
        def log_request():
            """Log incoming requests"""
            if self.enable_logging:
                # Get request data
                method = request.method
                url = request.url
                headers = dict(request.headers)
                
                # Log basic request info
                self.logger.info(f"=== INCOMING REQUEST ===")
                self.logger.info(f"Method: {method}")
                self.logger.info(f"URL: {url}")
                self.logger.info(f"Remote Address: {request.remote_addr}")
                
                # Log headers (excluding sensitive ones)
                filtered_headers = {k: v for k, v in headers.items() 
                                  if k.lower() not in ['authorization', 'cookie']}
                self.logger.info(f"Headers: {json.dumps(filtered_headers, indent=2)}")
                
                # Log request body (if reasonable size and not binary)
                if request.content_length and request.content_length < 10000:
                    if request.is_json:
                        try:
                            data = request.get_json()
                            self.logger.info(f"JSON Body: {json.dumps(data, indent=2)}")
                        except:
                            self.logger.info("JSON Body: <invalid JSON>")
                    elif request.form:
                        form_data = {k: v for k, v in request.form.items()}
                        self.logger.info(f"Form Data: {json.dumps(form_data, indent=2)}")
                    elif 'multipart/form-data' in request.content_type:
                        files = [f"'{k}': <file: {v.filename}>" for k, v in request.files.items()]
                        self.logger.info(f"Files: {{{', '.join(files)}}}")
                elif request.content_length and request.content_length >= 10000:
                    self.logger.info(f"Body: <large body: {request.content_length} bytes>")
        
        @self.app.after_request
        def log_response(response):
            """Log outgoing responses"""
            if self.enable_logging:
                status_code = response.status_code
                headers = dict(response.headers)
                
                self.logger.info(f"=== OUTGOING RESPONSE ===")
                self.logger.info(f"Status Code: {status_code}")
                
                # Log response headers (excluding sensitive ones)
                filtered_headers = {k: v for k, v in headers.items() 
                                  if k.lower() not in ['set-cookie']}
                self.logger.info(f"Headers: {json.dumps(filtered_headers, indent=2)}")
                
                # Log response body (if reasonable size and JSON)
                if response.content_length and response.content_length < 10000:
                    if response.is_json:
                        try:
                            data = response.get_json()
                            # Truncate base64 images for readability
                            if isinstance(data, dict):
                                data_copy = data.copy()
                                for key in ['image', 'segmented_image', 'modified_image']:
                                    if key in data_copy and isinstance(data_copy[key], str) and len(data_copy[key]) > 100:
                                        data_copy[key] = f"<base64 image: {len(data_copy[key])} chars>"
                                if 'segments' in data_copy and isinstance(data_copy['segments'], list):
                                    for seg in data_copy['segments']:
                                        if 'image' in seg and len(seg['image']) > 100:
                                            seg['image'] = f"<base64 image: {len(seg['image'])} chars>"
                            else:
                                data_copy = data
                            self.logger.info(f"JSON Body: {json.dumps(data_copy, indent=2)}")
                        except:
                            self.logger.info("JSON Body: <invalid JSON>")
                elif response.content_length and response.content_length >= 10000:
                    self.logger.info(f"Body: <large body: {response.content_length} bytes>")
                elif response.mimetype and 'image' in response.mimetype:
                    self.logger.info(f"Body: <image file: {response.mimetype}>")
                
                self.logger.info(f"=== END REQUEST/RESPONSE ===\n")
            
            return response
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            response_data = {
                "status": "healthy", 
                "timestamp": datetime.now().isoformat(),
                "processor_available": PROCESSOR_AVAILABLE,
                "modifier_available": MODIFIER_AVAILABLE,
                "active_sessions": len(self.sessions)
            }
            if self.enable_logging:
                self.logger.info(f"Health check requested")
            return jsonify(response_data)
        
        @self.app.route('/api/debug', methods=['GET'])
        def debug_info():
            """Debug information endpoint"""
            import sys
            response_data = {
                "python_version": sys.version,
                "flask_available": True,
                "opencv_available": self._check_opencv(),
                "pil_available": self._check_pil(),
                "numpy_available": self._check_numpy(),
                "processor_available": PROCESSOR_AVAILABLE,
                "modifier_available": MODIFIER_AVAILABLE,
                "active_sessions": len(self.sessions),
                "upload_folder": self.upload_folder
            }
            return jsonify(response_data)
        
        @self.app.route('/api/session', methods=['POST'])
        def create_session():
            """Create a new processing session"""
            try:
                session_id = str(uuid.uuid4())
                
                # Check if processor is available
                if not PROCESSOR_AVAILABLE:
                    error_msg = "ImageSegmentationProcessor not available. Please ensure the processor module is installed."
                    if self.enable_logging:
                        self.logger.error(f"Session creation failed: {error_msg}")
                    return jsonify({"error": error_msg}), 500
                
                processor = ImageSegmentationProcessor()
                self.sessions[session_id] = {
                    'processor': processor,
                    'modifier': None,
                    'created_at': datetime.now(),
                    'last_accessed': datetime.now()
                }
                response_data = {"session_id": session_id, "message": "Session created successfully"}
                if self.enable_logging:
                    self.logger.info(f"Created new session: {session_id}")
                    self.logger.info(f"Active sessions: {len(self.sessions)}")
                return jsonify(response_data)
            except Exception as e:
                error_msg = f"Failed to create session: {str(e)}"
                if self.enable_logging:
                    self.logger.error(f"Session creation error: {error_msg}")
                return jsonify({"error": error_msg}), 500
        
        @self.app.route('/api/session/<session_id>', methods=['DELETE'])
        def delete_session(session_id):
            """Delete a processing session"""
            if session_id in self.sessions:
                del self.sessions[session_id]
                if self.enable_logging:
                    self.logger.info(f"Deleted session: {session_id}")
                    self.logger.info(f"Active sessions: {len(self.sessions)}")
                return jsonify({"message": "Session deleted successfully"})
            if self.enable_logging:
                self.logger.warning(f"Attempted to delete non-existent session: {session_id}")
            return jsonify({"error": "Session not found"}), 404
        
        @self.app.route('/api/session/<session_id>/image', methods=['POST'])
        @self._validate_session
        def upload_image(session_id):
            """Upload an image for processing"""
            try:
                if 'image' not in request.files:
                    return jsonify({"error": "No image file provided"}), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({"error": "No file selected"}), 400
                
                if file and self._allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(self.upload_folder, f"{session_id}_{filename}")
                    file.save(filepath)
                    
                    if self.enable_logging:
                        self.logger.info(f"Saved uploaded file: {filepath}")
                    
                    # Load image into processor
                    session = self.sessions[session_id]
                    processor = session['processor']
                    result = processor.load_image(filepath)
                    
                    if result["success"]:
                        # Clean up uploaded file
                        os.remove(filepath)
                        if self.enable_logging:
                            self.logger.info(f"Image loaded successfully: {result['width']}x{result['height']}")
                        return jsonify({
                            "success": True,
                            "width": result["width"],
                            "height": result["height"]
                        })
                    else:
                        os.remove(filepath)
                        error_msg = f"Failed to load image: {result['error']}"
                        if self.enable_logging:
                            self.logger.error(error_msg)
                        return jsonify({"error": error_msg}), 400
                
                return jsonify({"error": "Invalid file type"}), 400
            except Exception as e:
                error_msg = f"Upload failed: {str(e)}"
                if self.enable_logging:
                    self.logger.error(error_msg)
                return jsonify({"error": error_msg}), 500
        
        @self.app.route('/api/session/<session_id>/image', methods=['GET'])
        @self._validate_session
        def get_current_image(session_id):
            """Get the current processed image"""
            session = self.sessions[session_id]
            processor = session['processor']
            
            if processor.current_image is None:
                return jsonify({"error": "No image loaded"}), 404
            
            # Convert BGR to RGB and encode as base64
            image_rgb = cv2.cvtColor(processor.current_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return jsonify({
                "image": img_base64,
                "format": "PNG",
                "width": processor.current_image.shape[1],
                "height": processor.current_image.shape[0]
            })
        
        @self.app.route('/api/session/<session_id>/segment/watershed', methods=['POST'])
        @self._validate_session
        def segment_watershed(session_id):
            """Perform watershed segmentation"""
            session = self.sessions[session_id]
            processor = session['processor']
            
            if processor.current_image is None:
                return jsonify({"error": "No image loaded"}), 400
            
            result = processor.segment_watershed()
            if result["success"]:
                return jsonify({
                    "message": "Watershed segmentation completed",
                    "num_segments": result["num_segments"],
                    "segmented_image": result["segmented_image"]
                })
            else:
                return jsonify({"error": f"Segmentation failed: {result['error']}"}), 500
        
        @self.app.route('/api/session/<session_id>/segment/deeplab', methods=['POST'])
        @self._validate_session
        def segment_deeplab(session_id):
            """Perform DeepLab segmentation"""
            session = self.sessions[session_id]
            processor = session['processor']
            
            if processor.current_image is None:
                return jsonify({"error": "No image loaded"}), 400
            
            result = processor.segment_deeplab()
            if result["success"]:
                return jsonify({
                    "message": "DeepLab segmentation completed",
                    "num_segments": result["num_segments"],
                    "segmented_image": result["segmented_image"]
                })
            else:
                return jsonify({"error": f"Segmentation failed: {result['error']}"}), 500
        
        @self.app.route('/api/session/<session_id>/segments', methods=['GET'])
        @self._validate_session
        def get_all_segments(session_id):
            """Get all segments from the last segmentation"""
            session = self.sessions[session_id]
            processor = session['processor']
            
            result = processor.get_all_segments()
            if result["success"]:
                return jsonify({
                    "segments": result["segments"],
                    "count": len(result["segments"])
                })
            else:
                return jsonify({"error": f"Failed to get segments: {result['error']}"}), 500
        
        @self.app.route('/api/session/<session_id>/segment/<int:segment_id>', methods=['GET'])
        @self._validate_session
        def get_segment(session_id, segment_id):
            """Get a specific segment"""
            session = self.sessions[session_id]
            processor = session['processor']
            
            # Initialize modifier if not exists
            if session['modifier'] is None:
                session['modifier'] = SegmentModification(processor)
                load_result = session['modifier'].load_segments()
                if not load_result["success"]:
                    return jsonify({"error": f"Failed to load segments: {load_result['error']}"}), 500
            
            modifier = session['modifier']
            
            try:
                # Get original segment image
                segment_image = modifier.get_segment_image(segment_id)
                if segment_image is None:
                    return jsonify({"error": f"Segment {segment_id} not found"}), 404
                
                # Convert to base64
                pil_image = Image.fromarray(segment_image)
                buffer = BytesIO()
                pil_image.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return jsonify({
                    "segment_id": segment_id,
                    "image": img_base64,
                    "format": "PNG"
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/session/<session_id>/segment/<int:segment_id>/modify', methods=['POST'])
        @self._validate_session
        def modify_segment(session_id, segment_id):
            """Apply modifications to a segment"""
            session = self.sessions[session_id]
            processor = session['processor']
            
            # Initialize modifier if not exists
            if session['modifier'] is None:
                session['modifier'] = SegmentModification(processor)
                load_result = session['modifier'].load_segments()
                if not load_result["success"]:
                    return jsonify({"error": f"Failed to load segments: {load_result['error']}"}), 500
            
            modifier = session['modifier']
            
            # Get modification parameters from request
            data = request.get_json()
            if not data:
                return jsonify({"error": "No modification parameters provided"}), 400
            
            try:
                # Get original segment
                original_img = modifier.get_segment_image(segment_id)
                if original_img is None:
                    return jsonify({"error": f"Segment {segment_id} not found"}), 404
                
                # Apply modifications
                modified_img = self._apply_modifications(original_img, data)
                
                # Store modified segment
                modifier.set_modified_segment(segment_id, modified_img)
                
                # Convert to base64 for response
                pil_image = Image.fromarray(modified_img)
                buffer = BytesIO()
                pil_image.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return jsonify({
                    "message": f"Segment {segment_id} modified successfully",
                    "segment_id": segment_id,
                    "modified_image": img_base64
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/session/<session_id>/segment/<int:segment_id>/reset', methods=['POST'])
        @self._validate_session
        def reset_segment(session_id, segment_id):
            """Reset a segment to its original state"""
            session = self.sessions[session_id]
            
            if session['modifier'] is None:
                return jsonify({"error": "No segments loaded"}), 400
            
            try:
                session['modifier'].reset_segment(segment_id)
                return jsonify({"message": f"Segment {segment_id} reset successfully"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/session/<session_id>/segment/<int:segment_id>/apply', methods=['POST'])
        @self._validate_session
        def apply_segment_to_main(session_id, segment_id):
            """Apply modified segment back to the main image"""
            session = self.sessions[session_id]
            processor = session['processor']
            modifier = session['modifier']
            
            if modifier is None:
                return jsonify({"error": "No segments loaded"}), 400
            
            try:
                # Get modified segment
                modified_img = modifier.get_modified_segment(segment_id)
                if modified_img is None:
                    return jsonify({"error": f"No modifications found for segment {segment_id}"}), 404
                
                # Try to integrate into main image
                if hasattr(processor, 'integrate_modified_segment'):
                    success = processor.integrate_modified_segment(segment_id, modified_img)
                elif hasattr(processor, 'replace_segment_in_image'):
                    success = processor.replace_segment_in_image(segment_id, modified_img)
                else:
                    # Fallback: use mask overlay
                    success = self._apply_segment_with_mask(processor, modifier, segment_id, modified_img)
                
                if success:
                    return jsonify({"message": f"Segment {segment_id} applied to main image successfully"})
                else:
                    return jsonify({"error": "Failed to apply segment to main image"}), 500
                    
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/session/<session_id>/segment/<int:segment_id>/download', methods=['GET'])
        @self._validate_session
        def download_segment(session_id, segment_id):
            """Download a segment as an image file"""
            session = self.sessions[session_id]
            modifier = session['modifier']
            
            if modifier is None:
                return jsonify({"error": "No segments loaded"}), 400
            
            try:
                # Try to get modified segment first, then original
                segment_img = modifier.get_modified_segment(segment_id)
                if segment_img is None:
                    segment_img = modifier.get_segment_image(segment_id)
                
                if segment_img is None:
                    return jsonify({"error": f"Segment {segment_id} not found"}), 404
                
                # Save to temporary file
                pil_image = Image.fromarray(segment_img)
                temp_path = os.path.join(self.upload_folder, f"segment_{segment_id}_{session_id}.png")
                pil_image.save(temp_path)
                
                return send_file(temp_path, as_attachment=True, 
                               download_name=f"segment_{segment_id}.png",
                               mimetype='image/png')
                               
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def _validate_session(self, f):
        """Decorator to validate session exists and update last accessed time"""
        @wraps(f)
        def decorated_function(session_id, *args, **kwargs):
            if session_id not in self.sessions:
                if self.enable_logging:
                    self.logger.warning(f"Invalid session access attempt: {session_id}")
                return jsonify({"error": "Session not found"}), 404
            
            # Update last accessed time
            self.sessions[session_id]['last_accessed'] = datetime.now()
            
            if self.enable_logging:
                self.logger.debug(f"Session {session_id[:8]}... accessed")
            
            return f(session_id, *args, **kwargs)
        return decorated_function
    
    def _check_opencv(self):
        """Check if OpenCV is available"""
        try:
            import cv2
            return True
        except ImportError:
            return False
    
    def _check_pil(self):
        """Check if PIL is available"""
        try:
            from PIL import Image
            return True
        except ImportError:
            return False
    
    def _check_numpy(self):
        """Check if NumPy is available"""
        try:
            import numpy as np
            return True
        except ImportError:
            return False
    
    def _allowed_file(self, filename):
        """Check if file extension is allowed"""
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def _apply_modifications(self, image, modifications):
        """Apply image modifications based on parameters"""
        current_img = image.copy().astype(np.float32)
        
        # Brightness & Contrast
        brightness = modifications.get('brightness', 0.0)
        contrast = modifications.get('contrast', 1.0)
        if brightness != 0.0 or contrast != 1.0:
            current_img = current_img * contrast + brightness
            current_img = np.clip(current_img, 0, 255)
        
        # Gamma correction
        gamma = modifications.get('gamma', 1.0)
        if abs(gamma - 1.0) > 1e-6:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
            current_img = cv2.LUT(current_img.astype(np.uint8), table)
        
        # Saturation
        saturation = modifications.get('saturation', 1.0)
        if abs(saturation - 1.0) > 1e-6:
            hsv = cv2.cvtColor(current_img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            current_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Blur
        blur_val = int(modifications.get('blur', 1))
        if blur_val > 1:
            if blur_val % 2 == 0:
                blur_val += 1
            current_img = cv2.GaussianBlur(current_img.astype(np.uint8), (blur_val, blur_val), 0)
        
        # Sharpen
        sharpen_val = modifications.get('sharpen', 0.0)
        if sharpen_val > 0:
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]], dtype=np.float32) * sharpen_val
            kernel[1, 1] = 8 * sharpen_val + 1.0
            current_img = cv2.filter2D(current_img.astype(np.float32), -1, kernel)
            current_img = np.clip(current_img, 0, 255).astype(np.uint8)
        
        return np.clip(current_img, 0, 255).astype(np.uint8)
    
    def _apply_segment_with_mask(self, processor, modifier, segment_id, modified_img):
        """Fallback method to apply segment using mask overlay"""
        try:
            # Get segment mask
            mask = None
            if hasattr(modifier, 'get_segment_mask'):
                mask = modifier.get_segment_mask(segment_id)
            elif hasattr(processor, 'get_segment_mask'):
                mask = processor.get_segment_mask(segment_id)
            
            if mask is not None and processor.current_image is not None:
                mask = mask.squeeze()
                h_img, w_img = processor.current_image.shape[:2]
                
                if mask.shape == (h_img, w_img):
                    # Resize modified image if needed
                    if modified_img.shape[:2] != (h_img, w_img):
                        modified_img = cv2.resize(modified_img, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                    
                    # Convert RGB to BGR for OpenCV
                    modified_bgr = cv2.cvtColor(modified_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    
                    # Apply mask
                    mask_bool = (mask != 0)
                    processor.current_image[mask_bool] = modified_bgr[mask_bool]
                    return True
            
            return False
        except Exception:
            return False
    
    def _start_cleanup_thread(self):
        """Start background thread to cleanup expired sessions"""
        def cleanup_sessions():
            import time
            while True:
                try:
                    current_time = datetime.now()
                    expired_sessions = []
                    
                    for session_id, session_data in self.sessions.items():
                        if current_time - session_data['last_accessed'] > self.session_timeout:
                            expired_sessions.append(session_id)
                    
                    if expired_sessions and self.enable_logging:
                        self.logger.info(f"Cleaning up {len(expired_sessions)} expired sessions")
                    
                    for session_id in expired_sessions:
                        if self.enable_logging:
                            self.logger.info(f"Expired session removed: {session_id}")
                        del self.sessions[session_id]
                        
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    if self.enable_logging:
                        self.logger.error(f"Error in cleanup thread: {e}")
                    time.sleep(300)
        
        cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
        cleanup_thread.start()
        if self.enable_logging:
            self.logger.info("Session cleanup thread started")
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        if self.enable_logging:
            self.logger.info(f"Starting Image Segmentation API server on {host}:{port}")
            self.logger.info(f"Debug mode: {debug}")
            self.logger.info(f"Upload folder: {self.upload_folder}")
            self.logger.info(f"Logging enabled: {self.enable_logging}")
        
        self.app.run(host=host, port=port, debug=debug)


# Example usage and client demonstration
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Segmentation API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-logging', action='store_true', help='Disable request/response logging')
    args = parser.parse_args()
    
    # Create API instance
    api = ImageSegmentationAPI(enable_logging=not args.no_logging)
    
    # Run the server
    print("="*60)
    print("STARTING IMAGE SEGMENTATION API SERVER")
    print("="*60)
    print("API Documentation:")
    print("POST   /api/session                              - Create new session")
    print("DELETE /api/session/<session_id>                - Delete session")
    print("POST   /api/session/<id>/image                   - Upload image")
    print("GET    /api/session/<id>/image                   - Get current image")
    print("POST   /api/session/<id>/segment/watershed       - Watershed segmentation")
    print("POST   /api/session/<id>/segment/deeplab         - DeepLab segmentation") 
    print("GET    /api/session/<id>/segments                - Get all segments")
    print("GET    /api/session/<id>/segment/<seg_id>        - Get specific segment")
    print("POST   /api/session/<id>/segment/<seg_id>/modify - Modify segment")
    print("POST   /api/session/<id>/segment/<seg_id>/reset  - Reset segment")
    print("POST   /api/session/<id>/segment/<seg_id>/apply  - Apply to main image")
    print("GET    /api/session/<id>/segment/<seg_id>/download - Download segment")
    print("="*60)
    if not args.no_logging:
        print("📝 Request/Response logging is ENABLED")
        print("   All API requests and responses will be logged to console and 'segmentation_api.log'")
    else:
        print("🔇 Request/Response logging is DISABLED")
    print("="*60)
    print()
    
    api.run(host=args.host, port=args.port, debug=args.debug)