from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile

from ImageSegmentationProcessor import ImageSegmentationProcessor

class ImageSegmentationAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.processor = ImageSegmentationProcessor()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/upload', methods=['POST'])
        def upload_image():
            if 'image' not in request.files:
                return jsonify({"success": False, "error": "No image file provided"}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({"success": False, "error": "No file selected"}), 400
            
            # Save temporarily and process
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                file.save(tmp_file.name)
                result = self.processor.load_image(tmp_file.name)
                os.unlink(tmp_file.name)
            
            return jsonify(result)
        
        @self.app.route('/segment/kmeans', methods=['POST'])
        def segment_kmeans():
            data = request.get_json()
            k = data.get('k', 5)
            
            result = self.processor.segment_kmeans(k)
            return jsonify(result)
        
        @self.app.route('/segment/watershed', methods=['POST'])
        def segment_watershed():
            result = self.processor.segment_watershed()
            return jsonify(result)
        
        @self.app.route('/segments', methods=['GET'])
        def get_segments():
            result = self.processor.get_all_segments()
            return jsonify(result)
        
        @self.app.route('/segment/<int:segment_id>', methods=['GET'])
        def get_segment(segment_id):
            result = self.processor.get_segment(segment_id)
            return jsonify(result)
    
    def run(self, host='localhost', port=5000, debug=True):
        """Run the API server"""
        self.app.run(host=host, port=port, debug=debug)

