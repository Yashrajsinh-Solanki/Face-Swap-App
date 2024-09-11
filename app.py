from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import importlib.util
# import os
import io
# from PIL import Image


def load_face_swap_function():
    module_name = "model.Swapping_model"
    function_name = "face_swap"
    
    # Find the module spec
    spec = importlib.util.find_spec(module_name)
    
    if spec is None:
        raise ImportError(f"Cannot find module {module_name}")
    
    # Load the module
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the face_swap function
    face_swap_func = getattr(module, function_name, None)
    
    if face_swap_func is None:
        raise ImportError(f"Cannot find function {function_name} in module {module_name}")
    
    return face_swap_func

# Load the function once when the app starts
face_swap = load_face_swap_function()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/swap-faces', methods=['POST'])
def swap_faces():
    try:
        # Get images from the request
        img1_file = request.files.get('image1')
        img2_file = request.files.get('image2')
        
        if not img1_file or not img2_file:
            return jsonify({'error': 'Both images are required!'}), 400
        
        # Read images using OpenCV
        img1 = cv2.imdecode(np.frombuffer(img1_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(img2_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Perform face swap
        swapped_image = face_swap(img1, img2)
        
        if swapped_image is None:
            return jsonify({'error': 'Face swap failed.'}), 500
        
        # Convert result to JPEG
        _, buffer = cv2.imencode('.jpg', swapped_image)
        img_io = io.BytesIO(buffer)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
