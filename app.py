from flask import Flask, request, jsonify, render_template
from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import LabelEncoder
import cv2
import pickle
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.transform import resize
from skimage.filters import sobel
import os

app = Flask(__name__)

# Define the model directory
model_dir = 'models'

# Load the trained models from the models folder
with open(os.path.join(model_dir, 'rfc_wheat.pkl'), 'rb') as f:
    model_wheat = pickle.load(f)
    
with open(os.path.join(model_dir, 'rfc_cotton.pkl'), 'rb') as f:
    model_cotton = pickle.load(f)
    
with open(os.path.join(model_dir, 'rfc_sugercane.pkl'), 'rb') as f:
    model_sugarcane = pickle.load(f)
    
with open(os.path.join(model_dir, 'svm_rice.pkl'), 'rb') as f:
    model_rice = pickle.load(f)

def extract_features(image, size=(64, 64)):
    try:
        # Resize image
        resized_image = resize(image, size, anti_aliasing=True)
            
        # Convert to grayscale
        gray_image = rgb2gray(resized_image)
        
        # HOG features
        hog_features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
        
        # LBP features
        lbp_features = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp_features.ravel(), bins=np.arange(0, 11), range=(0, 10))
        lbp_hist = lbp_hist.astype("float32")
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)
        
        # Sobel edge detection
        sobel_edges = sobel(gray_image)
        sobel_hist, _ = np.histogram(sobel_edges.ravel(), bins=256, range=(0, 1))
        sobel_hist = sobel_hist.astype("float32")
        sobel_hist = sobel_hist / (sobel_hist.sum() + 1e-7)
        
        # Color histograms in different color spaces (RGB, HSV, Lab)
        rgb_hist = np.histogram(resized_image.flatten(), bins=256, range=[0, 1])[0]
        hsv_image = rgb2hsv(resized_image)
        hsv_hist = np.histogram(hsv_image.flatten(), bins=256, range=[0, 1])[0]
        lab_image = rgb2lab(resized_image)
        lab_hist = np.histogram(lab_image.flatten(), bins=256, range=[0, 100])[0]
        
        rgb_hist = rgb_hist.astype("float32")
        hsv_hist = hsv_hist.astype("float32")
        lab_hist = lab_hist.astype("float32")
        
        rgb_hist = rgb_hist / (rgb_hist.sum() + 1e-7)
        hsv_hist = hsv_hist / (hsv_hist.sum() + 1e-7)
        lab_hist = lab_hist / (lab_hist.sum() + 1e-7)
        
        # Combine features
        combined_features = np.concatenate([hog_features, lbp_hist, sobel_hist, rgb_hist, hsv_hist, lab_hist])
        
        return combined_features, None
    except Exception as e:
        return None, str(e)

def process_image(file):
    try:
        if file.filename == '':
            return None, 'No selected file'
        
        # Convert file to numpy array (image)
        file_bytes = file.read()
        image = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        if image is None:
            return None, 'Unable to read the image'
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess the image
        preprocessed_image, error = extract_features(image)
        if error:
            return None, f'Feature extraction failed: {error}'
            
        return preprocessed_image, None
    except Exception as e:
        return None, f'Image processing failed: {str(e)}'

@app.route('/')
def index():
    return render_template('index.html')

# Define class mappings for each crop
wheat_classes = {0: 'Healthy', 1: 'Leaf Rust', 2: 'Stripe Rust', 3: 'Yellow Rust'}
cotton_classes = {0: 'Bacterial Blight', 1: 'Curl Virus', 2: 'Fussarium Wilt', 3: 'Healthy'}
sugarcane_classes = {0: 'Healthy', 1: 'Mosaic', 2: 'Red Rot', 3: 'Rust', 4: 'Yellow'}
rice_classes = {0: 'Bacterial Light', 1: 'Blast', 2: 'Brown Spot', 3: 'Healthy', 4: 'Tungro'}

# Define separate endpoints for each crop
@app.route('/predict/wheat', methods=['POST'])
def predict_wheat():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        features, error = process_image(file)
        if error:
            return jsonify({'error': error}), 400
        
        prediction = model_wheat.predict([features])[0]
        prediction_text = wheat_classes[prediction]
        return jsonify({'prediction': prediction_text})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/cotton', methods=['POST'])
def predict_cotton():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        features, error = process_image(file)
        if error:
            return jsonify({'error': error}), 400
        
        prediction = model_cotton.predict([features])[0]
        prediction_text = cotton_classes[prediction]
        return jsonify({'prediction': prediction_text})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/sugarcane', methods=['POST'])
def predict_sugarcane():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        features, error = process_image(file)
        if error:
            return jsonify({'error': error}), 400
        
        prediction = model_sugarcane.predict([features])[0]
        prediction_text = sugarcane_classes[prediction]
        return jsonify({'prediction': prediction_text})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/rice', methods=['POST'])
def predict_rice():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        features, error = process_image(file)
        if error:
            return jsonify({'error': error}), 400
        
        prediction = model_rice.predict([features])[0]
        prediction_text = rice_classes[prediction]
        return jsonify({'prediction': prediction_text})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)