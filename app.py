# app.py

import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import torch

# --- Local Imports ---
from garden_analyzer import run_full_analysis
from unet_model import UNet

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# --- App Initialization ---
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('predictions', exist_ok=True)

# --- Load Model (do this once on startup) ---
print("Loading model...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "unet_model_output/unet_balcony_best.pth"
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✔️ Model loaded successfully on {DEVICE}.")
except FileNotFoundError:
    print(f"❌ FATAL ERROR: Model file not found at {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load model. Error: {e}")
    model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predictions/<filename>')
def serve_prediction_image(filename):
    """Serves the generated overlay images."""
    return send_from_directory('predictions', filename)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """The main API endpoint for analyzing an uploaded image."""
    if model is None:
        return jsonify({"error": "Model is not loaded. Cannot process request."}), 500
        
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    language = request.form.get('language', 'english')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"Processing {filepath} for language: {language}")
        
        # Run the full analysis
        try:
            results = run_full_analysis(filepath, language, model, DEVICE)
            return jsonify(results)
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            return jsonify({"error": "An internal error occurred during analysis."}), 500

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/products')
def products_page():
    """Renders the products page."""
    return render_template('products.html')

@app.route('/community')
def community_page():
    """Renders the community page."""
    return render_template('community.html')

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)