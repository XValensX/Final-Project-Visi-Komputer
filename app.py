from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import torch
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
RESULTS_FOLDER = './static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

model_path = r"./yolov5/runs/train/scaled_down_run/weights/best.pt"
model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    if file:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(uploaded_image_path)


        results_folder_name = f"result_{timestamp}"
        result_subdir = os.path.join(app.config['RESULTS_FOLDER'], results_folder_name)
        os.makedirs(result_subdir, exist_ok=True)

        results = model(uploaded_image_path)
        results.save(save_dir=result_subdir, exist_ok=True) 


        detected_images = [f"/static/results/{results_folder_name}/{img}" for img in os.listdir(result_subdir) if img.endswith(('jpg', 'png'))]
        if not detected_images:
            return "No detections were made. Please try a different image.", 400

        return render_template('result.html', detected_images=detected_images)

if __name__ == '__main__':
    app.run(debug=True)

