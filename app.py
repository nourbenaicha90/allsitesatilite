import os
import secrets
import tifffile
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Secure random secret key

# Folders for uploads and images
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'tif', 'tiff'}

# Load your pretrained XGBoost model (make sure xgb_model.h5 exists)
model = xgb.XGBClassifier()
model.load_model("xgb_model.h5")

known_classes = ['non-agricultural', 'agricultural']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_image(img):
    # Example feature extraction: mean of each band
    features = np.mean(img, axis=(0, 1)).astype(np.float32).reshape(1, -1)
    preds = model.predict_proba(features)
    class_idx = int(np.argmax(preds))
    confidence = preds[0][class_idx] * 100
    predicted_class = known_classes[class_idx]
    return f"{predicted_class} ({confidence:.2f}%)"

def save_natural_rgb_image(img, save_path):
    # Assuming band indices: Red=3, Green=2, Blue=1 (adjust if needed)
    red = img[:, :, 3]
    green = img[:, :, 2]
    blue = img[:, :, 1]

    def normalize(b):
        return (b - b.min()) / (b.max() - b.min() + 1e-6)

    rgb = np.dstack([normalize(red), normalize(green), normalize(blue)])
    plt.imsave(save_path, rgb)

def calculate_green_percentage(img):
    red = img[:, :, 3].astype(np.float32)
    nir = img[:, :, 7].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    green_pixels = (ndvi > 0.2).sum()
    total_pixels = ndvi.size
    return (green_pixels / total_pixels) * 100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/logout')
def logout():
    return render_template('logout.html')

@app.route('/create_account')
def create_account():
    return render_template('create_account.html')

@app.route('/newpage')
def newpage():
    return render_template('newpage.html')

@app.route('/analyse')
def analyse_page():
    return render_template('Analyse.html')

@app.route('/analyse_result')
def analyse_result_page():
    return render_template('analyse_result.html')

@app.route('/important', methods=["GET", "POST"])
def important_class():
    result = None
    img_info = {}
    rgb_image_path = None
    green_percent = None

    if request.method == "POST":
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("📁 يرجى اختيار ملف .tif")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("❌ نوع الملف غير مدعوم. يرجى رفع ملف بصيغة .tif أو .tiff فقط.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            img = tifffile.imread(filepath)
        except Exception as e:
            flash(f"❌ خطأ في تحميل الصورة: {str(e)}")
            return redirect(request.url)

        if img.ndim != 3:
            flash("❌ الصورة يجب أن تكون ثلاثية الأبعاد (ارتفاع، عرض، عدد القنوات).")
            return redirect(request.url)

        result = classify_image(img)
        green_percent = calculate_green_percentage(img)

        # Image info
        img_info['height'], img_info['width'], img_info['bands'] = img.shape

        # Save natural image preview
        rgb_filename = filename.rsplit('.', 1)[0] + "_natural.png"
        rgb_save_path = os.path.join(IMAGE_FOLDER, rgb_filename)
        save_natural_rgb_image(img, rgb_save_path)
        rgb_image_path = f"images/{rgb_filename}"

    return render_template("important_class.html",
                           result=result,
                           img_info=img_info,
                           rgb_image=rgb_image_path,
                           green_percent=green_percent)

@app.route('/segment')
def segment():
    return render_template('Segment.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True, port=5001)
