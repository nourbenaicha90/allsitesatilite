# AllSite\routes\routeAnalyse.py
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app
from werkzeug.utils import secure_filename

route_analyse = Blueprint('route_analyse', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'tif', 'tiff', 'png', 'jpg', 'jpeg'}

def calculate_vegetation_indices(red, nir, blue=None, green=None):
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1, 1)

    evi = None
    if blue is not None:
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)

    L = 0.5
    savi = (1 + L) * (nir - red) / (nir + red + L)

    ndwi = None
    if green is not None:
        ndwi = (green - nir) / (green + nir)

    return ndvi, evi, savi, ndwi

def save_visualization(data, path, title, cmap='RdYlGn', vmin=None, vmax=None):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

@route_analyse.route('/analyse_model')
def analyse_model():
    return render_template('Analyse.html')

@route_analyse.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('route_analyse.analyse_model'))

    file = request.files['image']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('route_analyse.analyse_model'))

    if not allowed_file(file.filename):
        flash('Invalid file type.', 'error')
        return redirect(url_for('route_analyse.analyse_model'))

    try:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        with rasterio.open(upload_path) as src:
            count = src.count
            blue = src.read(1).astype('float32') if count >= 1 else None
            green = src.read(2).astype('float32') if count >= 2 else None
            red = src.read(3).astype('float32') if count >= 3 else None
            nir = src.read(4).astype('float32') if count >= 4 else None

            if red is None or nir is None:
                flash('At least red and NIR bands required.', 'error')
                return redirect(url_for('route_analyse.analyse_model'))

            ndvi, evi, savi, ndwi = calculate_vegetation_indices(red, nir, blue, green)
            mean_ndvi = float(np.nanmean(ndvi))

            if mean_ndvi >= 0.66:
                health_status = "نبات صحي جدًا"
            elif mean_ndvi >= 0.33:
                health_status = "نبات متوسط الصحة"
            elif mean_ndvi >= 0.0:
                health_status = "نبات مريض"
            else:
                health_status = "نبات ميت أو جسم غير حي"

            result_folder = current_app.config['RESULT_FOLDER']
            os.makedirs(result_folder, exist_ok=True)

            save_visualization(ndvi, os.path.join(result_folder, 'ndvi_result.png'), 'NDVI Index', vmin=-1, vmax=1)
            classified_ndvi = np.zeros_like(ndvi)
            classified_ndvi[ndvi < 0.2] = 0
            classified_ndvi[(ndvi >= 0.2) & (ndvi < 0.5)] = 1
            classified_ndvi[ndvi >= 0.5] = 2
            save_visualization(classified_ndvi, os.path.join(result_folder, 'ndvi_classified.png'), 'NDVI Classified', vmin=0, vmax=2)

            if count >= 3:
                rgb = np.dstack([src.read(3), src.read(2), src.read(1)])
                rgb_norm = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6) * 255).astype(np.uint8)
                save_visualization(rgb_norm, os.path.join(result_folder, 'ndvi_reference.png'), 'Reference RGB')

            image_files = [
                'ndvi_result.png',
                'ndvi_classified.png',
                'ndvi_reference.png'
            ]

            # return render_template('analyse_result.html', image_files=image_files, mean_ndvi=round(mean_ndvi, 2), count=count, health_status=health_status)
            return render_template(
            'Analyse.html',
            image_files=image_files,
            mean_ndvi=round(mean_ndvi, 2),
            count=count,
            health_status=health_status
            )

    



    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('route_analyse.analyse_model'))
