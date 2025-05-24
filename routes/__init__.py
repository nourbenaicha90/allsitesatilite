# __init__.py أو app.py
import os
from flask import Flask

app = Flask(__name__)
app.secret_key = 'supersecretkey123'  # Add your secret key

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(BASE_DIR, 'static', 'results')
