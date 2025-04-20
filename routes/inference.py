from flask import Blueprint, request, jsonify
import os
from tensorflow.keras.models import load_model
from utils.image_processing import process_image
import numpy as np
import uuid  

interface_bp = Blueprint('interface', __name__)

MODEL_PATH = os.path.join("models", "brain_tumor_classifier.h5")
model = load_model(MODEL_PATH)

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(image_path):
    img = process_image(image_path)

    prediction = model.predict(img)

    predicted_class_index = np.argmax(prediction)

    predicted_class = class_names[predicted_class_index]
    probability = prediction[0][predicted_class_index]

    return predicted_class, probability

@interface_bp.route("/predict", methods=["POST"])
def predict_image():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "message": "No file was uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "message": "No file was selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False, "message": "Allowed file types are: png, jpg, jpeg"}), 400
        
        
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)
        image_path = os.path.join(upload_folder, filename)
        file.save(image_path)
        
        predicted_class, probability = predict(image_path)

        return jsonify({
            "success": True,
            "message": "Image classified successfully",
            "data": {
                "result": predicted_class,
                "confidence": float(probability)
            }
        }), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
