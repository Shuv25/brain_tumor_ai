from flask import Blueprint, request, jsonify
import os
from tensorflow.keras.models import load_model
from utils.image_processing import process_image  
import numpy as np

interface_bp = Blueprint('interface', __name__)


MODEL_PATH = os.path.join("models", "brain_tumor_classifier.h5")
model = load_model(MODEL_PATH)

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

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
            return jsonify({"error": "No file part"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)
        image_path = os.path.join(upload_folder, file.filename)
        file.save(image_path)
        
        predicted_class, probability = predict(image_path)

        return jsonify({
            "result": predicted_class,
            "confidence": float(probability)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
