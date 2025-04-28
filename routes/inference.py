# from flask import Blueprint, request, jsonify
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model 
from utils.image_processing import process_image
# import numpy as np
# import uuid  

# interface_bp = Blueprint('interface', __name__)

# MODEL_PATH = os.path.join("models", "brain_tumor_classifier.h5")
# print(MODEL_PATH)
# model = load_model(MODEL_PATH, custom_objects={'InputLayer': tf.keras.layers.InputLayer})

# class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def predict(image_path):
#     img = process_image(image_path)

#     prediction = model.predict(img)

#     predicted_class_index = np.argmax(prediction)

#     predicted_class = class_names[predicted_class_index]
#     probability = prediction[0][predicted_class_index]

#     return predicted_class, probability

# @interface_bp.route("/predict", methods=["POST"])
# def predict_image():
#     try:
#         if "file" not in request.files:
#             return jsonify({"success": False, "message": "No file was uploaded"}), 400
        
#         file = request.files["file"]
#         if file.filename == "":
#             return jsonify({"success": False, "message": "No file was selected"}), 400

#         if not allowed_file(file.filename):
#             return jsonify({
#                 "success": False, "message": "Allowed file types are: png, jpg, jpeg"}), 400
        
        
#         filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
#         upload_folder = "uploads"
#         os.makedirs(upload_folder, exist_ok=True)
#         image_path = os.path.join(upload_folder, filename)
#         file.save(image_path)
        
#         predicted_class, probability = predict(image_path)

#         return jsonify({
#             "success": True,
#             "message": "Image classified successfully",
#             "data": {
#                 "result": predicted_class,
#                 "confidence": float(probability),
#                 "image_path": image_path
#             }
#         }), 200

#     except Exception as e:
#         return jsonify({"success": False, "message": str(e)}), 500
from flask import Blueprint, request, jsonify
from utils.image_processing import process_image
import os
import onnxruntime as ort
import numpy as np
import uuid
import logging
from werkzeug.utils import secure_filename

interface_bp = Blueprint('interface', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join("models", "brain_tumor_classifier.onnx")
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg' , "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Initialize ONNX session
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    input_details = ort_session.get_inputs()[0]
    logger.info(f"Loaded ONNX model. Input shape: {input_details.shape}")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise RuntimeError("Model initialization failed")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@interface_bp.route("/predict", methods=["POST"])
def predict_image():
    try:
        logger.info(f"Incoming request headers: {request.headers}")
        logger.info(f"Request files: {request.files}")

        if 'file' not in request.files:
            logger.error("No 'file' key in request.files")
            return jsonify({"success": False, "message": "No file part in request"}), 400

        file = request.files['file']
        
        if file.filename == '':
            logger.error("Empty filename submitted")
            return jsonify({"success": False, "message": "No selected file"}), 400

        if not allowed_file(file.filename):
            logger.error(f"Invalid file extension: {file.filename}")
            return jsonify({"success": False, "message": "Invalid file type"}), 400

        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        try:
            file.save(filepath)
            logger.info(f"File saved temporarily to {filepath}")

            # Process image
            img = process_image(filepath)
            logger.info(f"Processed image shape: {img.shape}")
            logger.info(f"Model expects: {input_details.shape}")

            
            # Verify spatial dimensions only (ignore batch size)
            if list(img.shape[1:]) != list(input_details.shape[1:]):
                logger.error(f"Shape mismatch. Got {img.shape[1:]}, needs {input_details.shape[1:]}")
                return jsonify({"success": False, "message": "Image processing error"}), 400

            # Ensure correct data type
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            
            # Run inference
            prediction = ort_session.run(None, {input_details.name: img})[0]
            class_index = np.argmax(prediction[0])
            confidence = float(prediction[0][class_index])

            return jsonify({
                "success": True,
                "data": {
                    "result": ['glioma', 'meningioma', 'notumor', 'pituitary'][class_index],
                    "confidence": confidence
                }
            })

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            return jsonify({"success": False, "message": "Processing error"}), 500

        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"Removed temporary file: {filepath}")
            except Exception as e:
                logger.error(f"File cleanup failed: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": "Internal server error"}), 500