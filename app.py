from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os
import gdown

load_dotenv()

from routes.inference import interface_bp
from routes.report import report_bp
from routes.chatbot import chatbot_bp
from routes.cha_with_pdf import chat_with_pdf_bp

app = Flask(__name__)


CORS(app)


app.register_blueprint(interface_bp, url_prefix="/api/interface")
app.register_blueprint(report_bp, url_prefix="/api/report")
app.register_blueprint(chatbot_bp, url_prefix="/api/chatbot")
app.register_blueprint(chat_with_pdf_bp,url_prefix="/api/chat_with_pdf")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_classifier.h5")
GDRIVE_URL = os.getenv("GDRIVE_MODEL_URL")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    if GDRIVE_URL:
        print("Downloading model from Google Drive...")
        try:
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading the model: {str(e)}")
            raise
    else:
        raise ValueError("GDRIVE_MODEL_URL not found in .env file!")

@app.route("/")
def home():
    return {"message": "Brain Tumor Backend API is running."}

if __name__ == '__main__':
    app.run(debug=True)
