import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "best_mobilenetv2_paddy.h5"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro"
]

# -----------------------
# APP INIT
# -----------------------
app = Flask(__name__)
CORS(app)

# -----------------------
# LOAD MODEL
# -----------------------
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# -----------------------
# ROUTES
# -----------------------
@app.route("/")
def home():
    return jsonify({"message": "Paddy Disease Classification API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    # Save temp image
    temp_path = "temp.jpg"
    image_file.save(temp_path)

    # Preprocess image
    img = load_img(temp_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    confidence = float(np.max(predictions))
    class_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[class_index]

    # Cleanup
    os.remove(temp_path)

    return jsonify({
        "predicted_disease": predicted_class,
        "confidence": round(confidence * 100, 2)
    })

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
