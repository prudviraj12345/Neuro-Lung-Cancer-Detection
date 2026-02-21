from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import matplotlib.pyplot as plt
import cv2
import base64
# ... (your other imports like Flask, numpy, etc.) ...
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow frontend and backend to communicate
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return send_from_directory('.', 'index.html')

# Load your trained model
try:
    model = tf.keras.models.load_model("lung_cancer_model.h5")
    class_names = ['Benign', 'Malignant', 'Normal']
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Define the prediction endpoint
# app.py

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image file
    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image: {e}'}), 400

    # Preprocess the image for the model
    img_size = 128  # Should match your model's input size
    image_resized = image.resize((img_size, img_size))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(np.max(prediction))

    # Generate Grad-CAM heatmap
    score = CategoricalScore([predicted_class_index])
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    cam = gradcam(score, img_array, penultimate_layer=-1)

    # Use matplotlib to create the heatmap visualization
    heatmap = np.uint8(plt.cm.jet(cam[0])[..., :3] * 255)

    # Superimpose the heatmap on the original image
    original_image_cv = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(heatmap, 0.4, original_image_cv, 0.6, 0)

    # Encode the final image to a base64 string to send in JSON
    retval, buffer = cv2.imencode('.png', superimposed_img)
    heatmap_str = base64.b64encode(buffer).decode('utf-8')

    # Return the prediction AND the heatmap
    return jsonify({
        'prediction': predicted_class_name,
        'confidence': confidence,
        'heatmap': heatmap_str
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)