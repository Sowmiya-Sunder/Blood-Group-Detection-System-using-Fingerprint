from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('blood_group_predictor.keras')

# Blood group labels
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Preprocess uploaded image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (128, 128))            # Resize to 128x128
    image = image / 255.0                            # Normalize to [0, 1]
    image = np.reshape(image, (1, 128, 128, 1))      # Reshape for model
    return image

# Route to upload and predict blood group
@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        file = request.files['image']
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Preprocess image
        processed_image = preprocess_image(image)

        # Predict blood group
        prediction = model.predict(processed_image)
        blood_group = BLOOD_GROUPS[np.argmax(prediction)]

        return jsonify({'blood_group': blood_group})

    return render_template('index.html')

# Optional: Handle favicon.ico requests
@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, port=5001)



