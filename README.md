# Digit_Recognizer
🧠 Digit Recognizer Streamlit App – Code Explanation
This Python app allows users to draw digits (0–9) on a canvas and uses a pre-trained deep learning model to recognize and classify the digit in real time. Built with Streamlit, it offers an intuitive interface and is ideal for demonstrating handwritten digit recognition using the MNIST dataset.

📁 Code Breakdown
1. Import Libraries
import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
numpy and cv2: For image processing and array manipulation.

streamlit and st_canvas: Used to build the interactive UI and drawing canvas.

tensorflow.keras: To load the pre-trained digit classification model.

2. App Title
st.title('🧠 Digit Recognizer Streamlit App')
Sets the title of the Streamlit web application.

3. Load the Pre-trained Model
model_path = "C:\\Users\\Lenovo\\Downloads\\Best_9927.h5"
model = load_model(model_path, compile=False)
Loads the pre-trained Keras model (typically trained on MNIST).

compile=False avoids warnings about missing optimizer state and metrics, which aren't needed for inference.

4. Canvas Settings and UI
SIZE = 280
draw_mode = st.checkbox('Enable Drawing Mode', True)
Sets the canvas size to 280x280 pixels.

Allows users to toggle drawing mode on or off.

5. Draw Canvas UI
canvas_result = st_canvas(...)
Renders an interactive canvas for drawing digits.

stroke_color="#FFFFFF" draws in white on a black background, mimicking MNIST-style input.

6. Display Drawn Image Preview
if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    preview_img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.markdown("### ✏️ Drawn Image Preview")
    st.image(preview_img)
When the user draws something:
The image is downscaled to 28x28 (to match MNIST input size).Then upscaled again for preview display. This helps visualize the image the model will receive.

7. Prediction Button
if st.button('🔍 Predict'):
Waits for the user to click Predict before making a classification.

8. Preprocessing for Model
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
normalized = gray_img / 255.0
input_tensor = normalized.reshape(1, 28, 28, 1)
Converts the image to grayscale (MNIST format).

Normalizes pixel values between 0 and 1. Reshapes it to a 4D tensor as required by Keras CNNs: (batch_size, height, width, channels).

9. Model Prediction
prediction = model.predict(input_tensor, verbose=0)
predicted_digit = np.argmax(prediction[0])

Predicts the digit using the loaded CNN model.

np.argmax() gets the digit with the highest probability score.

11. Display Prediction & Probability Chart
st.success(f"✅ Predicted Digit: **{predicted_digit}**")
st.markdown("### 🔢 Prediction Probabilities")
st.bar_chart(prediction[0])
Displays the predicted digit.

Plots a bar chart of all digit probabilities (0 to 9).

✅ Output Example
After drawing a digit (e.g., 4), you'll see: A preview of the digit
The predicted result: Predicted Digit: 4

A probability distribution bar chart showing the confidence for each digit class

📝 Conclusion
This app is a real-time digit recognizer using deep learning, built with:
✅ Keras-trained CNN model
✅ Interactive Streamlit UI
✅ OpenCV for preprocessing


