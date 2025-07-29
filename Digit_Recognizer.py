import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# Title
st.title('🧠 Digit Recognizer Streamlit App')

# Load the trained model (with compile=False to avoid optimizer warnings)
model_path = "C:\\Users\\Lenovo\\Downloads\\Best_9927.h5"
model = load_model(model_path, compile=False)

# Canvas settings
SIZE = 280  # Size of the canvas (280x280)
draw_mode = st.checkbox('Enable Drawing Mode', True)

# Create a drawable canvas
canvas_result = st_canvas(
    fill_color="#000000",       # Not used in free draw
    stroke_width=20,            # Thickness of the pen
    stroke_color="#FFFFFF",     # White ink
    background_color="#000000", # Black background
    width=SIZE,
    height=SIZE,
    drawing_mode='freedraw' if draw_mode else 'transform',
    key="canvas",
)

# Display the drawn image (after resizing for clarity)
if canvas_result.image_data is not None:
    # Resize the image to 28x28 for model input
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    preview_img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

    st.markdown("### ✏️ Drawn Image Preview")
    st.image(preview_img)

# Predict when button is pressed
if st.button('🔍 Predict'):
    if canvas_result.image_data is not None:
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalize to [0,1]
        normalized = gray_img / 255.0
        
        # Reshape to match model input: (1, 28, 28, 1)
        input_tensor = normalized.reshape(1, 28, 28, 1)

        # Make prediction
        prediction = model.predict(input_tensor, verbose=0)
        predicted_digit = np.argmax(prediction[0])

        # Show results
        st.success(f"✅ Predicted Digit: **{predicted_digit}**")
        st.markdown("### 🔢 Prediction Probabilities")
        st.bar_chart(prediction[0])
    else:
        st.warning("Please draw a digit first before predicting.")
