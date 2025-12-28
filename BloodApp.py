import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Dummy credentials
USERNAME = "123"
PASSWORD = "123"

# Load trained model
MODEL_PATH = "blood_group_model.h5"
model = load_model(MODEL_PATH)

IMG_HEIGHT = 64
IMG_WIDTH = 64
class_labels = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

def predict_blood_group(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class]

# --- LOGIN SYSTEM ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    # HTML styled login page
    login_html = """
    <div style="width:400px; margin:auto; padding:30px; border:1px solid #ccc; 
                border-radius:10px; background-color:#f9f9f9;">
        <h2 style="text-align:center; color:#333;">🔐 Login</h2>
        <p style="text-align:center; color:gray;">Enter your credentials to continue</p>
    </div>
    """
    st.markdown(login_html, unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful! 🎉")
        else:
            st.error("Invalid credentials. Please try again.")
else:
    # --- MAIN RESULT PAGE ---
    st.markdown(
        """
        <div style="text-align:center; padding:20px; background-color:#eef; border-radius:10px;">
            <h1 style="color:#2c3e50;">🩸 Blood Group Prediction</h1>
            <p style="color:#555;">Upload an image to predict the blood group using the trained model.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","bmp"])

    if uploaded_file is not None:
        temp_path = os.path.join("temp_image.png")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        result = predict_blood_group(temp_path)
        result_html = f"""
        <div style="margin-top:20px; padding:20px; border:2px solid #2ecc71; 
                    border-radius:10px; background-color:#eafbea; text-align:center;">
            <h2 style="color:#27ae60;">✅ Predicted Blood Group: {result}</h2>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.info("You have been logged out.")