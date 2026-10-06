import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Load Model
# -----------------------------
import os
import urllib.request
import tensorflow as tf

MODEL_PATH = "/tmp/wound_model.keras"

MODEL_URL = "https://github.com/nanomoonlabs002-art/Wound-Infection-Classification/releases/download/v1.0/wound_model.keras"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Wound Condition Classification",
    page_icon="🩹",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.block-container {
    padding-top: 3rem;
    padding-bottom: 0.5rem;
    max-width: 1000px;
}

.title {
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 2px;
}

.subtitle {
    text-align: center;
    font-size: 15px;
    color: #777;
    margin-bottom: 12px;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-top: 5px;
    margin-bottom: 5px;
}

.footer {
    text-align: center;
    color: #777;
    font-size: 11px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">🩹 Wound Condition Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Assisted Digital Image Classification System</div>',
    unsafe_allow_html=True
)

# -----------------------------
# About Project
# -----------------------------
with st.expander("ℹ️ About this Project", expanded=False):

    st.write(
        "This application uses a Convolutional Neural Network (CNN) "
        "to classify an uploaded wound image into two image classes:"
    )

    st.markdown(
        "**1. Ulcer**  \n"
        "**2. Normal / Healthy Skin**"
    )

    st.caption(
        "Developed for educational and research purposes."
    )

# -----------------------------
# Upload Section
# -----------------------------
st.markdown(
    '<div class="section-title">📤 Upload Wound Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# -----------------------------
# Image + Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # -------------------------
    # Image Preview
    # -------------------------
    with col1:

        st.markdown("### 🖼️ Image Preview")

        st.image(
            image,
            caption="Uploaded Image",
            width=280
        )

    # -------------------------
    # Prediction
    # -------------------------
    with col2:

        st.markdown("### 📊 Prediction")

        if st.button(
            "🔍 Analyze Image",
            use_container_width=True
        ):

            with st.spinner("Analyzing..."):

                img = image.resize((128, 128))

                img_array = np.array(img) / 255.0

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                prediction = model.predict(
                    img_array,
                    verbose=0
                )[0][0]

            # -------------------------
            # Classification
            # -------------------------
            if prediction >= 0.5:

                result = "Normal / Healthy Skin"
                confidence = prediction * 100

            else:

                result = "Ulcer"
                confidence = (1 - prediction) * 100

            # -------------------------
            # Result
            # -------------------------
            if result == "Ulcer":

                st.error(
                    f"🔴 **{result}**"
                )

            else:

                st.success(
                    f"🟢 **{result}**"
                )

            st.write(
                f"**Confidence: {confidence:.2f}%**"
            )

            st.progress(
                min(int(confidence), 100)
            )

# -----------------------------
# Disclaimer
# -----------------------------
st.caption(
    "⚠️ Educational and research purposes only. "
    "Not a medical diagnostic tool."
)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    '<div class="footer">'
    'AI-Assisted Wound Condition Classification System<br>'
    'CNN-Based Digital Image Processing'
    '</div>',
    unsafe_allow_html=True
)
