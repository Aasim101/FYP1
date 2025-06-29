import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
from model_utils import load_models, preprocess_image, predict, generate_gradcam

st.title("🧠 Brain Tumor Classification & Grad-CAM Visualization")
st.write("Upload an MRI image to detect brain tumor and visualize important regions via Grad-CAM with a heatmap colormap.")

uploaded = st.file_uploader("Choose an MRI image...", type=["jpg","jpeg","png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Preprocess image
    input_tensor = preprocess_image(image)

    # Load ensemble model (cached)
    model = load_models()

    # Predict tumor type
    pred_label, probs = predict(model, input_tensor)
    st.markdown(f"**Predicted Tumor Type:** {pred_label}")
    st.bar_chart(probs)

    # Generate Grad-CAM
    heatmap = generate_gradcam(model, input_tensor, class_idx=list(probs.keys()).index(pred_label))
    # Convert to uint8 and apply colormap
    heatmap_uint = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_pil = Image.fromarray(heatmap_color).resize(image.size)

    # Blend with original
    overlay = Image.blend(image.convert("RGBA"), heatmap_pil.convert("RGBA"), alpha=0.5)
    st.image(overlay, caption="Grad-CAM Overlay (Jet Colormap)", use_container_width=True)
